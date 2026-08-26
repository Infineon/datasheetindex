"""Per-cell token usage + cost rollup across chamber benchmark models.

Reads:
  - `archive/latest_chamber.json`           (Sonnet, default model)
  - `archive/latest_chamber.gpt-5.1.json`
  - `archive/latest_chamber.qwen3.5-27b.json`
  - `archive/latest_traces.jsonl`           (agentic per-cell usage, Sonnet)
  - `archive/latest_traces.gpt-5.1.jsonl`
  - `archive/latest_traces.qwen3.5-27b.jsonl`

The result JSONs carry baseline `usage` rolled up by the test runner; agentic
usage may or may not be in the results yet (older runs predate the rollup
patch). Where the results file lacks agentic usage, we fall back to the
trace files and aggregate `final_output` + one-per-turn `tool_call`.

Pricing is documented inline. Sonnet uses Anthropic's public list; gpt-5.1
uses OpenAI's public list (the Azure passthrough on the internal gateway
may differ); qwen3.5-27b uses a hosted-vLLM representative because the
internal gateway's per-token cost is project-internal. All figures are
labelled with their pricing source.

Usage:
    uv run python scripts/chamber_cost_summary.py
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from chamberbench.claimsio import archive_dir

RESULTS_DIR = archive_dir()


# (input $/M, output $/M, cache_read $/M, source label)
PRICING: dict[str, tuple[float, float, float, str]] = {
    "claudesonnet4.6": (3.00, 15.00, 0.30, "Anthropic public list"),
    "gpt-5.1": (1.25, 10.00, 0.125, "OpenAI public list (Azure gateway varies)"),
    "qwen3.5-27b": (0.20, 0.60, 0.20, "hosted-vLLM 27B representative"),
}

RESULT_FILES: dict[str, str] = {
    "claudesonnet4.6": "latest_chamber.json",
    "gpt-5.1": "latest_chamber.gpt-5.1.json",
    "qwen3.5-27b": "latest_chamber.qwen3.5-27b.json",
}

TRACE_FILES: dict[str, str] = {
    "claudesonnet4.6": "latest_traces.jsonl",
    "gpt-5.1": "latest_traces.gpt-5.1.jsonl",
    "qwen3.5-27b": "latest_traces.qwen3.5-27b.jsonl",
}


def _agentic_usage_from_traces(path: Path) -> dict[str, dict[str, int]]:
    """Aggregate per-claim agentic usage from a JSONL trace file.

    Sums final_output's tokens plus one-per-turn tool_call tokens (the
    per-turn usage is duplicated to every tool_call in that turn; taking
    one per `turn_idx` avoids double-counting).
    """
    per_claim: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
        }
    )
    if not path.exists():
        return {}

    # Track which turn_idx we've already counted per claim.
    seen_turns: dict[str, set[int]] = defaultdict(set)
    with path.open() as f:
        for line in f:
            e = json.loads(line)
            if e.get("engine") != "agentic":
                continue
            cid = e.get("claim_id")
            if not cid:
                continue
            kind = e.get("kind")
            if kind == "final_output":
                pass  # always count
            elif kind == "tool_call":
                t = e.get("turn_idx")
                if t in seen_turns[cid]:
                    continue
                seen_turns[cid].add(t)
            else:
                continue
            u = per_claim[cid]
            u["input_tokens"] += e.get("input_tokens") or 0
            u["output_tokens"] += e.get("output_tokens") or 0
            u["cache_read_tokens"] += e.get("cache_read_tokens") or 0
            u["cache_creation_tokens"] += e.get("cache_creation_tokens") or 0
    return per_claim


def _usage_from_result(rec: dict) -> dict[str, int] | None:
    """Return rec['usage'] only if it carries non-zero tokens."""
    u = rec.get("usage")
    if not u:
        return None
    if sum(u.get(k, 0) for k in ("input_tokens", "output_tokens")) == 0:
        return None
    return u


def _cost_dollars(u: dict[str, int], model: str) -> float:
    in_per_m, out_per_m, cr_per_m, _ = PRICING[model]
    return (
        u["input_tokens"] * in_per_m
        + u["output_tokens"] * out_per_m
        + u["cache_read_tokens"] * cr_per_m
    ) / 1e6


def main() -> None:
    rows = []  # (model, engine, claim_id, tokens..., cost, source)
    for model, rfname in RESULT_FILES.items():
        rpath = RESULTS_DIR / rfname
        results: dict = {}
        if rpath.exists():
            results = json.loads(rpath.read_text())["results"]
        else:
            print(f"(missing {rpath})")
        agentic_from_traces = _agentic_usage_from_traces(
            RESULTS_DIR / TRACE_FILES[model]
        )

        # Pass 1: rows from results.json (any engine).
        result_keys_seen: set[tuple[str, str]] = set()
        for rec in results.values():
            engine = rec.get("engine")
            claim_id = rec.get("claim_id")
            if rec.get("engine_error"):
                continue
            if not engine or not claim_id:
                continue
            result_keys_seen.add((engine, claim_id))
            usage = _usage_from_result(rec)
            source = "results.usage"
            if usage is None and engine == "agentic":
                usage = agentic_from_traces.get(claim_id)
                source = "traces.jsonl"
            if usage is None:
                continue
            rows.append(
                (
                    model,
                    engine,
                    claim_id,
                    usage["input_tokens"],
                    usage["output_tokens"],
                    usage["cache_read_tokens"],
                    usage["cache_creation_tokens"],
                    _cost_dollars(usage, model),
                    source,
                )
            )

        # Pass 2: agentic cells that exist only in traces.jsonl (the
        # `-k baseline` rerun overwrites results.json with baseline-only
        # cells, so agentic usage has to be recovered from the JSONL).
        for claim_id, usage in agentic_from_traces.items():
            if ("agentic", claim_id) in result_keys_seen:
                continue
            if not usage["input_tokens"] and not usage["output_tokens"]:
                continue
            rows.append(
                (
                    model,
                    "agentic",
                    claim_id,
                    usage["input_tokens"],
                    usage["output_tokens"],
                    usage["cache_read_tokens"],
                    usage["cache_creation_tokens"],
                    _cost_dollars(usage, model),
                    "traces.jsonl",
                )
            )

    # Per (model, engine) rollup. Values are heterogenous (lists of numbers
    # plus a set of source labels), so the dict is annotated `Any` rather
    # than a tighter shape that would force unions everywhere.
    from typing import Any

    rollup: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"in": [], "out": [], "cr": [], "cw": [], "cost": [], "src": set()}
    )
    for model, engine, _cid, i, o, cr, cw, cost, src in rows:
        r = rollup[(model, engine)]
        r["in"].append(i)
        r["out"].append(o)
        r["cr"].append(cr)
        r["cw"].append(cw)
        r["cost"].append(cost)
        r["src"].add(src)

    print("Chamber cost summary  (pricing notes at end)")
    print()
    hdr = (
        f"{'model':<18} {'engine':<10} {'cells':>5} {'mean_in':>10} "
        f"{'mean_out':>10} {'mean_$/cell':>13} {'total_$':>10} {'source':<14}"
    )
    print(hdr)
    print("-" * len(hdr))
    for (model, engine), r in sorted(rollup.items()):
        n = len(r["in"])
        mean_in = statistics.mean(r["in"]) if r["in"] else 0
        mean_out = statistics.mean(r["out"]) if r["out"] else 0
        mean_cost = statistics.mean(r["cost"]) if r["cost"] else 0
        total = sum(r["cost"])
        src = "/".join(sorted(r["src"])) or "-"
        print(
            f"{model:<18} {engine:<10} {n:>5} {mean_in:>10.0f} "
            f"{mean_out:>10.0f} ${mean_cost:>11.4f}  ${total:>8.3f}  {src}"
        )
    print()
    print("Pricing (per 1M tokens; in / out / cache-read):")
    for model, (i, o, cr, src) in PRICING.items():
        print(f"  {model:<18} ${i:.2f} / ${o:.2f} / ${cr:.3f}    -- {src}")


if __name__ == "__main__":
    main()
