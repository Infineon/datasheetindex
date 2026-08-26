"""Render canonical paper numbers for the chamber EMNLP paper.

The cross-model agentic results (Table 1, `tab:results`) come from
`variance_chamber.json` and are reported as mean +/- std over the
repeated runs (revision item 3). The cost ratios and the
reproducibility verdicts come from `baseline_chamber.json`: the variance
run is agentic-only, and reproducibility is agent-independent so it
carries no run-to-run variance by construction.

Every experimental-result number in the paper should come from this
script rather than be transcribed by hand. Benchmark context:
docs/datasheetindex_chamber_benchmark.md. Run:

    uv run python scripts/render_paper_tables.py
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from chamberbench.claimsio import archive_dir

BASELINE = archive_dir() / "baseline_chamber.json"
# Env override so the renderer can be pointed at a snapshot while a live
# variance run is rewriting the canonical artifact.
VARIANCE = Path(
    os.environ.get(
        "VARIANCE_JSON",
        str(archive_dir() / "variance_chamber.json"),
    )
)

# (input $/M, output $/M, cache_read $/M). Matches scripts/chamber_cost_summary.py;
# cache-creation tokens are not separately priced there, so they are omitted here
# too for consistency with the project's established cost methodology.
PRICING = {
    "claudesonnet4.6": (3.00, 15.00, 0.30),
    "gpt-5.1": (1.25, 10.00, 0.125),
    "qwen3.6-27b": (0.20, 0.60, 0.20),
}

MODELS = ["claudesonnet4.6", "gpt-5.1", "qwen3.6-27b"]
DISPLAY = {
    "claudesonnet4.6": "Claude Sonnet 4.6",
    "gpt-5.1": "GPT-5.1",
    "qwen3.6-27b": "Qwen3.6-27B",
}


def cost_dollars(usage: dict, model: str) -> float:
    """Per-cell dollar cost from a usage dict, using public list pricing."""
    in_m, out_m, cr_m = PRICING[model]
    return (
        usage.get("input_tokens", 0) * in_m
        + usage.get("output_tokens", 0) * out_m
        + usage.get("cache_read_tokens", 0) * cr_m
    ) / 1e6


def _fmt_pm(
    mean: float | None, std: float | None, *, prec: int = 2, latex: bool = False
) -> str:
    """Format `mean +/- std`; the mean alone when std is None; n/a when mean is."""
    if mean is None:
        return "n/a"
    pm = r" $\pm$ " if latex else " +/- "
    m = f"{mean:.{prec}f}"
    if std is None:
        return m
    return f"{m}{pm}{std:.{prec}f}"


# ---------------------------------------------------------------------------
# Table 1 -- cross-model agentic results, from the variance aggregate
# ---------------------------------------------------------------------------


def render_table1(variance: dict) -> None:
    agg = variance["aggregate"]
    print()
    print("=== Cross-model agentic results (variance: mean +/- std over repeats) ===")
    print(f"variance timestamp: {variance.get('timestamp')}")
    print(f"n_repeats: {variance.get('n_repeats')}")
    for model in MODELS:
        a = agg.get(model)
        if not a:
            continue
        fid, conf, lat = a["fidelity"], a["confidence"], a["latency_s"]
        ee, stab = a["engine_errors"], a["claim_stability"]
        print()
        print(DISPLAY[model])
        print(
            f"  fidelity pass:   per_run={fid['per_run']}  -> {_fmt_pm(fid['mean'], fid['std'], prec=1)} / 25"
        )
        print(f"  engine errors:   per_run={ee['per_run']}  total={ee['total']}")
        print(
            f"  mean confidence: {_fmt_pm(conf['mean'], conf['std'])}  "
            f"per_run={[round(x, 3) if x is not None else None for x in conf['per_run']]}"
        )
        print(
            f"  mean latency:    {_fmt_pm(lat['mean'], lat['std'], prec=1)} s  "
            f"per_run={[round(x, 1) if x is not None else None for x in lat['per_run']]}"
        )
        n_claims = stab["stable"] + stab["flipped"]
        print(
            f"  claim stability: {stab['stable']}/{n_claims} identical fidelity verdict across all repeats"
        )
        for fc in stab["flipped_claims"]:
            print(f"    flipped: {fc['id']}  {fc['pattern']}")


def render_latex_table1(variance: dict) -> None:
    agg = variance["aggregate"]
    print()
    print("=== LaTeX: Table 1 (tab:results) body rows ===")
    for model in MODELS:
        a = agg.get(model)
        if not a:
            continue
        fid, conf, lat = a["fidelity"], a["confidence"], a["latency_s"]
        ee = a["engine_errors"]
        fid_s = _fmt_pm(fid["mean"], fid["std"], prec=1, latex=True)
        conf_s = _fmt_pm(conf["mean"], conf["std"], prec=2, latex=True)
        lat_s = _fmt_pm(lat["mean"], lat["std"], prec=0, latex=True)
        ee_s = "/".join(str(x) for x in ee["per_run"])
        print(f"{DISPLAY[model]} & {fid_s} & {ee_s} & {conf_s} & {lat_s}\\,s \\\\")


# ---------------------------------------------------------------------------
# Cost ratios and reproducibility -- from baseline_chamber.json
# ---------------------------------------------------------------------------


def render_cost_ratios(baseline: dict) -> None:
    results = baseline["results"]
    claim_ids = baseline["claim_ids"]

    def eng_costs(engine: str, model: str) -> list[float]:
        out = []
        for cid in claim_ids:
            c = results.get(cid, {}).get(engine, {}).get(model)
            if c and c.get("status") == "ok" and not c.get("engine_error"):
                out.append(cost_dollars(c["usage"], model))
        return out

    son_ag = eng_costs("agentic", "claudesonnet4.6")
    son_bl = eng_costs("baseline", "claudesonnet4.6")
    qwen_ag = eng_costs("agentic", "qwen3.6-27b")
    son_ag_m = statistics.mean(son_ag)
    son_bl_m = statistics.mean(son_bl)
    qwen_m = statistics.mean(qwen_ag)
    print()
    print("=== Cost ratios (from baseline_chamber.json, single run) ===")
    print(f"Sonnet agentic   mean $/cell: ${son_ag_m:.4f}  (n={len(son_ag)})")
    print(f"Sonnet baseline  mean $/cell: ${son_bl_m:.4f}  (n={len(son_bl)})")
    print(f"Qwen agentic     mean $/cell: ${qwen_m:.4f}  (n={len(qwen_ag)})")
    print(f"  Sonnet baseline / Sonnet agentic = {son_bl_m / son_ag_m:.2f}x")
    print(f"  Sonnet agentic / Qwen agentic    = {son_ag_m / qwen_m:.2f}x")


def render_reproducibility(baseline: dict) -> None:
    results = baseline["results"]
    claim_ids = baseline["claim_ids"]
    print()
    print("=== Reproducibility verdicts (claim level, of 25) ===")
    verdicts: dict[str, str | None] = {}
    inconsistent: list[tuple[str, set]] = []
    for cid in claim_ids:
        seen: set[str] = set()
        for engine in ("agentic", "baseline"):
            for model in MODELS:
                c = results.get(cid, {}).get(engine, {}).get(model)
                if c and c.get("reproducibility"):
                    vd = c["reproducibility"].get("verdict")
                    if vd:
                        seen.add(vd)
        if len(seen) > 1:
            inconsistent.append((cid, seen))
        verdicts[cid] = next(iter(seen)) if len(seen) == 1 else None
    counts: dict[str, int] = {}
    for v in verdicts.values():
        key = v if v is not None else "(none)"
        counts[key] = counts.get(key, 0) + 1
    total = len(claim_ids)
    for v in sorted(counts):
        print(f"  {v:<14s} {counts[v]:2d} / {total}  ({100 * counts[v] / total:.1f}%)")
    if inconsistent:
        print("  WARNING: verdict varies across cells for:", inconsistent)
    else:
        print("  (verdict is consistent across every model/engine for each claim)")


def main() -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    variance = json.loads(VARIANCE.read_text(encoding="utf-8"))

    print("Canonical paper numbers")
    print(f"  Table 1:          {VARIANCE.name}")
    print(f"  cost / repro:     {BASELINE.name}")

    render_table1(variance)
    render_cost_ratios(baseline)
    render_reproducibility(baseline)
    render_latex_table1(variance)


if __name__ == "__main__":
    main()
