"""Baseline-vs-agentic comparison on the chamber benchmark corpus.

Backs the corpus half of the paper's Section 6.3 ("Baseline versus
agentic"): on datasheets that fit a context window the agentic engine
does not change the verdict, but it costs and runs measurably more.
This script derives those numbers from the canonical post-audit
baseline so every figure quoted in the paper is reproducible.

Reports:
  - verdict agreement between the two engines (fidelity, reproducibility)
    over the paired status-ok cells;
  - per-model mean cost and latency for each engine, and the
    agentic/baseline ratio.

Run:
    uv run python scripts/baseline_vs_agentic.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from chamberbench.claimsio import archive_dir, short_path

BASELINE = archive_dir() / "baseline_chamber.json"

MODELS = ("claudesonnet4.6", "gpt-5.1", "qwen3.6-27b")

# (input $/M, output $/M, cache_read $/M) -- matches scripts/render_paper_figures.py.
PRICING = {
    "claudesonnet4.6": (3.00, 15.00, 0.30),
    "gpt-5.1": (1.25, 10.00, 0.125),
    "qwen3.6-27b": (0.20, 0.60, 0.20),
}


def cost_dollars(usage: dict, model: str) -> float:
    """Per-cell dollar cost from a usage dict, using public list pricing."""
    in_m, out_m, cr_m = PRICING[model]
    return (
        usage.get("input_tokens", 0) * in_m
        + usage.get("output_tokens", 0) * out_m
        + usage.get("cache_read_tokens", 0) * cr_m
    ) / 1e6


def ok_cells(
    results: dict, claim_ids: list, engine: str, model: str
) -> list[dict[str, Any]]:
    """Status-ok cells for one engine and model, in claim order."""
    out: list[dict[str, Any]] = []
    for cid in claim_ids:
        c = results.get(cid, {}).get(engine, {}).get(model)
        if c and c.get("status") == "ok":
            out.append(c)
    return out


def main() -> int:
    data = json.loads(BASELINE.read_text(encoding="utf-8"))
    results, claim_ids = data["results"], data["claim_ids"]

    # --- verdict agreement over paired ok cells ---
    paired = fid_agree = repro_agree = 0
    for cid in claim_ids:
        for model in MODELS:
            b = results[cid].get("baseline", {}).get(model)
            a = results[cid].get("agentic", {}).get(model)
            if not (b and a and b.get("status") == "ok" and a.get("status") == "ok"):
                continue
            paired += 1
            if b["fidelity"]["overall_pass"] == a["fidelity"]["overall_pass"]:
                fid_agree += 1
            if (b.get("reproducibility") or {}).get("verdict") == (
                a.get("reproducibility") or {}
            ).get("verdict"):
                repro_agree += 1

    print("=" * 66)
    print("BASELINE vs. AGENTIC -- chamber benchmark corpus")
    print("=" * 66)
    print(f"  source: {short_path(BASELINE)}")
    print(f"  paired status-ok cells:      {paired}")
    print(f"  fidelity-verdict agreement:  {fid_agree}/{paired}")
    print(f"  reproducibility agreement:   {repro_agree}/{paired}")
    print()

    # --- per-model cost and latency, both engines ---
    print(f"  {'model':<18}{'engine':<10}{'mean cost':>12}{'mean lat':>12}")
    for model in MODELS:
        ratios: dict[str, dict[str, float]] = {}
        for engine in ("baseline", "agentic"):
            cells = ok_cells(results, claim_ids, engine, model)
            mean_cost = statistics.mean(cost_dollars(c["usage"], model) for c in cells)
            mean_lat = statistics.mean(c["latency_s"] for c in cells)
            ratios[engine] = {"cost": mean_cost, "lat": mean_lat}
            print(f"  {model:<18}{engine:<10}{mean_cost:>11.4f}${mean_lat:>10.1f}s")
        cost_x = ratios["agentic"]["cost"] / ratios["baseline"]["cost"]
        lat_x = ratios["agentic"]["lat"] / ratios["baseline"]["lat"]
        print(f"  {'':<18}{'ratio a/b':<10}{cost_x:>11.2f}x{lat_x:>10.2f}x")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
