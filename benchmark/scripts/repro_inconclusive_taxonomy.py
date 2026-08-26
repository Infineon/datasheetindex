"""Classify the chamber benchmark's inconclusive reproducibility verdicts.

The reproducibility axis routes most claims to ``inconclusive``. A
reviewer reading "88% inconclusive" may take it as an apparatus that
cannot decide. It is not: every inconclusive verdict maps to a
documented physical limitation. This script reads the canonical
post-audit baseline and bins each inconclusive claim into one of four
classes, so the 5 / 5 / 12 / 1 split quoted in the reproducibility-
decomposition section of the paper is reproducible from the same source
as every other number.

Classes:
  engagement_only      -- the chamber stages no experiment for the
                          quantity at all (the ACS70331 current-sensor
                          leg); the claim instruments the fidelity axis
                          and was never a reproducibility candidate.
  no_reference         -- the chamber can stage the physical condition
                          but lacks a traceable reference standard to
                          grade an absolute-accuracy claim against.
  unmatched_condition  -- the chamber cannot match a load-bearing
                          condition: a controlled stimulus it cannot
                          apply, or a device-internal quantity it
                          cannot observe.
  resolution_limited   -- the chamber stages the claim and matches its
                          conditions, but its measurement uncertainty
                          exceeds the spec tolerance: the apparatus
                          lacks the resolution to grade the claim.

Run:
    uv run python scripts/repro_inconclusive_taxonomy.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from chamberbench.claimsio import archive_dir

BASELINE = archive_dir() / "baseline_chamber.json"

# Sub-split of unmatched_condition: a stimulus the chamber cannot apply
# vs. a device-internal quantity it cannot observe. Keyed by substrings
# that appear in the unmatched load-bearing condition names.
STIMULUS_MARKERS = ("controlled", "ripple", "sweep")
REFERENCE_MARKERS = ("reference", "calibration")


def classify(rationale: str, unmatched: list[str]) -> tuple[str, str]:
    """Return ``(class, sub_class)`` for one inconclusive claim."""
    if "engagement-diagnostic" in rationale:
        return "engagement_only", "engagement_only"
    # Resolution gate: the chamber stages the claim and matches its
    # conditions, but its measurement sigma exceeds the spec tolerance --
    # the apparatus lacks the resolution to grade it either way. Keyed on
    # the verdict rationale emitted by reproducibility.verdict().
    if "cannot resolve this claim" in rationale:
        return "resolution_limited", "resolution_limited"
    names = " ".join(unmatched).lower()
    if any(mark in names for mark in REFERENCE_MARKERS):
        return "no_reference", "no_reference"
    if any(mark in names for mark in STIMULUS_MARKERS):
        return "unmatched_condition", "uncontrolled_stimulus"
    return "unmatched_condition", "unobservable_internal"


def main() -> int:
    data = json.loads(BASELINE.read_text(encoding="utf-8"))
    results, claim_ids = data["results"], data["claim_ids"]

    by_class: dict[str, list[str]] = defaultdict(list)
    by_sub: dict[str, list[str]] = defaultdict(list)
    n_pass = 0

    for cid in claim_ids:
        # Reproducibility is per-claim (model- and engine-independent);
        # read it off any available cell.
        cell: dict[str, Any] | None = None
        for engine in ("agentic", "baseline"):
            for model_cell in results[cid].get(engine, {}).values():
                cell = model_cell
                break
            if cell:
                break
        if not cell:
            continue
        rep = cell.get("reproducibility") or {}
        verdict = rep.get("verdict")
        if verdict == "pass":
            n_pass += 1
            continue
        if verdict != "inconclusive":
            continue
        rationale = rep.get("rationale") or ""
        unmatched = (cell.get("measurement") or {}).get("unmatched_conditions") or []
        cls, sub = classify(rationale, unmatched)
        by_class[cls].append(cid)
        by_sub[sub].append(cid)

    total_inconclusive = sum(len(v) for v in by_class.values())
    print("=" * 66)
    print("CHAMBER REPRODUCIBILITY: INCONCLUSIVE-VERDICT TAXONOMY")
    print("=" * 66)
    print("  source:", BASELINE.relative_to(PROJECT_ROOT))
    print("  definitive pass:", n_pass)
    print("  inconclusive:   ", total_inconclusive)
    print()
    for cls in (
        "engagement_only",
        "no_reference",
        "unmatched_condition",
        "resolution_limited",
    ):
        cids = sorted(by_class.get(cls, []))
        print(f"  {cls:<22} {len(cids):2d}")
        for cid in cids:
            print("      " + cid)
    print()
    print("  unmatched_condition sub-split:")
    for sub in ("uncontrolled_stimulus", "unobservable_internal"):
        print(f"    {sub:<24} {len(by_sub.get(sub, [])):2d}")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
