"""Score a blind re-derivation of the grading surface against the live one.

Reports the three things the camera-ready appendix promises:

  1. agreement per claim and in aggregate,
  2. every disagreement, named, whatever it shows,
  3. whether each disagreement would change a published fidelity verdict.

(3) is mechanical, not a judgement call: the archived extractions are
re-scored under the annotator's needles and floor, and any cell whose
verdict moves is printed. Only the post-audit baseline run stores
extraction payloads, so that analysis covers those cells and not all 207
-- the same coverage limit the exact-value re-score appendix already
states.

Agreement is reported separately for numeric and non-numeric needles. The
unit is visible to the annotator (it is in the claim spec, and the
datasheet states it anyway), so unit agreement is not independent
evidence; the numeric needle is where fitting could have happened.

Run:
    uv run python scripts/score_rederivation.py --derivation data/rederivation.annotator2.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from chamberbench.claimsio import archive_dir, data_dir
from chamberbench.grading import DEFAULT_CONFIDENCE_FLOOR

CLAIMS = data_dir() / "claims.yaml"
BASELINE = archive_dir() / "baseline_chamber.json"
# Shared with `evaluate_case`; see grading.DEFAULT_CONFIDENCE_FLOOR.
DEFAULT_FLOOR = DEFAULT_CONFIDENCE_FLOOR


def _evaluator() -> Any:
    """The PRODUCTION scoring function, not a re-implementation.

    The published verdicts come from tests.eval.helpers.evaluate_case, which
    substring-matches against serialize_numerical() -- a haystack that
    deliberately excludes source_text. The first version of this analysis
    reused scripts/strict_fidelity_rescore._score instead, which compares
    numeric needles NUMERICALLY and searches a haystack that includes
    source_text. That varied the matcher and the needles at once, so every
    "flip" it produced was uninterpretable. Hold the matcher fixed; vary only
    the surface under test.
    """
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from chamberbench.grading import evaluate_case
    from chamberbench.models import ParameterResult

    return ParameterResult, evaluate_case


def _numeric(needle: str) -> bool:
    return bool(re.search(r"\d", str(needle)))


def _split(needles: list[str]) -> tuple[set[str], set[str]]:
    """(numeric needles, non-numeric needles), normalised for comparison."""
    items = [str(n).strip() for n in needles or []]
    return {n for n in items if _numeric(n)}, {
        n.casefold() for n in items if not _numeric(n)
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--derivation",
        type=Path,
        required=True,
        help="The filled blind re-derivation file",
    )
    args = ap.parse_args()
    path = (
        args.derivation
        if args.derivation.is_absolute()
        else Path.cwd() / args.derivation
    )

    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    theirs = {c["id"]: c for c in (doc.get("claims") or [])}
    ours = {
        c["id"]: c for c in yaml.safe_load(CLAIMS.read_text(encoding="utf-8"))["claims"]
    }
    annotator = (doc.get("metadata") or {}).get("annotator") or "unknown"

    answered, abstained = [], []
    for cid in ours:
        entry = theirs.get(cid) or {}
        if entry.get("value_contains"):
            answered.append(cid)
        else:
            abstained.append(cid)

    print("=" * 74)
    print("BLIND RE-DERIVATION OF THE GRADING SURFACE")
    print("=" * 74)
    print(f"  Annotator:  {annotator}")
    print(
        f"  Claims:     {len(ours)}   answered: {len(answered)}   abstained: {len(abstained)}"
    )
    if abstained:
        print(f"  Abstained:  {', '.join(sorted(abstained))}")
    if not answered:
        print("\nNothing to score yet.")
        return 0

    exact = num_match = unit_match = floor_match = 0
    disagreements: list[tuple[str, dict[str, Any]]] = []
    for cid in sorted(answered):
        our_num, our_unit = _split(ours[cid].get("value_contains") or [])
        their_num, their_unit = _split(theirs[cid].get("value_contains") or [])
        our_floor = ours[cid].get("confidence_min", DEFAULT_FLOOR)
        their_floor = theirs[cid].get("confidence_min")
        same_num, same_unit = our_num == their_num, our_unit == their_unit
        same_floor = (
            their_floor is not None
            and abs(float(their_floor) - float(our_floor)) < 1e-9
        )
        num_match += same_num
        unit_match += same_unit
        floor_match += same_floor
        if same_num and same_unit and same_floor:
            exact += 1
        else:
            disagreements.append(
                (
                    cid,
                    {
                        "ours": ours[cid].get("value_contains"),
                        "theirs": theirs[cid].get("value_contains"),
                        "our_floor": our_floor,
                        "their_floor": their_floor,
                        "notes": theirs[cid].get("notes") or "",
                    },
                )
            )

    n = len(answered)
    print()
    print(f"  Exact agreement (needles + floor):  {exact}/{n} = {exact / n:.1%}")
    print(
        f"  Numeric needles identical:          {num_match}/{n} = {num_match / n:.1%}   <- the load-bearing one"
    )
    print(
        f"  Non-numeric needles identical:      {unit_match}/{n} = {unit_match / n:.1%}"
        "   (unit was visible; not independent)"
    )
    print(
        f"  Confidence floor identical:         {floor_match}/{n} = {floor_match / n:.1%}"
    )

    if disagreements:
        print()
        print("-" * 74)
        print(f"DISAGREEMENTS ({len(disagreements)})")
        print("-" * 74)
        for cid, d in disagreements:
            print(f"  {cid}")
            print(f"      ours:   {d['ours']}   floor {d['our_floor']}")
            print(f"      theirs: {d['theirs']}   floor {d['their_floor']}")
            if d["notes"]:
                print(f"      note:   {d['notes']}")

    _verdict_flips(disagreements, ours, theirs)
    return 0


def _verdict_flips(
    disagreements: list[tuple[str, dict[str, Any]]],
    ours: dict[str, Any],
    theirs: dict[str, Any],
) -> None:
    """Re-score the archived extractions under the annotator's surface."""
    print()
    print("-" * 74)
    print("WOULD ANY DISAGREEMENT CHANGE A PUBLISHED VERDICT?")
    print("-" * 74)
    if not disagreements:
        print("  No disagreement to test.")
        return
    if not BASELINE.exists():
        print(f"  {BASELINE} is absent -- cannot test.")
        return

    result_cls, evaluate_case = _evaluator()
    doc = json.loads(BASELINE.read_text(encoding="utf-8"))
    flips, tested, unreproducible = [], 0, []
    for cid, _ in disagreements:
        for model, cell in (
            (doc.get("results") or {}).get(cid, {}).get("agentic") or {}
        ).items():
            if not isinstance(cell, dict):
                continue
            raw = ((cell.get("claim_result") or {}).get("extracted")) or {}
            if not raw:
                continue
            published = bool((cell.get("fidelity") or {}).get("overall_pass"))
            try:
                parsed = result_cls.model_validate(raw)
            except Exception:  # noqa: BLE001
                unreproducible.append(f"{cid}/{model}: extraction does not parse")
                continue

            def expect(claim: dict[str, Any]) -> dict[str, Any]:
                spec: dict[str, Any] = {
                    "found": True,
                    "value_contains": [
                        str(x) for x in claim.get("value_contains") or []
                    ],
                }
                floor = claim.get("confidence_min")
                if floor is not None:
                    spec["confidence_min"] = float(floor)
                return spec

            # Control: our own surface must reproduce the published verdict.
            # If it does not, this cell cannot support a claim about theirs.
            control = evaluate_case(parsed, expect(ours[cid]))
            if bool(control["overall_pass"]) != published:
                unreproducible.append(
                    f"{cid}/{model}: control {control['overall_pass']} != published {published}"
                )
                continue

            tested += 1
            theirs_eval = evaluate_case(parsed, expect(theirs[cid]))
            if bool(theirs_eval["overall_pass"]) != published:
                flips.append(
                    (cid, model, published, theirs_eval.get("failure_reason") or "")
                )

    print(
        f"  Cells with a stored extraction on a disagreeing claim, control reproduced: {tested}"
    )
    if unreproducible:
        print(
            f"  Excluded, control did not reproduce the published verdict: {len(unreproducible)}"
        )
        for line in unreproducible:
            print(f"    {line}")
    if not flips:
        print("  No published verdict changes under the re-derived surface.")
        return
    print(f"  {len(flips)} verdict(s) would change:")
    for cid, model, published, reason in flips:
        arrow = "PASS -> FAIL" if published else "FAIL -> PASS"
        print(f"    {cid} / {model}: {arrow}   {reason}")


if __name__ == "__main__":
    raise SystemExit(main())
