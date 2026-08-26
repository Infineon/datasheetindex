"""Re-grade the archived extractions under the CURRENT claim set.

This is the script that makes the grading surface auditable, and it exists
because the obvious alternative does not work: `render_paper_tables.py` prints
verdicts that were computed at run time and stored in the archive, so pointing
`CHAMBERBENCH_DATA_DIR` at a modified claim set changes nothing there. Only the
cells that kept their raw extraction can actually be re-graded, and this script
is the one that does it.

    uv run python scripts/regrade_archive.py
    CHAMBERBENCH_DATA_DIR=/path/to/your/claims uv run python scripts/regrade_archive.py

With no override it is a self-check: every re-grade should match the published
verdict, and any mismatch means the shipped claim set no longer reproduces the
shipped results. With an override it answers the question the blind
re-derivation was designed around -- would a different, defensible grading
surface have changed the published outcome?

Scope, stated plainly because it bounds the claim:

  * Only `baseline_chamber.json` retains `claim_result` (the raw extraction).
    The variance repeats store verdicts only, so repeats 2 and 3 -- which
    produce Table 1's spread and the Qwen instability result -- CANNOT be
    re-graded from this archive. Those verdicts must be taken on trust.
  * Grading uses `chamberbench.grading.evaluate_case`, the same function that
    produced the published verdicts, so the matcher is held fixed and only the
    claim set varies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from chamberbench.claimsio import archive_dir, data_dir, load_claims
from chamberbench.grading import evaluate_case
from chamberbench.models import ParameterResult


def _expectation(claim: Any) -> dict[str, Any]:
    """The `expected` dict `evaluate_case` grades against.

    Mirrors `score_rederivation.expect` exactly; the two must not drift, or a
    re-grade and a re-derivation score would disagree for reasons that have
    nothing to do with the claim set.
    """
    spec: dict[str, Any] = {
        "found": True,
        "value_contains": [str(x) for x in (claim.value_contains or [])],
    }
    if claim.confidence_min is not None:
        spec["confidence_min"] = float(claim.confidence_min)
    return spec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--claims",
        default="claims.yaml",
        help="Claim file inside the data dir (default: claims.yaml).",
    )
    ap.add_argument(
        "--verbose", action="store_true", help="List every cell, not just mismatches."
    )
    args = ap.parse_args()

    baseline_path = archive_dir() / "baseline_chamber.json"
    if not baseline_path.exists():
        print(f"ERROR: {baseline_path} not found.", file=sys.stderr)
        return 1

    claims = {c.id: c for c in load_claims(args.claims)}
    doc = json.loads(baseline_path.read_text(encoding="utf-8"))

    print("Re-grading archived extractions")
    print(f"  claims:  {data_dir() / args.claims}")
    print(f"  archive: {baseline_path}")
    print()

    agree = disagree = ungradable = 0
    flips: list[str] = []

    for cid, engines in (doc.get("results") or {}).items():
        claim = claims.get(cid)
        if claim is None:
            continue
        for engine, models in engines.items():
            if not isinstance(models, dict):
                continue
            for model, cell in models.items():
                if not isinstance(cell, dict):
                    continue
                raw = ((cell.get("claim_result") or {}).get("extracted")) or {}
                if not raw:
                    continue
                published = bool((cell.get("fidelity") or {}).get("overall_pass"))
                try:
                    parsed = ParameterResult.model_validate(raw)
                except Exception as exc:  # noqa: BLE001
                    ungradable += 1
                    print(
                        f"  UNGRADABLE {cid}/{engine}/{model}: {type(exc).__name__}: {exc}"
                    )
                    continue
                regraded = bool(
                    evaluate_case(parsed, _expectation(claim))["overall_pass"]
                )
                if regraded == published:
                    agree += 1
                    if args.verbose:
                        print(f"  ok   {cid}/{engine}/{model}: {published}")
                else:
                    disagree += 1
                    flips.append(
                        f"  FLIP {cid}/{engine}/{model}: published={published} -> regraded={regraded}"
                    )

    for line in flips:
        print(line)
    if flips:
        print()

    total = agree + disagree
    print(f"re-gradable cells: {total}  (ungradable: {ungradable})")
    print(f"  agree with published verdict:    {agree}")
    print(f"  disagree (verdict would change): {disagree}")

    if total == 0:
        print(
            "\nERROR: nothing was re-gradable; the archive or claim set is wrong.",
            file=sys.stderr,
        )
        return 1

    print(
        "\nNOTE: only baseline_chamber.json retains raw extractions. The variance\n"
        "      repeats store verdicts only and are NOT re-graded here."
    )
    # A disagreement is a finding, not an error: it is exactly what an
    # alternative grading surface is meant to surface. Exit 0 either way, and
    # let the caller read the count.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
