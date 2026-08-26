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
from typing import Any

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
    # `ClaimSpec.confidence_min` is a non-Optional float defaulting to
    # `grading.DEFAULT_CONFIDENCE_FLOOR`, so it is always set. Passing it
    # explicitly (rather than relying on evaluate_case's own default) keeps this
    # correct if the field ever becomes optional.
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

    try:
        claims = {c.id: c for c in load_claims(args.claims)}
    except FileNotFoundError:
        print(
            f"ERROR: no claim file at {data_dir() / args.claims}.\n"
            "  CHAMBERBENCH_DATA_DIR must point at a directory containing a file\n"
            f"  named {args.claims!r}. The simplest way to make one is to copy the\n"
            "  shipped data/ directory and edit the fields you want to change --\n"
            "  keep every claim and every other field, or coverage drops silently.",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:  # noqa: BLE001 - yaml.YAMLError or pydantic ValidationError
        print(
            f"ERROR: {data_dir() / args.claims} could not be parsed as a claim set.\n"
            f"  {type(exc).__name__}: {str(exc)[:800]}\n"
            "  Each entry must be a complete ClaimSpec: quote numeric needles as\n"
            "  strings (value_contains: ['20', 'mV'], not [20, mV]), give\n"
            "  confidence_min a number rather than leaving it blank, and add no\n"
            "  extra keys. Copying data/claims.yaml and editing in place avoids\n"
            "  all three.",
            file=sys.stderr,
        )
        return 1

    doc = json.loads(baseline_path.read_text(encoding="utf-8"))

    print("Re-grading archived extractions")
    print(f"  claims:  {data_dir() / args.claims}")
    print(f"  archive: {baseline_path}")
    print()

    # Coverage, reported BEFORE any verdict. A claim set that omits ids simply
    # grades fewer cells, and without this the tool answers a smaller question
    # while looking like it answered the whole one -- an abstention, which the
    # annotator guide explicitly invites, would produce a CLEANER result than
    # our own self-check with nothing to say that cells had vanished.
    archive_ids = set(doc.get("results") or {})
    claim_ids = set(claims)
    missing_from_claims = sorted(archive_ids - claim_ids)
    missing_from_archive = sorted(claim_ids - archive_ids)

    print(f"  claims in set:        {len(claim_ids)}")
    print(f"  claim ids in archive: {len(archive_ids)}")
    if missing_from_claims:
        print(
            f"  NOT GRADED ({len(missing_from_claims)} archive ids absent from your claim set):"
        )
        for cid in missing_from_claims:
            print(f"    - {cid}")
    if missing_from_archive:
        print(
            f"  no archive cells ({len(missing_from_archive)} claims unused): "
            + ", ".join(missing_from_archive)
        )
    print()

    agree = disagree = ungradable = no_extraction = 0
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
                    # No stored extraction (e.g. the submit tool was never
                    # called). Correctly un-re-gradable, but it must be counted
                    # or the denominator cannot be checked against the
                    # archive's own cell count.
                    no_extraction += 1
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
    print(f"cells considered:      {total + ungradable + no_extraction}")
    print(f"  with an extraction:  {total + ungradable}")
    print(f"  without one:         {no_extraction}  (no verdict to re-check)")
    print(f"  failed to parse:     {ungradable}")
    print(f"re-graded:             {total}")
    print(f"  agree with published verdict:    {agree}")
    print(f"  disagree (verdict would change): {disagree}")

    if total == 0:
        print(
            "\nERROR: nothing was re-gradable; the archive or claim set is wrong.",
            file=sys.stderr,
        )
        return 1

    if missing_from_claims:
        print(
            f"\nPARTIAL: {len(missing_from_claims)} of {len(archive_ids)} archive claims "
            "were not covered by your claim set,\n         so the counts above describe a "
            "subset of the corpus."
        )
        return 2

    print(
        "\nNOTE: the variance repeats store verdicts without the extractions that\n"
        "      produced them, so they are not re-gradable here."
    )
    # A disagreement is a finding, not an error: it is exactly what an
    # alternative grading surface is meant to surface. Exit 0 either way, and
    # let the caller read the count.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
