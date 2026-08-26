"""Opportunistic scan: does any recorded chamber measurement already fail a
faithfully-stated claim, without perturbation? If yes -> a natural headline.
If no -> it substantiates the paper's claim that the corpus lacks naturally-
occurring reproducibility failures (the controlled perturbation supplies them).
Offline; no LLM."""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.error import URLError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from chamberbench.claimsio import load_claims as _load_claims
from chamberbench.reproducibility import run_protocol, verdict


def main() -> int:
    counts: dict[str, int] = {"pass": 0, "inconclusive": 0, "fail": 0}
    fails = []
    skipped: list[str] = []
    for claim in _load_claims():
        try:
            m = run_protocol(claim)
        except (OSError, URLError) as exc:
            # Chamber data could not be loaded -- almost always a missing or
            # unreachable download. This is NOT a result: the claims that need
            # chamber data are precisely the ones that could produce a
            # non-inconclusive verdict, so swallowing this and printing
            # "natural fails: 0" would report the paper's conclusion without
            # having tested it. Counted, and made fatal below.
            skipped.append(claim.id)
            print("  SKIP " + claim.id + ": " + type(exc).__name__ + ": " + str(exc))
            continue
        v = verdict(claim, m)
        counts[v.verdict] = counts.get(v.verdict, 0) + 1
        print("  " + claim.id.ljust(34) + " " + v.verdict)
        if v.verdict == "fail":
            fails.append((claim.id, v.rationale))
    print()

    if skipped:
        print(
            "INCOMPLETE: " + str(len(skipped)) + " claim(s) could not load chamber "
            "data, so the scan did not run."
        )
        print(
            "  The counts below are NOT a result. Every remaining claim "
            "short-circuits to"
        )
        print("  an unmatched-condition stub, which can only ever be 'inconclusive'.")
        print("  Skipped: " + ", ".join(skipped))
        print()
        print("verdict counts: " + str(counts) + "   (incomplete)")
        return 1

    print("verdict counts: " + str(counts))
    print("natural fails: " + str(len(fails)))
    for cid, why in fails:
        print("  FAIL " + cid + ": " + why)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
