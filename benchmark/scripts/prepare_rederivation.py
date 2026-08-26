"""Prepare the blind re-derivation of the grading surface.

The paper's camera-ready deliverable, promised to R-T4Gz against RTR-5:
a second annotator, blind to the current values, independently re-derives
the two hand-authored fields that decide every fidelity number --
``value_contains`` (the required-substring needles) and ``confidence_min``
(the per-claim confidence floor) -- from the datasheets alone.

Why it is needed: both fields were fixed inside the window in which the
prompt builder was leaking answers and tolerances, and neither was
revisited when that leak was closed. Re-running the models cannot address
it, because the surface they are graded against is the thing in question.

What this script withholds, and why each one matters:

  value_contains   the answer to the exercise
  confidence_min   the answer to the exercise
  source_page      says which page to look at; the annotator must find it
  source_text      the datasheet sentence the needles were cut from
  claimed_min/max  the numeric value; needles follow almost mechanically

What it keeps is what a curator would have started from: the parameter
name, the prose description, the operating conditions, the expected unit,
and which datasheet to read.

Run:
    uv run python scripts/prepare_rederivation.py --out eval/chamber/rederivation.<name>.yaml
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from chamberbench.claimsio import data_dir

CLAIMS = data_dir() / "claims.yaml"
DATASHEETS = data_dir() / "datasheets"

# Fields that would hand the annotator the answer.
WITHHELD = (
    "value_contains",
    "confidence_min",
    "source_page",
    "source_text",
    "claimed_min",
    "claimed_max",
)

# claim-id prefix -> the datasheet to read.
COMPONENT_PDF = {
    "dps310": "barometer.pdf",
    "si115x": "fce10002ec99_light_sensor.pdf",
    "acs70331": "current_sensor.pdf",
}


# Sentences mentioning any of these are curator working notes, not claim text.
# They are dropped from the annotator's view and listed so the removal is
# auditable. The trigger for this: acs70331-saturation-low's description says
# "an earlier draft conflated typ with max from the adjacent column; the model
# under test extracted both columns correctly" -- which decides the derivation
# for the annotator and names the system under test.
#
# Deliberately NOT filtered: sentences about the chamber. The promise is that
# the annotator works from "the claim text, the component datasheet and the
# protocol description", so chamber-testability notes are theirs by right.
_CURATOR_NOTE = re.compile(
    r"curator|earlier draft|conflat|the model|model under test|agent\b|extract",
    re.IGNORECASE,
)


def _claim_text(description: str) -> tuple[str, list[str]]:
    """(what the annotator sees, what was withheld as curator commentary)."""
    sentences = [s for s in re.split(r"(?<=[.])\s+", description or "") if s.strip()]
    kept = [s for s in sentences if not _CURATOR_NOTE.search(s)]
    dropped = [s for s in sentences if _CURATOR_NOTE.search(s)]
    return " ".join(kept), dropped


def _component(claim_id: str) -> str:
    return claim_id.split("-", 1)[0]


def _conditions(claim: dict[str, Any]) -> str:
    """One-line rendering of the operating conditions, load-bearing marked."""
    parts = []
    for cond in claim.get("operating_conditions") or []:
        unit = cond.get("unit") or ""
        mark = "*" if cond.get("load_bearing") else ""
        parts.append(f"{cond.get('name')}={cond.get('value')}{unit}{mark}")
    return ", ".join(parts) if parts else "(none stated)"


def _leaks_a_number(claim: dict[str, Any]) -> list[str]:
    """Numbers in the kept prose that also appear in the withheld needles.

    A description like "accuracy is +/- 0.06 hPa" would hand over the answer
    even with value_contains withheld. Reported so the preparer can reword
    the description rather than discover the leak after the fact.
    """
    needles = [str(n) for n in (claim.get("value_contains") or [])]
    numeric = [n for n in needles if re.search(r"\d", n)]
    prose = " ".join(str(claim.get(k) or "") for k in ("description", "parameter"))
    # Digit-boundary match, not plain substring: the needle "1" occurs inside
    # "DPS310" in almost every description, and a check that fires on every
    # claim is a check nobody reads.
    return [n for n in numeric if re.search(rf"(?<!\d){re.escape(n)}(?!\d)", prose)]


def build(
    claims: list[dict[str, Any]],
    annotator: str,
    datasheet_prefix: str = "the datasheet corpus (see docs/reproducing.md)",
) -> tuple[str, list[str], list[str]]:
    lines: list[str] = []
    lines.append("# BLIND RE-DERIVATION OF THE GRADING SURFACE")
    lines.append("#")
    lines.append(
        "# INSTRUCTIONS: docs/chamber_rederivation_guide.md -- read that first."
    )
    lines.append("#")
    lines.append(
        "# For each claim below, read the named datasheet and decide, on your own:"
    )
    lines.append(
        "#   value_contains -- the substrings an extraction MUST contain to count"
    )
    lines.append(
        "#                     as correct for this claim (see the guide for how"
    )
    lines.append("#                     many and what kind).")
    lines.append("#   confidence_min -- the confidence floor below which you would not")
    lines.append(
        "#                     accept the answer even if the substrings matched."
    )
    lines.append("#")
    lines.append("# Do NOT open data/claims.yaml. It holds the values you are")
    lines.append("# re-deriving, and the whole exercise is whether you land on them")
    lines.append("# independently. Also avoid archive/*.json (model runs).")
    lines.append("#")
    lines.append(
        "# If you cannot find the parameter in the datasheet, leave value_contains"
    )
    lines.append(
        "# empty and say so in notes. An honest abstention is a result; a guess is"
    )
    lines.append("# not, and abstentions are reported separately.")
    lines.append("metadata:")
    lines.append(
        f"  annotator: '{annotator}'"
        if annotator
        else "  annotator: ''  # FILL IN: your name"
    )
    lines.append(f"  n_claims: {len(claims)}")
    lines.append("  withheld_fields:")
    for field in WITHHELD:
        lines.append(f"    - {field}")
    lines.append("claims:")

    warnings: list[str] = []
    notes: list[str] = []
    for claim in claims:
        cid = claim["id"]
        pdf = COMPONENT_PDF.get(_component(cid), "(unknown -- ask)")
        claim_text, dropped = _claim_text(str(claim.get("description") or ""))
        leaked = _leaks_a_number(claim)
        if leaked:
            warnings.append(f"{cid}: kept prose contains withheld needle(s) {leaked}")
        for sentence in dropped:
            notes.append(f"{cid}: dropped curator note -- {sentence[:88]}")
        lines.append("")
        lines.append(f"  # {'=' * 70}")
        lines.append(f"  # CLAIM: {cid}")
        lines.append(f"  # {'=' * 70}")
        lines.append(f"  # PARAMETER:  {claim.get('parameter')}")
        lines.append(f"  # DATASHEET:  {datasheet_prefix}/{pdf}")
        lines.append(f"  # UNIT:       {claim.get('expected_unit') or '(none)'}")
        # Labelled "AS RECORDED" because these come off the claim, and the claim
        # mixes the datasheet's own test conditions with chamber-side settings
        # under one key with nothing to tell them apart. Headed plainly
        # "CONDITIONS" it reads as "what the datasheet specifies", and the first
        # annotator correctly abstained on dps310-relative-accuracy when the
        # claim's VDD=3.0V did not match the datasheet's stated VDD=1.8V. Until
        # the two are separable in the schema, say where the line came from and
        # let the annotator trust the datasheet over it.
        lines.append(
            f"  # CONDITIONS AS RECORDED ON THE CLAIM: {_conditions(claim)}   (* = load-bearing)"
        )
        lines.append("  #   (these may include chamber-side settings; where they")
        lines.append("  #    disagree with the datasheet, the datasheet is right)")
        for chunk in _wrap(claim_text, 66):
            lines.append(f"  # {chunk}")
        lines.append(f"  - id: '{cid}'")
        # The example must not be any real claim's answer: the first draft used
        # ['0.06', 'hPa'], which is dps310-relative-accuracy's actual needle list.
        lines.append(
            "    value_contains: []      # FILL IN: list of substrings, e.g. ['12.3', 'kOhm']"
        )
        lines.append("    confidence_min:         # FILL IN: a number between 0 and 1")
        lines.append("    notes: ''")
    return "\n".join(lines) + "\n", warnings, notes


def _wrap(text: str, width: int) -> list[str]:
    out: list[str] = []
    line = ""
    for word in text.split():
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


GUIDE = PROJECT_ROOT / "docs" / "chamber_rederivation_guide.md"


def _needles() -> tuple[set[str], set[str]]:
    """(numeric needles, non-numeric needles) of three or more characters.

    Only the numeric ones are answers. Units and symbols are visible by design
    -- the annotator is given the expected unit and the parameter name, and the
    datasheet states the unit anyway -- which is exactly why
    score_rederivation.py reports numeric and non-numeric agreement separately
    and calls the numeric column the load-bearing one.
    """
    claims = yaml.safe_load(CLAIMS.read_text(encoding="utf-8"))["claims"]
    everything = {
        str(n)
        for c in claims
        for n in (c.get("value_contains") or [])
        if len(str(n)) >= 3
    }
    numeric = {n for n in everything if re.search(r"\d", n)}
    return numeric, everything - numeric


def _audit_bundle(bundle: Path) -> tuple[list[str], list[str]]:
    """(leaked answers, expected unit/symbol exposure) across the bundle."""
    numeric, other = _needles()
    leaks: list[str] = []
    expected: list[str] = []
    for path in sorted(bundle.rglob("*")):
        if not path.is_file() or path.suffix.lower() == ".pdf":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")

        def hits(pool: set[str], haystack: str = text) -> list[str]:
            return sorted(
                {
                    n
                    for n in pool
                    if re.search(rf"(?<![\d.]){re.escape(n)}(?![\d])", haystack)
                }
            )

        if found := hits(numeric):
            leaks.append(f"{path.relative_to(bundle)}: {found}")
        if found := hits(other):
            expected.append(f"{path.relative_to(bundle)}: {found}")
    return leaks, expected


def _make_bundle(bundle: Path, skeleton: str) -> int:
    """A self-contained folder for an outside annotator -- no repo clone.

    The repository is not safe to hand to an outsider for this exercise. Four
    tracked files carry the answers: claims.yaml itself, eval/chamber/README.md
    (6 needles), the pre-screen output (25), and classifier_gold.yaml, which
    prints "GOLD must-contain" above each of its cells and covers 20 of the 25
    claims -- the same exposure that disqualifies the annotator who did that
    labelling. Asking someone not to look at four files is a weaker control
    than not shipping them.
    """
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "datasheets").mkdir(exist_ok=True)
    (bundle / "README.md").write_text(
        GUIDE.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (bundle / "rederivation.yaml").write_text(skeleton, encoding="utf-8")
    for pdf in sorted(set(COMPONENT_PDF.values())):
        shutil.copy2(DATASHEETS / pdf, bundle / "datasheets" / pdf)
    leaks, expected = _audit_bundle(bundle)
    print(f"bundle: {bundle}")
    print(
        f"  README.md, rederivation.yaml, datasheets/ ({len(set(COMPONENT_PDF.values()))} PDFs)"
    )
    if leaks:
        print("  LEAK AUDIT FAILED -- a NUMERIC needle (an answer) is present:")
        for line in leaks:
            print(f"    {line}")
        return 1
    print("  leak audit: no numeric needle appears in any text file in the bundle")
    for line in expected:
        print(f"  visible by design (units/symbols, scored separately): {line}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", type=Path, required=True, help="Where to write the blind skeleton"
    )
    ap.add_argument("--annotator", default="", help="Pre-fill the annotator name")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an --out file that already carries answers",
    )
    ap.add_argument(
        "--bundle",
        type=Path,
        help="Also write a self-contained folder (guide + blind file + datasheets) to hand to an outsider",
    )
    args = ap.parse_args()
    out = (args.out if args.out.is_absolute() else Path.cwd() / args.out).resolve()

    if out.exists() and not args.force:
        existing = yaml.safe_load(out.read_text(encoding="utf-8")) or {}
        filled = [
            c
            for c in (existing.get("claims") or [])
            if c.get("value_contains") or c.get("confidence_min")
        ]
        if filled:
            who = (existing.get("metadata") or {}).get("annotator") or "unknown"
            print(
                f"REFUSING to overwrite {out}: {len(filled)} claims already answered by {who!r}."
            )
            print("Pass --out <new-path>, or --force to discard that work.")
            return 1

    claims = yaml.safe_load(CLAIMS.read_text(encoding="utf-8"))["claims"]
    text, warnings, notes = build(claims, args.annotator)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} -- {len(claims)} claims, {len(WITHHELD)} fields withheld")

    missing = sorted({_component(c["id"]) for c in claims} - set(COMPONENT_PDF))
    if missing:
        print(f"WARNING: no datasheet mapped for component(s): {missing}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    if not warnings:
        print("no withheld needle appears in the kept prose")
    print(f"dropped {len(notes)} curator-note sentence(s) from the annotator's view:")
    for note in notes:
        print(f"  - {note}")
    if args.bundle:
        print()
        bundle = (
            args.bundle if args.bundle.is_absolute() else Path.cwd() / args.bundle
        ).resolve()
        # Re-render with the path the annotator will actually have: the bundle
        # is flat, so a repo-relative DATASHEET line points at a directory that
        # is not there.
        flat, _, _ = build(claims, args.annotator, datasheet_prefix="datasheets")
        return _make_bundle(bundle, flat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
