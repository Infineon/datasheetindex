"""Re-score fidelity under exact-value matching instead of substring matching.

`value_contains` is checked with a case-insensitive substring test
(`chamberbench.grading.check_value_contains`). 20 of the 25 chamber claims carry at
least one needle of two characters or fewer, and 8 of the 30 *numeric* needles
are that short -- two different quantities, both true, and worth keeping apart.
So `['1', 'hPa']` passes on "+/-0.1 hPa", "11 hPa" or "1200 hPa", and
`['6', 'mA']` passes on "16 mA" or "0.6 mA". Every headline the rebuttal reports
-- the 25/25 fidelity, the closed-book memory-recovery rate, the
memorisation-prevalence gradient -- is measured through that matcher.

This script re-scores the same stored extractions with numeric needles compared
*numerically* against every number in the record, so a needle can neither sit
inside a longer literal (`1` in `1200`) nor be defeated by float formatting
(`1200` against the serialized value `1200.0`). Non-numeric needles (units,
symbols) keep the substring test, since "hPa" inside "hPa)" is a legitimate
match; `report_unit_containment` audits every pass that relies on it.

The point is not that the strict matcher is the right one -- both are defensible
-- but that a claim gated on a bare `1` is weak evidence either way, and the
difference between the two scores bounds how much of each headline rests on it.

Three limits this reports rather than hides: it reaches only the cells that store
an extracted record (repeat 1, not all 207 -- `report_coverage`); the closed-book
arm is scored against the submitted quotation alone because that is all its
records keep, so its 8 -> 7 is not commensurable with the baseline's 25/25; and a
permutation control (`report_permutation_control`) bounds whether the needles
discriminate at all.

Run:
    uv run python scripts/strict_fidelity_rescore.py
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from chamberbench.claimsio import archive_dir, data_dir

RESULTS_DIR = archive_dir()
CLAIMS_PATH = data_dir() / "claims.yaml"

# Needle grammar. Accepts a leading sign, thousands separators and an exponent, so
# that a needle in any of those forms is COMPARED numerically rather than silently
# falling back to the substring test. No claim currently uses those forms; this is
# hardening, because a needle that quietly takes the substring path is the exact
# failure this script exists to remove, and the fallback is invisible when it happens.
_NUMERIC = re.compile(r"^[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?$")

# Numeric comparison tolerance -- see `_strict_hit` for why it is not bit-exact.
_EPS = 1e-9

# Datasheets write negative numbers with an en dash, em dash or Unicode minus
# ("-40 C"), so a matcher keyed on ASCII hyphen reports a spurious miss. The
# grounding pass normalises the same way (`source_locator.py`).
_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"), "-")


def _norm(text: str) -> str:
    """NFKC-fold and unify dash variants so needles compare fairly."""
    return unicodedata.normalize("NFKC", text).translate(_DASHES)


def _norm_dashes(text: str) -> str:
    """Unify dash variants WITHOUT NFKC folding.

    Used for the numeric side. NFKC maps superscripts and vulgar fractions onto
    plain digits (`mm^2` -> `mm2`, `1/2` -> `1/2`), which would manufacture
    numbers that are not in the document.
    """
    return text.translate(_DASHES)


def _is_numeric_needle(needle: str) -> bool:
    return bool(_NUMERIC.match(needle.strip()))


# A numeric token in the haystack: optional sign, digits with optional thousands
# separators, optional fractional part, optional exponent.
#
# The boundaries are asymmetric on purpose, and the asymmetry is chosen from the
# data rather than by taste. The lookbehind excludes a digit or decimal point but
# NOT a letter, because 216 numbers in the corpus are glued to a unit or symbol
# ("1.8V", "1200h", "IDD8mA") and a letter-blocking lookbehind -- which is what
# this pattern had -- made every one of them invisible, producing spurious
# downgrades. The right guard likewise excludes only digits and decimal points.
#
# Consequence, accepted knowingly: a hyphen or colon between digits does not merge
# its operands, so "2026-06-05" yields 2026, 6 and 5 rather than nothing. That is
# the safer trade here because the corpus contains 22 hyphenated ranges that MUST
# decompose ("300-1200", "1.7-3.6", "40-85" are real endpoint pairs a claim needs
# to match) and zero dates or times. Re-verify both counts before changing this.
_NUM_TOKEN = re.compile(
    r"(?<![\d.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\d.])",
)


def _numeric_tokens(haystack: str) -> list[float]:
    """Every number in `haystack`, as floats.

    Comparing numbers instead of substrings is what makes the check *exact*
    rather than merely narrower. An earlier version of this function used a
    regex with a `(?![\\d.])` guard, and that was wrong in a way that mattered:
    `serialize_numerical` stringifies structured values as floats, so a claim
    whose needle is the integer `1200` was rejected against the extracted value
    `1200.0`. The needle could then only be satisfied by the model's own prose
    (`source_text`, `conditions`), which moved the grading surface off the
    structured fields and onto the one channel the closed-book arm proves is
    fabricated. Numeric comparison fixes that, and also makes `1,200` and
    `6.00` match while `11200` and `1200.5` still do not.
    """
    out: list[float] = []
    for match in _NUM_TOKEN.finditer(haystack):
        try:
            out.append(float(match.group(0).replace(",", "")))
        except ValueError:  # pragma: no cover -- the pattern guarantees a float
            continue
    return out


def _strict_hit(haystack: str, needle: str) -> bool:
    """True if `needle` is present in `haystack` as an exact value.

    Numeric needles are compared numerically against every number in the
    haystack, so `1` fails inside `1200`, `0.1` and `1.7` but succeeds against
    `1`, `1.0` and `1.00`. Non-numeric needles (units, symbols) keep the
    substring test, since `hPa` inside `hPa)` is a legitimate match -- see
    `report_unit_containment` for the audit of what that permits.

    NFKC folding is applied to the haystack for dash and width normalisation
    only; it is *not* applied to digits, because folding superscripts would let
    the `2` of `mm^2` satisfy a numeric needle.
    """
    haystack_norm = _norm_dashes(haystack)
    needle = _norm(needle).strip()
    if not _is_numeric_needle(needle):
        return needle.lower() in _norm(haystack).lower()
    want = float(needle.replace(",", ""))
    # Comparison is to within _EPS, not bit-exact: the values arrive as decimal
    # strings parsed to binary floats, so `0.06` from the claim and `0.06` from the
    # record need not be the same double. _EPS is far below the precision of any
    # datasheet value (the smallest magnitude in the claim set is 0.06) and far
    # above float error, so it cannot merge two distinct datasheet values. Stated
    # because "exact matching" would otherwise imply bit equality.
    return any(abs(tok - want) < _EPS for tok in _numeric_tokens(haystack_norm))


def _haystack(extracted: dict[str, Any]) -> str:
    """Everything the substring matcher would have searched, concatenated."""
    parts: list[str] = []
    for value in extracted.get("values") or []:
        for key in (
            "min_value",
            "max_value",
            "typical_value",
            "unit",
            "source_text",
            "conditions",
        ):
            v = value.get(key)
            if v is not None:
                parts.append(str(v))
    for key in ("text_value", "source_text", "original_terminology"):
        v = extracted.get(key)
        if v:
            parts.append(str(v))
    for v in extracted.get("list_value") or []:
        parts.append(str(v))
    return " ".join(parts)


def _score(extracted: dict[str, Any], needles: list[str]) -> tuple[bool, list[str]]:
    hay = _haystack(extracted)
    missed = [n for n in needles if not _strict_hit(hay, n)]
    return (not missed), missed


def _closed_book(claims: dict[str, Any]) -> None:
    print("=" * 78)
    print("CLOSED-BOOK PASSES UNDER STRICT MATCHING")
    print("=" * 78)
    for model in ("claudesonnet4.6", "gpt-5.1"):
        path = RESULTS_DIR / f"closed_book.{model}.json"
        if not path.exists():
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        passes = [c for c in doc["cells"] if c["fidelity_pass"]]
        survived: list[str] = []
        per_component: Counter[str] = Counter()
        print(f"\n{model}: {len(passes)} substring passes")
        for cell in passes:
            claim = claims[cell["claim_id"]]
            needles = list(claim.get("value_contains") or [])
            # The stored cell keeps only a truncated source_text, so the
            # strict check runs on that plus the needles' own context. A pass
            # here is therefore conservative: it cannot invent a match.
            hay = cell.get("submitted_source_text") or ""
            missed = [n for n in needles if not _strict_hit(hay, n)]
            ok = not missed
            if ok:
                survived.append(cell["claim_id"])
                per_component[cell["claim_id"].split("-")[0]] += 1
            print(
                f"    {cell['claim_id']:<34} needles={needles!s:<26} "
                f"{'STRICT PASS' if ok else 'weak: missing ' + ','.join(missed)}"
            )
        print(f"  -> {len(survived)}/{len(passes)} survive strict matching")
        n_cells = len(doc["cells"])
        print(
            f"     corrupt-success rate: {len(passes)}/{n_cells} = {100 * len(passes) / n_cells:.0f}%"
        )
        # P(right | answered) under both matchers. The response headlines the
        # substring figure, so the strict one has to be printed next to it or the
        # headline silently keeps the looser number.
        answered = sum(
            1
            for c in doc["cells"]
            if not str(c.get("fidelity_failure_reason") or "").startswith(
                "Found mismatch"
            )
        )
        if answered:
            print(
                f"     P(right | answered): substring {len(passes)}/{answered}"
                f" = {100 * len(passes) / answered:.0f}%, strict {len(survived)}/{answered}"
                f" = {100 * len(survived) / answered:.0f}%"
            )
        if per_component:
            print(f"     by component: {dict(per_component)}")
        # The memorisation-prevalence gradient, printed as ratios against each
        # component's own claim count -- the form the response states it in.
        totals = Counter(cid.split("-")[0] for cid in claims)
        gradient = ", ".join(
            f"{per_component.get(comp, 0)}/{totals[comp]}"
            for comp in ("dps310", "si115x", "acs70331")
        )
        print(f"     prevalence gradient (dps310, si115x, acs70331): {gradient}")


def _baseline_headline(claims: dict[str, Any]) -> None:
    """Re-score the 25/25 agentic headline on the run that stores extractions."""
    path = RESULTS_DIR / "baseline_chamber.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    print()
    print("=" * 78)
    print("AGENTIC FIDELITY HEADLINE UNDER STRICT MATCHING (post-audit run)")
    print("=" * 78)
    per_model: dict[str, dict[str, Any]] = {}
    for claim_id, by_engine in (doc.get("results") or {}).items():
        for model, cell in (by_engine.get("agentic") or {}).items():
            if not isinstance(cell, dict):
                continue
            fid = cell.get("fidelity") or {}
            if not fid.get("overall_pass"):
                continue
            extracted = ((cell.get("claim_result") or {}).get("extracted")) or {}
            needles = list((claims.get(claim_id) or {}).get("value_contains") or [])
            ok, missed = _score(extracted, needles)
            row = per_model.setdefault(model, {"passes": 0, "strict": 0, "weak": []})
            row["passes"] += 1
            row["strict"] += int(ok)
            if not ok:
                row["weak"].append(f"{claim_id} (missing {','.join(missed)})")
    for model in sorted(per_model):
        row = per_model[model]
        print(
            f"  {model:<20} substring {row['passes']}/25 -> strict {row['strict']}/25"
        )
        for w in row["weak"]:
            print(f"        downgraded: {w}")


def report_coverage() -> None:
    """State how much of the 207-cell clean population this re-score can reach.

    Only `baseline_chamber.json` stores the extracted record; `variance_chamber
    .json` cells carry verdicts, tool counts and usage but no extraction. So the
    re-score covers repeat 1 only, and the unreachable remainder includes Qwen
    repeats 2 and 3 -- the only cells with non-perfect fidelity, i.e. exactly
    where a matcher artifact would matter most. Saying "we re-scored every
    headline" without this is an overclaim.
    """
    variance = json.loads(
        (RESULTS_DIR / "variance_chamber.json").read_text(encoding="utf-8")
    )
    reachable = unreachable = 0
    detail: list[str] = []
    for model, repeats in (variance.get("runs") or {}).items():
        for idx, run in enumerate(repeats or [], 1):
            clean = sum(
                1
                for cell in (run.get("cells") or {}).values()
                if isinstance(cell, dict)
                and (cell.get("fidelity") or {}).get("overall_pass")
                and not (cell.get("fidelity") or {}).get("engine_error")
                and not cell.get("engine_error")
            )
            imported = str(run.get("source", "")).startswith("imported:")
            if imported:
                reachable += clean
            else:
                unreachable += clean
                detail.append(f"{model} repeat{idx} ({clean} clean)")
    print("=" * 78)
    print("RE-SCORE COVERAGE")
    print("=" * 78)
    print(
        f"clean cells with a stored extraction (re-scorable): {reachable}/{reachable + unreachable}"
    )
    print(f"clean cells without one (verdict-only, NOT re-scored): {unreachable}")
    print(f"  {', '.join(detail)}")
    print()


def report_unit_containment(claims: dict[str, Any]) -> None:
    """Audit what the surviving substring test on non-numeric needles permits.

    A unit needle can still match inside a longer unit token (`Pa` in `hPa`).
    Rather than concede that as an unbounded weakness, enumerate every pass that
    actually relies on it and say what the containing token was.
    """
    base = json.loads(
        (RESULTS_DIR / "baseline_chamber.json").read_text(encoding="utf-8")
    )
    rows: list[tuple[str, str, str, str]] = []
    for claim_id, by_engine in (base.get("results") or {}).items():
        for model, cell in (by_engine.get("agentic") or {}).items():
            if not isinstance(cell, dict) or not (cell.get("fidelity") or {}).get(
                "overall_pass"
            ):
                continue
            hay = _haystack(((cell.get("claim_result") or {}).get("extracted")) or {})
            for needle in (claims.get(claim_id) or {}).get("value_contains") or []:
                needle = str(needle)
                if _is_numeric_needle(needle) or needle.lower() not in hay.lower():
                    continue
                if re.search(rf"(?<![A-Za-z]){re.escape(needle)}(?![A-Za-z])", hay):
                    continue
                token = next(
                    (
                        t
                        for t in re.findall(r"[A-Za-z]+", hay)
                        if needle.lower() in t.lower() and t != needle
                    ),
                    "?",
                )
                rows.append((model, claim_id, needle, token))
    print("=" * 78)
    print("PASSES RELYING ON A UNIT NEEDLE MATCHING INSIDE A LONGER TOKEN")
    print("=" * 78)
    for model, claim_id, needle, token in rows:
        print(f"  {model:<18}{claim_id:<34}needle={needle!r:<8}inside {token!r}")
    print(
        f"  -> {len(rows)} such passes. Judge each: a compound unit is a correct match, a"
    )
    print("     different unit is not.")
    print()


def _structured_haystack(extracted: dict[str, Any]) -> str:
    """Only the structured numeric fields -- no model-authored free text."""
    parts: list[str] = []
    for value in extracted.get("values") or []:
        for key in ("min_value", "max_value", "typical_value", "unit"):
            v = value.get(key)
            if v is not None:
                parts.append(str(v))
    return " ".join(parts)


def report_prose_only_passes(claims: dict[str, Any]) -> None:
    """Passes whose numeric needle is satisfied ONLY in model-authored prose.

    This is the sharpest form of a reviewer's original concern about `value_contains`,
    and it is the number the response cites. A fidelity pass that no structured
    field supports rests entirely on `source_text` / `conditions` /
    `original_terminology` -- the channel the closed-book arm shows can be
    fabricated wholesale. Reported as a fraction of the re-scorable cells.
    """
    base = json.loads(
        (RESULTS_DIR / "baseline_chamber.json").read_text(encoding="utf-8")
    )
    prose_only: list[tuple[str, str, list[str]]] = []
    considered = 0
    for claim_id, by_engine in (base.get("results") or {}).items():
        needles = [
            str(n)
            for n in ((claims.get(claim_id) or {}).get("value_contains") or [])
            if _is_numeric_needle(str(n))
        ]
        for model, cell in (by_engine.get("agentic") or {}).items():
            if not isinstance(cell, dict) or not (cell.get("fidelity") or {}).get(
                "overall_pass"
            ):
                continue
            if not needles:
                continue
            considered += 1
            extracted = ((cell.get("claim_result") or {}).get("extracted")) or {}
            structured = _structured_haystack(extracted)
            missing = [n for n in needles if not _strict_hit(structured, n)]
            if missing and all(_strict_hit(_haystack(extracted), n) for n in needles):
                prose_only.append((model, claim_id, missing))
    print("=" * 78)
    print("PASSES RESTING ONLY ON MODEL-AUTHORED PROSE")
    print("=" * 78)
    print("A numeric needle satisfied nowhere in the structured value fields, only in")
    print("the model's own quotation -- the channel the closed-book arm shows is")
    print("fabricated on 8 of 8 answered cells.")
    print()
    for model, claim_id, missing in prose_only:
        print(
            f"  {model:<18}{claim_id:<34}needle(s) absent from structured fields: {missing}"
        )
    if considered:
        pct = 100 * len(prose_only) / considered
        print()
        print(
            f"  {len(prose_only)}/{considered} fidelity-passing cells with a numeric needle = {pct:.1f}%"
        )
    print()


def report_permutation_control(claims: dict[str, Any]) -> None:
    """Does the matcher discriminate, or would any extraction satisfy any claim?

    The re-score answers "did the loose matcher inflate the score". It does not
    answer "is the matcher measuring anything at all". Scoring every claim's
    needles against every *other* claim's stored extraction bounds that: a high
    mismatched pass rate would mean the 25/25 is an artifact of the needles
    being easy to satisfy by accident, whichever matcher is used.
    """
    base = json.loads(
        (RESULTS_DIR / "baseline_chamber.json").read_text(encoding="utf-8")
    )
    print("=" * 78)
    print("PERMUTATION CONTROL (mismatched claim x extraction)")
    print("=" * 78)
    for model in ("claudesonnet4.6", "gpt-5.1"):
        haystacks: dict[str, str] = {}
        for claim_id, by_engine in (base.get("results") or {}).items():
            cell = (by_engine.get("agentic") or {}).get(model)
            if isinstance(cell, dict) and (cell.get("fidelity") or {}).get(
                "overall_pass"
            ):
                haystacks[claim_id] = _haystack(
                    ((cell.get("claim_result") or {}).get("extracted")) or {}
                )
        if not haystacks:
            continue
        pairs = hits = 0
        per_claim: Counter[str] = Counter()
        for claim_id, needles in (
            (k, (claims.get(k) or {}).get("value_contains") or []) for k in haystacks
        ):
            for other_id, hay in haystacks.items():
                if other_id == claim_id:
                    continue
                pairs += 1
                if all(_strict_hit(hay, str(n)) for n in needles):
                    hits += 1
                    per_claim[claim_id] += 1
        print(
            f"  {model:<18}{hits}/{pairs} mismatched pairs pass = {100 * hits / pairs:.1f}%"
        )
        if per_claim:
            worst = ", ".join(f"{k} ({v})" for k, v in per_claim.most_common(4))
            print(f"      concentrated in: {worst}")
    print()
    print(
        "  Reading: a low rate means the needles discriminate and the 25/25 is not an"
    )
    print(
        "  accident of loose matching. The claims that DO leak are the weak-needle ones,"
    )
    print(
        "  so any figure resting mainly on those deserves the caveat the headline does not."
    )
    print()


def main() -> int:
    claims = {
        c["id"]: c
        for c in yaml.safe_load(CLAIMS_PATH.read_text(encoding="utf-8"))["claims"]
    }
    numeric_lengths = Counter(
        len(str(n))
        for c in claims.values()
        for n in (c.get("value_contains") or [])
        if _is_numeric_needle(str(n))
    )
    short_numeric = sum(v for k, v in numeric_lengths.items() if k <= 2)
    total_numeric = sum(numeric_lengths.values())
    # Both scopings, because they are different quantities and we quoted one as a
    # correction of the other. Per CLAIM: how many claims carry at least one short
    # needle of any kind. Per NUMERIC NEEDLE: how many numeric needles are short.
    claims_with_short = sum(
        1
        for c in claims.values()
        if any(len(str(n)) <= 2 for n in (c.get("value_contains") or []))
    )
    all_needles = [
        str(n) for c in claims.values() for n in (c.get("value_contains") or [])
    ]
    print(
        f"claims carrying >=1 needle of <= 2 characters: {claims_with_short}/{len(claims)}"
    )
    print(
        f"needles of <= 2 characters (all kinds): {sum(1 for n in all_needles if len(n) <= 2)}/{len(all_needles)}"
    )
    print(
        f"numeric needles: {total_numeric}, of which {short_numeric} are <= 2 characters"
    )
    print(f"numeric length distribution: {dict(sorted(numeric_lengths.items()))}")
    print()
    report_coverage()
    _closed_book(claims)
    _baseline_headline(claims)
    report_unit_containment(claims)
    report_prose_only_passes(claims)
    report_permutation_control(claims)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
