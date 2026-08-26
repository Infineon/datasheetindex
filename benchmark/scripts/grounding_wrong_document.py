"""Does a content-grounding check catch faithful transcription of the wrong document?

The wrong-content arm of the null-tool-injection experiment (``scripts/null_tool.py``)
serves every document tool from a decoy datasheet. Several runs then quote the decoy --
confidently, verbatim, with the decoy's own tables cross-checked against each
other. That is the failure class both of the benchmark's presence predicates are
structurally blind to: navigation happened, the cross-check predicate is
satisfied, every read succeeded, and the answer is wrong.

The deployed system carries a content predicate for exactly this shape
(``source_locator.py``, which re-finds each quoted ``source_text`` in the PDF via
``locate_text``). This script asks the only question that decides whether it
would help: re-finding the quote in *which* document?

  * against the datasheet the claim is about -- the check flags a miss
  * against the decoy actually served -- the check would confirm the quote

The gap between those two columns is the measurement. It also exposes the
paraphrase floor: a quotation the model summarised rather than copied fails to
locate in either document, so it is flagged for the wrong reason and tells us
nothing about the check's discriminating power.

Needs two things Tier 1 does not supply: the corpus PDFs on disk (not archived
output -- see docs/reproducing.md, "The corpus") and ``datasheetindex`` itself,
which is what provides ``locate_text`` and which arrives with the ``harness``
extra. This is the only script under ``scripts/`` that imports the library; the
rest re-derive published numbers from ``archive/`` alone. Both are missing on a
plain Tier-1 install, and this script degrades to a note rather than an error
for either, so the offline reproduction path never depends on a fetch or on the
second install tier.

Run:
    uv run python scripts/grounding_wrong_document.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from chamberbench.claimsio import archive_dir, corpus_dir, data_dir

RESULTS_DIR = archive_dir()
CLAIMS_PATH = data_dir() / "claims.yaml"

# The decoy the wrong-content arm served, and the local mirror of each
# component's real datasheet.
DECOY = corpus_dir() / "motor_driver.pdf"
INTENDED = {
    "dps310": corpus_dir() / "barometer.pdf",
    "si115x": corpus_dir() / "light_sensor.pdf",
    "acs70331": corpus_dir() / "current_sensor.pdf",
}

# locate_text matches an exact span; a long quotation is truncated so a trailing
# paraphrase does not defeat an otherwise verbatim head.
_QUERY_CHARS = 180


def _load_quotations() -> list[tuple[str, str, bool, str]]:
    """Every non-empty ``source_text`` the wrong-content arm submitted.

    The third element is whether the cell *answered* (committed to a value) rather
    than declining. It matters for the denominator: a production content check is
    only reached on an answer, so discrimination has to be reported over the
    answered subset as well as over every stored quotation.
    """
    out: list[tuple[str, str, bool, str]] = []
    for model in ("claudesonnet4.6", "gpt-5.1"):
        path = RESULTS_DIR / f"wrong_content.{model}.json"
        if not path.exists():
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        for cell in doc.get("cells") or []:
            quote = (cell.get("submitted_source_text") or "").strip()
            if quote:
                declined = str(cell.get("fidelity_failure_reason") or "").startswith(
                    "Found mismatch"
                )
                out.append((model, cell["claim_id"], not declined, quote))
    return out


def _locator_cache() -> dict[Path, Any]:
    return {}


def _load_locator() -> Any:
    """``datasheetindex.DatasheetTools``, or ``None`` when it is not installed.

    ``locate_text`` *is* the measurement here, so this is the one script under
    ``scripts/`` that needs the library -- everything else re-derives a
    published number from ``archive/`` alone. The library ships with the
    ``harness`` extra, not with Tier 1, so a Tier-1 reader who has fetched the
    corpus would otherwise meet a ``ModuleNotFoundError`` raised from inside
    the per-quotation loop, halfway through an already-printed table. Degrade
    up front instead, exactly as the missing-corpus path does.
    """
    try:
        from datasheetindex import DatasheetTools
    except ImportError:
        return None
    return DatasheetTools


def _hits(locator: Any, cache: dict[Path, Any], pdf: Path, query: str) -> int:
    """Number of locate_text matches, treating any failure as 'not found'."""
    if pdf not in cache:
        cache[pdf] = locator(str(pdf))
    try:
        return len(
            cache[pdf].locate_text(query[:_QUERY_CHARS], page=None, max_results=5)
        )
    except Exception:  # noqa: BLE001 -- a locator failure is a miss, not an error
        return 0


def main() -> int:
    missing = [p for p in (DECOY, *INTENDED.values()) if not p.exists()]
    if missing:
        print("=" * 78)
        print("GROUNDING A QUOTATION LIFTED FROM THE WRONG DOCUMENT")
        print("=" * 78)
        print(
            "corpus datasheet(s) not found -- this analysis re-runs locate_text against"
        )
        print(
            "the actual manufacturer PDFs, which are not redistributed with the archive."
        )
        print(f'Fetch them per docs/reproducing.md ("The corpus") into {corpus_dir()}')
        print("and re-run. Missing:")
        for path in missing:
            print(f"  {path.name}")
        return 0

    locator = _load_locator()
    if locator is None:
        print("=" * 78)
        print("GROUNDING A QUOTATION LIFTED FROM THE WRONG DOCUMENT")
        print("=" * 78)
        print(
            "datasheetindex not installed -- this analysis re-runs locate_text over the"
        )
        print(
            "corpus PDFs, so unlike the other analyses it needs the library itself. It"
        )
        print("ships with the harness extra; install that and re-run:")
        print("  uv pip install -e '.[harness]'")
        return 0

    claims = {
        c["id"]: c
        for c in yaml.safe_load(CLAIMS_PATH.read_text(encoding="utf-8"))["claims"]
    }
    quotations = _load_quotations()
    cache = _locator_cache()

    print("=" * 78)
    print("GROUNDING A QUOTATION LIFTED FROM THE WRONG DOCUMENT")
    print("=" * 78)
    print(f"decoy served: {DECOY.name}")
    print(f"quotations stored by the wrong-content arm: {len(quotations)}")
    print()
    print(f"{'model':<18}{'claim':<34}{'answered':>9}{'intended':>9}{'decoy':>7}")

    n_intended = n_decoy = 0
    a_total = a_intended = a_decoy = 0
    for model, claim_id, answered, quote in quotations:
        component = claim_id.split("-")[0]
        intended = INTENDED.get(component)
        if intended is None or claim_id not in claims:
            continue
        h_intended = _hits(locator, cache, intended, quote)
        h_decoy = _hits(locator, cache, DECOY, quote)
        n_intended += h_intended > 0
        n_decoy += h_decoy > 0
        if answered:
            a_total += 1
            a_intended += h_intended > 0
            a_decoy += h_decoy > 0
        print(f"{model:<18}{claim_id:<34}{answered!s:>9}{h_intended:>9}{h_decoy:>7}")

    total = len(quotations)
    print()
    print(f"located in the INTENDED datasheet: {n_intended}/{total}")
    print(f"located in the SERVED decoy      : {n_decoy}/{total}")
    print()
    print("Over the subset that actually ANSWERED -- the only cells a production")
    print("content check would ever have been reached on:")
    print(f"located in the INTENDED datasheet: {a_intended}/{a_total}")
    print(f"located in the SERVED decoy      : {a_decoy}/{a_total}")
    print()
    print("Reading: a content predicate keyed on the intended document flags every")
    print("quotation it cannot re-find, so it covers this class. But only the")
    print("decoy-locatable subset demonstrates *discrimination* -- the rest are")
    print("paraphrases that would fail to locate even when the answer is correct,")
    print("which is the same paraphrase floor that rules the check out as a gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
