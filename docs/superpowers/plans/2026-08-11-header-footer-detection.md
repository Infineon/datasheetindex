# Running Header/Footer Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drop running page furniture (header, footer, revision line, page number) from the page-matched text file, so `search_text` stops returning walls of identical header hits.

**Architecture:** A new pure-function module `core/furniture.py` decides which normalized block keys are furniture, based on how many pages they recur on. `core/textfile.py` keeps all geometry: it gains `_ordered_blocks` (a refactor of existing reading-order logic) and `_extract_page_blocks`, which tags each block as inside or outside a 20% top/bottom band. `scan_pages` buffers blocks in its existing single traversal, asks `detect_furniture` for the key set, then joins the survivors.

**Tech Stack:** Python 3.13, PyMuPDF 1.28.2 (the only runtime dependency), pytest, ruff, ty, uv.

**Spec:** `docs/superpowers/specs/2026-08-11-header-footer-detection-design.md`. Read it before starting; it records why each constant has the value it does and which alternatives were measured and rejected.

## Global Constraints

- **PyMuPDF is the only runtime dependency.** Do not add one. Do not import `pymupdf4llm` or `pymupdf.layout` anywhere outside `core/engine.py` (see CLAUDE.md — the import installs a process-global hook).
- **The default lane must stay green.** A plain `uv sync` excludes the `[layout]` extra. Tests needing it use `pytest.mark.layout` and `pytest.importorskip("pymupdf.layout")`.
- **The preamble must not change.** `core/preamble.py:245` calls `_extract_page_text` directly and must keep receiving unstripped text. Do not move stripping into `_extract_page_text`.
- **No emoji or Unicode symbols** in scripts or test files (user's global instruction). Use `encoding="utf-8"` explicitly on file reads/writes.
- **No f-strings without a variable.**
- Ruff line-length 88. Pre-commit runs ruff check, ruff format, ty, and the unit tests. **Never use `--no-verify`.**
- Commit messages must not mention Claude or include `Co-Authored-By`.
- Work on branch `main` (content changes start on GitHub per CLAUDE.md).

### Constants (exact values, from the spec)

| Name | Value | Meaning |
|---|---|---|
| `MAX_FURNITURE_CHARS` | `200` | Longer raw block text is body prose, never furniture |
| `PAGE_FRACTION` | `0.5` | Share of pages a key must recur on |
| `MIN_PAGES` | `3` | Absolute floor, so 1- and 2-page documents never strip |
| `_FURNITURE_BAND_FRAC` | `0.20` | Top/bottom band, as a fraction of that page's height |
| caption prefixes | `figure`, `fig.`, `table`, `chart` | Case-insensitive; such blocks are never furniture |

**There is deliberately no line-count rule.** An earlier draft excluded blocks of 3+ lines; measured, that discards the PSoC footer (one block of 4 short lines) on 132 of 134 pages. Task 1 pins the opposite with a test.

## File Structure

| File | Responsibility |
|---|---|
| `src/datasheetindex/core/furniture.py` (create) | Pure decision logic: `normalize_key`, `is_candidate`, `furniture_threshold`, `detect_furniture`. No PyMuPDF, no env, no I/O. |
| `src/datasheetindex/core/textfile.py` (modify) | All geometry: `_ordered_blocks`, `_is_banded`, `_extract_page_blocks`, `_furniture_enabled_by_env`, and the two-pass `scan_pages`. |
| `tests/test_furniture.py` (create) | Unit tests for the pure module. |
| `tests/test_textfile.py` (modify) | Integration tests: guards, escape hatch, byte-identical no-furniture case. |
| `tests/test_layout_integration.py` (modify) | ONNX oracle precision check (`layout` marker). |
| `CHANGELOG.md`, `README.md`, `docs/datasheetindex_architecture.md`, `CLAUDE.md`, `pyproject.toml` (modify) | Docs and the 0.33.0 version bump. |

---

### Task 1: The pure decision module

**Files:**
- Create: `src/datasheetindex/core/furniture.py`
- Test: `tests/test_furniture.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `normalize_key(text: str) -> str`
  - `is_candidate(text: str) -> bool`
  - `furniture_threshold(total_pages: int) -> int`
  - `detect_furniture(page_keys: Sequence[Iterable[str]], total_pages: int) -> frozenset[str]`
  - Constants `MAX_FURNITURE_CHARS = 200`, `PAGE_FRACTION = 0.5`, `MIN_PAGES = 3`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_furniture.py`:

```python
"""Unit tests for running header/footer decision logic."""

from __future__ import annotations

from datasheetindex.core.furniture import (
    MAX_FURNITURE_CHARS,
    detect_furniture,
    furniture_threshold,
    is_candidate,
    normalize_key,
)


def test_normalize_key_collapses_whitespace_and_masks_digits():
    assert normalize_key("  Datasheet   46 \n 002-23185 Rev. *S  ") == (
        "Datasheet # #-# Rev. *S"
    )


def test_normalize_key_keeps_letters_distinct():
    """Masking digits must not make two different headers compare equal."""
    assert normalize_key("Chapter 3: Timers") != normalize_key("Chapter 4: Serial")


def test_normalize_key_of_blank_text_is_empty():
    assert normalize_key("   \n  ") == ""


def test_is_candidate_accepts_a_short_multi_line_block():
    """The PSoC footer is ONE block of four short lines.

    An earlier design excluded blocks of three or more lines, copying
    PageIndex. Measured, that discards this exact footer on 132 of 134
    pages -- the majority of the furniture the feature exists to remove.
    This test pins the removed rule so it cannot be reintroduced.
    """
    footer = "Datasheet\n46\n002-23185 Rev. *S\n2025-11-06"
    assert is_candidate(footer) is True


def test_is_candidate_rejects_long_blocks():
    assert is_candidate("x" * (MAX_FURNITURE_CHARS + 1)) is False
    assert is_candidate("x" * MAX_FURNITURE_CHARS) is True


def test_is_candidate_rejects_caption_prefixes():
    for caption in (
        "Table 43 (continued) USB specifications",
        "table 8 alternate functions",
        "Figure 12. Block diagram",
        "Fig. 3 timing",
        "Chart 2",
    ):
        assert is_candidate(caption) is False, caption


def test_is_candidate_does_not_reject_words_merely_starting_with_a_prefix():
    """'Tables' is not the caption keyword 'Table'; the boundary matters."""
    assert is_candidate("Tablet computer interface") is True


def test_is_candidate_rejects_blank_text():
    assert is_candidate("   \n ") is False


def test_furniture_threshold_uses_the_page_fraction():
    assert furniture_threshold(134) == 67
    assert furniture_threshold(42) == 21
    assert furniture_threshold(25) == 13  # ceil(12.5)


def test_furniture_threshold_floor_protects_short_documents():
    """A 1- or 2-page document can never reach the floor, so never strips."""
    assert furniture_threshold(1) == 3
    assert furniture_threshold(2) == 3
    assert furniture_threshold(0) == 3


def test_detect_furniture_counts_each_key_once_per_page():
    """A key repeated within one page counts once, not twice."""
    page_keys = [["hdr", "hdr"], ["hdr"], ["hdr"], ["other"]]
    assert detect_furniture(page_keys, total_pages=4) == frozenset({"hdr"})


def test_detect_furniture_requires_the_threshold():
    page_keys = [["a"], ["a"], ["b"], ["b"], ["b"], ["b"]]
    # threshold = max(3, ceil(0.5 * 6)) = 3; "a" has 2, "b" has 4.
    assert detect_furniture(page_keys, total_pages=6) == frozenset({"b"})


def test_detect_furniture_on_a_two_page_document_finds_nothing():
    page_keys = [["hdr"], ["hdr"]]
    assert detect_furniture(page_keys, total_pages=2) == frozenset()


def test_detect_furniture_on_no_pages_is_empty():
    assert detect_furniture([], total_pages=0) == frozenset()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_furniture.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'datasheetindex.core.furniture'`.

- [ ] **Step 3: Write the implementation**

Create `src/datasheetindex/core/furniture.py`:

```python
"""Running header/footer ("page furniture") detection.

Pure functions over strings and counts. This module never touches a
``pymupdf.Page``, reads no environment and does no I/O, so it is testable
without a PDF and cannot reach the process-global layout engine.

The method is a simplified Lin page-association (SPIE 2003): a block is
furniture when the same normalized text recurs, in a page-edge band, on a
large share of the document's pages. ``core/textfile.py`` owns the geometry
half of that decision; this module owns the text and counting half.

See ``docs/superpowers/specs/2026-08-11-header-footer-detection-design.md``
for the measurements behind every constant here.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence

#: Raw block text longer than this is body prose, never furniture. This is the
#: ONLY size guard. There is deliberately no line-count rule: PyMuPDF's
#: ``get_text("blocks")`` groups a whole footer into one block of several short
#: lines -- the PSoC 6 footer is a single 4-line, 41-character block -- so
#: excluding multi-line blocks discards real footers. Measured across seven
#: documents, a ">= 3 lines" rule missed genuine footers on five of them.
MAX_FURNITURE_CHARS = 200

#: Share of a document's pages a key must appear on to count as furniture.
#: Measured: real furniture recurs on 52-100% of pages; the two values below
#: 92% are both on one document. Lowering this to 0.33 starts deleting running
#: section headings ("6 Electrical specifications" on 47 of the PSoC's 134
#: pages), so 0.5 is the last value at which the survey corpus stays clean.
PAGE_FRACTION = 0.5

#: Absolute floor on the page count, so a 1- or 2-page document can never
#: produce furniture. With no recurrence evidence, keeping the text is the
#: honest answer.
MIN_PAGES = 3

_WHITESPACE_RE = re.compile(r"\s+")
_DIGIT_RUN_RE = re.compile(r"\d+")

#: A block opening with a caption keyword is content, even when it recurs.
#: Load-bearing rather than defensive: ``figures.caption_entries`` reads the
#: stripped text, so without this a caption near a page edge could vanish from
#: the figure index, and ``Table N (continued)`` captions are what
#: ``TocNode.continued_tables`` is built from.
_CAPTION_PREFIX_RE = re.compile(r"(?i)^(figure|fig\.|table|chart)\b")


def normalize_key(text: str) -> str:
    """Collapse whitespace and mask digit runs, giving a cross-page key.

    ``002-23185 Rev. *S | 2025-11-06`` becomes ``#-# Rev. *S | #-#-#``, so a
    revision line and a page number match across pages while the letters
    still have to agree. Deliberately no fuzzy matching: a similarity
    threshold can delete a genuine one-off line that resembles its
    neighbours, and missing furniture is the safer failure.
    """
    collapsed = _WHITESPACE_RE.sub(" ", text).strip()
    return _DIGIT_RUN_RE.sub("#", collapsed)


def is_candidate(text: str) -> bool:
    """Whether a block's text is eligible to be furniture at all."""
    stripped = text.strip()
    if not stripped:
        return False
    if len(stripped) > MAX_FURNITURE_CHARS:
        return False
    return _CAPTION_PREFIX_RE.match(stripped) is None


def furniture_threshold(total_pages: int) -> int:
    """Pages a key must appear on before it counts as furniture."""
    return max(MIN_PAGES, math.ceil(PAGE_FRACTION * total_pages))


def detect_furniture(
    page_keys: Sequence[Iterable[str]], total_pages: int
) -> frozenset[str]:
    """Return the keys that recur on enough pages to be furniture.

    ``page_keys`` is one iterable of normalized keys per page. Each key is
    counted once per page whatever the caller passes, so a header repeated
    twice on one page does not count double.
    """
    counts: dict[str, int] = {}
    for keys in page_keys:
        for key in set(keys):
            counts[key] = counts.get(key, 0) + 1
    threshold = furniture_threshold(total_pages)
    return frozenset(key for key, seen in counts.items() if seen >= threshold)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_furniture.py -q`
Expected: PASS (15 tests).

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check src/datasheetindex/core/furniture.py tests/test_furniture.py && uv run ruff format --check src/datasheetindex/core/furniture.py tests/test_furniture.py && uv run ty check`
Expected: all pass. If `ruff format --check` complains, run `uv run ruff format <paths>` and re-run the tests.

- [ ] **Step 6: Commit**

```bash
git add src/datasheetindex/core/furniture.py tests/test_furniture.py
git commit -m "feat: add pure running-furniture decision logic

Text and counting half of a simplified Lin page-association detector.
No line-count rule: PyMuPDF groups a whole footer into one block of short
lines, so excluding multi-line blocks discards real footers -- measured,
it misses the PSoC footer on 132 of 134 pages. A test pins that."
```

---

### Task 2: Extract `_ordered_blocks` without changing any output

**Files:**
- Modify: `src/datasheetindex/core/textfile.py:162-216` (`_extract_page_text`)
- Test: `tests/test_textfile.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `_ordered_blocks(page: pymupdf.Page) -> list[tuple[Any, ...]]`, returning PyMuPDF block tuples in final reading order. `_extract_page_text` keeps its exact signature `(page: pymupdf.Page) -> str` and its exact output.

This task is a pure refactor. Its whole deliverable is that nothing changes.

- [ ] **Step 1: Write the characterization test**

Append to `tests/test_textfile.py`:

```python
def test_ordered_blocks_joined_equals_extract_page_text(tmp_path):
    """_extract_page_text must stay a plain join over _ordered_blocks.

    Characterization test for the Task 2 refactor: it pins the existing
    output so the extraction of _ordered_blocks cannot alter reading order,
    column handling, or the empty-page case.
    """
    import pymupdf

    from datasheetindex.core.textfile import _extract_page_text, _ordered_blocks

    pdf = tmp_path / "two-column.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    # Two columns plus a full-width heading above them.
    page.insert_text((50, 60), "Wide heading across the page", fontsize=11)
    for row in range(6):
        page.insert_text((50, 120 + row * 14), f"left line {row}", fontsize=9)
        page.insert_text((320, 120 + row * 14), f"right line {row}", fontsize=9)
    doc.new_page()  # deliberately empty: the no-blocks path
    doc.save(str(pdf))
    doc.close()

    doc = pymupdf.open(str(pdf))
    try:
        for page in doc:
            joined = "\n".join(b[4] for b in _ordered_blocks(page))
            assert joined == _extract_page_text(page)
    finally:
        doc.close()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_textfile.py::test_ordered_blocks_joined_equals_extract_page_text -q`
Expected: FAIL with `ImportError: cannot import name '_ordered_blocks'`.

- [ ] **Step 3: Perform the refactor**

In `src/datasheetindex/core/textfile.py`, replace the whole body of `_extract_page_text` (currently lines 162-216) with these two functions. The partitioning and sorting code is moved verbatim; only the `return` statements change from joining to returning the list.

```python
def _ordered_blocks(page: pymupdf.Page) -> list[tuple[Any, ...]]:
    """Page text blocks in reading order.

    Uses ``page.get_text("blocks")`` to detect two-column layouts and
    reorder blocks so the left column is read before the right column.
    Falls back to standard top-to-bottom, left-to-right ordering when
    no column structure is detected.

    Split out of ``_extract_page_text`` so ``scan_pages`` can reach the
    block geometry -- specifically each block's vertical position -- which a
    joined string has already discarded. Reading order is decided here and
    nowhere else.
    """
    raw_blocks = page.get_text("blocks")
    text_blocks = [b for b in raw_blocks if b[_BLOCK_TYPE] == 0]

    if not text_blocks:
        return []

    page_width = page.rect.width
    result = _detect_columns(text_blocks, page_width)

    if result is None:
        # No columns detected -- standard reading order
        text_blocks.sort(key=lambda b: (b[_BLOCK_Y0], b[_BLOCK_X0]))
        return text_blocks

    gutter_x, col_top, col_bottom = result
    wide_threshold = page_width * _WIDE_BLOCK_FRAC

    above: list[tuple[Any, ...]] = []
    left_col: list[tuple[Any, ...]] = []
    right_col: list[tuple[Any, ...]] = []
    below: list[tuple[Any, ...]] = []

    for b in text_blocks:
        block_width = b[_BLOCK_X1] - b[_BLOCK_X0]
        mid_x = (b[_BLOCK_X0] + b[_BLOCK_X1]) / 2

        if block_width > wide_threshold:
            if b[_BLOCK_Y1] <= col_top:
                above.append(b)
            else:
                below.append(b)
        elif b[_BLOCK_Y1] <= col_top:
            above.append(b)
        elif b[_BLOCK_Y0] >= col_bottom:
            below.append(b)
        elif mid_x < gutter_x:
            left_col.append(b)
        else:
            right_col.append(b)

    return (
        sorted(above, key=lambda b: (b[_BLOCK_Y0], b[_BLOCK_X0]))
        + sorted(left_col, key=lambda b: b[_BLOCK_Y0])
        + sorted(right_col, key=lambda b: b[_BLOCK_Y0])
        + sorted(below, key=lambda b: (b[_BLOCK_Y0], b[_BLOCK_X0]))
    )


def _extract_page_text(page: pymupdf.Page) -> str:
    """Extract text from a page with column-aware reading order.

    Unstripped: this is what ``core/preamble.py`` reads for the page-marked
    front matter, whose documented contract is raw text with zero heuristics.
    Running-furniture stripping lives in ``scan_pages``, not here.
    """
    return "\n".join(b[_BLOCK_TEXT] for b in _ordered_blocks(page))
```

- [ ] **Step 4: Run the whole suite to verify nothing changed**

Run: `uv run pytest -q 2>&1 | tail -5`
Expected: PASS, 747 passed / 9 skipped (746 before, plus the new characterization test). If any existing test fails, the refactor altered behaviour — fix the refactor, do not adjust the test.

- [ ] **Step 5: Commit**

```bash
git add src/datasheetindex/core/textfile.py tests/test_textfile.py
git commit -m "refactor: split _ordered_blocks out of _extract_page_text

Pure refactor, no output change: scan_pages needs each block's vertical
position, which a joined string has already discarded. A characterization
test pins the join equivalence."
```

---

### Task 3: Band tagging and the escape hatch

**Files:**
- Modify: `src/datasheetindex/core/textfile.py` (add imports and three helpers)
- Test: `tests/test_textfile.py`

**Interfaces:**
- Consumes: `_ordered_blocks` (Task 2).
- Produces:
  - `_is_banded(block: tuple[Any, ...], page_height: float) -> bool`
  - `_extract_page_blocks(page: pymupdf.Page) -> list[tuple[str, bool]]` — `(text, banded)` pairs in reading order
  - `_furniture_enabled_by_env() -> bool`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_textfile.py`:

```python
def test_extract_page_blocks_tags_the_top_and_bottom_bands(tmp_path):
    """A block counts as banded only if it lies wholly inside an edge band."""
    import pymupdf

    from datasheetindex.core.textfile import _extract_page_blocks

    pdf = tmp_path / "bands.pdf"
    doc = pymupdf.open()
    page = doc.new_page()  # default letter page: 792pt tall
    height = page.rect.height
    page.insert_text((50, height * 0.05), "running header", fontsize=9)
    page.insert_text((50, height * 0.50), "body text in the middle", fontsize=9)
    page.insert_text((50, height * 0.96), "running footer", fontsize=9)
    doc.save(str(pdf))
    doc.close()

    doc = pymupdf.open(str(pdf))
    try:
        blocks = _extract_page_blocks(doc[0])
    finally:
        doc.close()

    banded = {text.strip(): flag for text, flag in blocks}
    assert banded["running header"] is True
    assert banded["running footer"] is True
    assert banded["body text in the middle"] is False


def test_extract_page_blocks_preserves_reading_order(tmp_path):
    """The pairs must arrive in the same order _extract_page_text joins them."""
    import pymupdf

    from datasheetindex.core.textfile import _extract_page_blocks, _extract_page_text

    pdf = tmp_path / "order.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    for row in range(5):
        page.insert_text((50, 100 + row * 20), f"line {row}", fontsize=9)
    doc.save(str(pdf))
    doc.close()

    doc = pymupdf.open(str(pdf))
    try:
        page = doc[0]
        joined = "\n".join(text for text, _ in _extract_page_blocks(page))
        assert joined == _extract_page_text(page)
    finally:
        doc.close()


def test_furniture_enabled_by_env_accepts_the_spellings_people_reach_for(
    monkeypatch,
):
    """Mirrors _parallel_enabled_by_env: matching only "0" would silently
    ignore =false and leave the escape hatch looking broken."""
    from datasheetindex.core.textfile import _furniture_enabled_by_env

    monkeypatch.delenv("DATASHEETINDEX_FURNITURE", raising=False)
    assert _furniture_enabled_by_env() is True

    for off in ("0", "false", "FALSE", "no", "off", "  Off  "):
        monkeypatch.setenv("DATASHEETINDEX_FURNITURE", off)
        assert _furniture_enabled_by_env() is False, off

    for on in ("1", "true", "yes", "anything else"):
        monkeypatch.setenv("DATASHEETINDEX_FURNITURE", on)
        assert _furniture_enabled_by_env() is True, on
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_textfile.py -q -k "banded or reading_order or furniture_enabled"`
Expected: FAIL with `ImportError: cannot import name '_extract_page_blocks'`.

- [ ] **Step 3: Write the implementation**

In `src/datasheetindex/core/textfile.py`, add `import logging` and `import os` to the stdlib imports at the top (after `from __future__ import annotations`, alongside `import re`), and add the module logger after the imports:

```python
logger = logging.getLogger(__name__)
```

Add this constant next to the other block constants (near `_MIN_GUTTER_PTS`):

```python
# Fraction of a page's height, at each edge, within which a block may be
# running furniture. Applied per page, so landscape and mixed-size pages need
# no special case. This band is what separates a running header from a
# "Table N" caption: measured on the PSoC, "Table #" recurs 89 times but never
# inside the band.
_FURNITURE_BAND_FRAC = 0.20
```

Add these three functions immediately after `_extract_page_text`:

```python
def _is_banded(block: tuple[Any, ...], page_height: float) -> bool:
    """Whether a block lies wholly inside the top or bottom edge band."""
    if page_height <= 0:
        return False
    top_limit = page_height * _FURNITURE_BAND_FRAC
    bottom_limit = page_height * (1.0 - _FURNITURE_BAND_FRAC)
    return block[_BLOCK_Y1] <= top_limit or block[_BLOCK_Y0] >= bottom_limit


def _extract_page_blocks(page: pymupdf.Page) -> list[tuple[str, bool]]:
    """Reading-ordered ``(text, banded)`` pairs for one page.

    Joining the texts reproduces ``_extract_page_text`` exactly; the flag is
    the geometry ``scan_pages`` needs and a joined string cannot carry.
    """
    page_height = page.rect.height
    return [
        (b[_BLOCK_TEXT], _is_banded(b, page_height)) for b in _ordered_blocks(page)
    ]


def _furniture_enabled_by_env() -> bool:
    """Whether DATASHEETINDEX_FURNITURE permits header/footer stripping.

    Accepts the spellings a user actually reaches for, for the reason
    ``structure._parallel_enabled_by_env`` records: matching only the literal
    "0" would silently ignore ``DATASHEETINDEX_FURNITURE=false``, leaving the
    escape hatch looking broken to the person who most needs it.
    """
    value = os.environ.get("DATASHEETINDEX_FURNITURE", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_textfile.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/datasheetindex/core/textfile.py tests/test_textfile.py
git commit -m "feat: tag blocks inside the page-edge band, add the escape hatch

The 20% band is what separates a running header from a 'Table N' caption:
measured on the PSoC, 'Table #' recurs 89 times and never inside it.
DATASHEETINDEX_FURNITURE accepts the same spellings as
DATASHEETINDEX_PARALLEL."
```

---

### Task 4: Two-pass `scan_pages`

**Files:**
- Modify: `src/datasheetindex/core/textfile.py:232-262` (`scan_pages`)
- Test: `tests/test_textfile.py`

**Interfaces:**
- Consumes: `normalize_key`, `is_candidate`, `detect_furniture` (Task 1); `_extract_page_blocks`, `_furniture_enabled_by_env` (Task 3).
- Produces: `scan_pages` keeps its signature `(doc, *, min_area_pct=DEFAULT_MIN_AREA_PCT) -> PageScan` and the `PageScan` shape is unchanged.

**Ordering hazard, read before writing code.** Today `scan_pages` appends figures per page as `raster_regions(page)` then `caption_entries(page_num, text)`, so the `figures` list is interleaved page by page. Splitting into two passes naively would emit every page's rasters first and every page's captions second, silently reordering the figure index. Buffer the per-page raster lists in pass 1 and re-interleave them in pass 2, exactly as written below.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_textfile.py`:

```python
def _furniture_pdf(path, pages, *, header=True, footer=True, caption=False):
    """A document with an optional running header, footer and top caption."""
    import pymupdf

    doc = pymupdf.open()
    for p in range(pages):
        page = doc.new_page()
        height = page.rect.height
        if header:
            page.insert_text((50, height * 0.05), "ACME AWC-3200 Controller",
                             fontsize=9)
        if caption:
            # A caption high on the page: banded, recurring, must survive.
            page.insert_text((50, height * 0.12), f"Table {p + 1} Pin assignments",
                             fontsize=9)
        page.insert_text((50, height * 0.45),
                         f"Body sentence unique to page {p + 1}.", fontsize=9)
        if footer:
            page.insert_text((50, height * 0.94), "Datasheet", fontsize=8)
            page.insert_text((50, height * 0.96), f"{p + 1}", fontsize=8)
            page.insert_text((50, height * 0.98), "AWC-3200 Rev. B", fontsize=8)
    doc.save(str(path))
    doc.close()


def test_scan_pages_drops_the_running_header_and_footer(tmp_path):
    import pymupdf

    from datasheetindex.core.textfile import scan_pages

    pdf = tmp_path / "furniture.pdf"
    _furniture_pdf(pdf, pages=8)
    doc = pymupdf.open(str(pdf))
    try:
        text = scan_pages(doc).text
    finally:
        doc.close()

    assert "ACME AWC-3200 Controller" not in text
    assert "AWC-3200 Rev. B" not in text
    # Body survives, and every page marker is still emitted.
    assert "Body sentence unique to page 4." in text
    for p in range(1, 9):
        assert f"--- PAGE {p} ---" in text


def test_scan_pages_keeps_a_recurring_table_caption(tmp_path):
    """The caption guard.

    A 'Table N' caption placed high on the page recurs on every page and
    normalizes to the same key as its neighbours. It must survive: these are
    the captions TocNode.continued_tables is built from, and the line-level
    approach this design replaced would have deleted them.
    """
    import pymupdf

    from datasheetindex.core.textfile import scan_pages

    pdf = tmp_path / "captions.pdf"
    _furniture_pdf(pdf, pages=8, caption=True)
    doc = pymupdf.open(str(pdf))
    try:
        text = scan_pages(doc).text
    finally:
        doc.close()

    assert "ACME AWC-3200 Controller" not in text  # furniture still goes
    for p in range(1, 9):
        assert f"Table {p} Pin assignments" in text  # captions all stay


def test_scan_pages_strips_nothing_from_a_two_page_document(tmp_path):
    """Below the MIN_PAGES floor there is no recurrence evidence."""
    import pymupdf

    from datasheetindex.core.textfile import scan_pages

    pdf = tmp_path / "short.pdf"
    _furniture_pdf(pdf, pages=2)
    doc = pymupdf.open(str(pdf))
    try:
        text = scan_pages(doc).text
    finally:
        doc.close()

    assert text.count("ACME AWC-3200 Controller") == 2


def test_scan_pages_leaves_a_furniture_free_document_unchanged(tmp_path):
    """No running furniture means byte-identical output."""
    import pymupdf

    from datasheetindex.core.textfile import _extract_page_text, scan_pages

    pdf = tmp_path / "plain.pdf"
    _furniture_pdf(pdf, pages=6, header=False, footer=False)
    doc = pymupdf.open(str(pdf))
    try:
        expected = "\n".join(
            part
            for i in range(len(doc))
            for part in (f"--- PAGE {i + 1} ---", _extract_page_text(doc[i]))
        )
        assert scan_pages(doc).text == expected
    finally:
        doc.close()


def test_scan_pages_escape_hatch_restores_the_unstripped_text(tmp_path,
                                                              monkeypatch):
    import pymupdf

    from datasheetindex.core.textfile import _extract_page_text, scan_pages

    pdf = tmp_path / "hatch.pdf"
    _furniture_pdf(pdf, pages=8)
    monkeypatch.setenv("DATASHEETINDEX_FURNITURE", "0")
    doc = pymupdf.open(str(pdf))
    try:
        expected = "\n".join(
            part
            for i in range(len(doc))
            for part in (f"--- PAGE {i + 1} ---", _extract_page_text(doc[i]))
        )
        assert scan_pages(doc).text == expected
    finally:
        doc.close()


def test_preamble_still_sees_the_running_furniture(tmp_path):
    """The preamble's "raw text, zero heuristics" contract.

    preamble.py reads _extract_page_text, not scan_pages, so stripping must
    not reach it. This is the test that fails if someone "simplifies" the
    design by moving the strip into _extract_page_text -- which would look
    like a tidy-up and would silently break a documented guarantee.
    """
    import pymupdf

    from datasheetindex.core.preamble import generate_preamble
    from datasheetindex.core.textfile import scan_pages

    pdf = tmp_path / "preamble.pdf"
    _furniture_pdf(pdf, pages=8)
    doc = pymupdf.open(str(pdf))
    try:
        preamble = generate_preamble(doc)
        stripped = scan_pages(doc).text
    finally:
        doc.close()

    assert "ACME AWC-3200 Controller" in preamble
    assert "ACME AWC-3200 Controller" not in stripped


def test_scan_pages_keeps_the_figure_index_interleaved_by_page(tmp_path):
    """Splitting into two passes must not reorder the figure index.

    Figures are appended per page as rasters-then-captions. A naive two-pass
    split emits all rasters before all captions, which silently reorders the
    index that build_datasheet publishes.
    """
    import pymupdf

    from datasheetindex.core.textfile import scan_pages

    pdf = tmp_path / "figs.pdf"
    doc = pymupdf.open()
    for p in range(4):
        page = doc.new_page()
        page.insert_text((50, 300), f"Figure {p + 1}. Diagram for page {p + 1}",
                         fontsize=9)
    doc.save(str(pdf))
    doc.close()

    doc = pymupdf.open(str(pdf))
    try:
        figures = scan_pages(doc).figures
    finally:
        doc.close()

    pages = [f["page"] for f in figures]
    assert pages == sorted(pages), f"figure index is not page-ordered: {pages}"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_textfile.py -q -k "scan_pages or preamble_still"`
Expected: `test_scan_pages_drops_the_running_header_and_footer`,
`test_scan_pages_keeps_a_recurring_table_caption` and
`test_preamble_still_sees_the_running_furniture` FAIL, all for the same reason
-- furniture is still present in `scan_pages` output. The byte-identical,
escape-hatch, short-document and figure-order tests PASS already; they are
guards against the change, not drivers of it, so passing now and still passing
later is exactly what they are for.

- [ ] **Step 3: Rewrite `scan_pages`**

Add to the imports at the top of `src/datasheetindex/core/textfile.py`:

```python
from datasheetindex.core.furniture import (
    detect_furniture,
    is_candidate,
    normalize_key,
)
```

Replace the body of `scan_pages` with:

```python
def scan_pages(
    doc: pymupdf.Document, *, min_area_pct: float = DEFAULT_MIN_AREA_PCT
) -> PageScan:
    """Extract page-matched text and the figure index in one traversal.

    Folded into the existing per-page pass rather than run as a second sweep:
    ``get_image_info()`` still costs what it costs, but reopening and
    re-loading every page object does not have to be paid twice.

    Two passes over a buffer, not two passes over the PDF. Furniture
    detection is a document-level decision -- a block is furniture because
    the same text recurs on other pages -- so pass 1 reads every page once
    and buffers its blocks, and pass 2 works from that buffer without
    touching the document again. A second PDF traversal was measured at +22%
    of this function's runtime, against ~200KB of buffer for a 134-page
    datasheet.
    """
    total_pages = len(doc)
    stripping = _furniture_enabled_by_env()

    page_blocks: list[list[tuple[str, bool]]] = []
    page_keys: list[set[str]] = []
    page_rasters: list[list[dict[str, object]]] = []
    excluded = 0

    # Pass 1: read each page once.
    for page_idx in range(total_pages):
        page = doc[page_idx]
        blocks = _extract_page_blocks(page)
        page_blocks.append(blocks)
        page_keys.append(
            {
                normalize_key(text)
                for text, banded in blocks
                if banded and is_candidate(text)
            }
        )
        rasters, page_excluded = raster_regions(page, min_area_pct=min_area_pct)
        page_rasters.append(rasters)
        excluded += page_excluded

    furniture = (
        detect_furniture(page_keys, total_pages) if stripping else frozenset()
    )

    # Pass 2: assemble from the buffer. Figures stay interleaved per page --
    # rasters then captions -- because that is the order the figure index has
    # always had, and build_datasheet publishes it.
    parts: list[str] = []
    figures: list[dict[str, object]] = []
    dropped = 0

    for page_idx, blocks in enumerate(page_blocks):
        page_num = page_idx + 1
        kept: list[str] = []
        for text, banded in blocks:
            if (
                furniture
                and banded
                and is_candidate(text)
                and normalize_key(text) in furniture
            ):
                dropped += 1
                continue
            kept.append(text)
        text = "\n".join(kept)

        parts.append(f"--- PAGE {page_num} ---")
        parts.append(text)

        figures.extend(page_rasters[page_idx])
        # Captions read the column-aware text, never page.get_text().
        figures.extend(caption_entries(page_num, text))

    if furniture:
        logger.info(
            "Dropped %d running header/footer blocks across %d pages: %s",
            dropped,
            total_pages,
            sorted(furniture),
        )

    return PageScan(
        text="\n".join(parts),
        figures=figures,
        excluded_below_min_area=excluded,
    )
```

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q 2>&1 | tail -5`
Expected: all pass. Existing tests in `tests/test_index.py`, `tests/test_figure_captions.py`, `tests/test_continued_tables.py` and `tests/test_reuse.py` also read this text — if one fails, read it before changing anything: it may be a genuine regression in figure ordering or page alignment.

- [ ] **Step 5: Commit**

```bash
git add src/datasheetindex/core/textfile.py tests/test_textfile.py
git commit -m "feat: drop running headers and footers from the text file

Two passes over a buffer, not over the PDF: furniture is a document-level
decision, and a second traversal measured +22% of scan_pages against ~200KB
of buffer. Figures stay interleaved per page so the published index keeps
its order. The preamble is untouched -- it reads _extract_page_text."
```

---

### Task 5: Real-document and oracle validation

**Files:**
- Modify: `tests/test_textfile.py` (real-PDF regression, `real_pdf` marker)
- Modify: `tests/test_layout_integration.py` (ONNX oracle precision, `layout` marker)

**Interfaces:**
- Consumes: everything from Tasks 1-4.
- Produces: no new production code.

**Re-measure before pinning.** The spec's figures came from a standalone prototype. Run the assertions once, and if a number disagrees, investigate the difference before editing the number — the prototype orders blocks by a plain `(y, x)` sort rather than the column-aware order, which cannot change *which* blocks are dropped.

- [ ] **Step 1: Write the real-document regression test**

Append to `tests/test_textfile.py`:

```python
@pytest.mark.real_pdf
def test_psoc_furniture_is_gone_and_search_is_cleaner():
    """The user-facing goal, on the bundled 134-page datasheet.

    The numbers are measured, not round targets. An earlier draft of the
    spec asserted search_text("PSOC") would fall "to under 20"; it falls to
    76, because PSOC legitimately appears throughout the body of a PSoC
    datasheet. If one of these disagrees, find out which number is wrong
    before weakening the assertion.
    """
    import pymupdf

    from datasheetindex.core.textfile import scan_pages, search_text

    pdf = Path(__file__).resolve().parent.parent / (
        "infineon-psoc-6-mcu-cy8c62x8-cy8c62xa-datasheet-datasheet-en.pdf"
    )
    if not pdf.exists():
        pytest.skip("bundled PSoC datasheet not present")

    doc = pymupdf.open(str(pdf))
    try:
        text = scan_pages(doc).text
    finally:
        doc.close()

    # The running header, and the 4-line footer block a line-count rule
    # would have kept. The header carries a trademark sign; it is spelled
    # with an escape so this file stays ASCII, per the repo's style rule.
    running_header = "PSOC" + "\u2122" + " 62 MCU"
    assert running_header not in text
    assert "002-23185 Rev. *S" not in text

    # Body content survives.
    assert "Electrical specifications" in text

    # Search precision: the measured improvement.
    assert len(search_text(text, "Datasheet", max_results=500)) <= 10
    assert len(search_text(text, "002-23185", max_results=500)) <= 3
    # No longer saturates the agent-visible default cap of 200.
    assert len(search_text(text, "PSOC", max_results=200)) < 200
```

Ensure `tests/test_textfile.py` has `import pytest` and `from pathlib import Path` at the top; add them if missing.

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_textfile.py -q -k psoc_furniture -m real_pdf`
Expected: PASS. If a bound is exceeded, print the actual counts and reconcile against the spec before editing.

- [ ] **Step 3: Write the ONNX oracle precision test**

Append to `tests/test_layout_integration.py`:

```python
def test_dropped_blocks_agree_with_the_layout_model():
    """Precision against an independent oracle.

    The ML layout model classifies page-header/page-footer directly. It is
    too slow and too optional to ship in the text path (~0.95s/page against
    an ~8s build, behind a 49MB extra), but it is an excellent cross-check:
    a block we delete should be one the model also calls furniture.

    Precision is asserted; recall is only reported. We knowingly detect less
    than the model does -- we skip fuzzy matching and it does not -- so a
    recall assertion would fail for a designed reason.
    """
    import pymupdf

    from datasheetindex.core.furniture import (
        detect_furniture,
        is_candidate,
        normalize_key,
    )
    from datasheetindex.core.textfile import (
        _extract_page_blocks,
        _is_banded,
        _ordered_blocks,
    )

    pdf = Path(__file__).resolve().parent.parent / (
        "infineon-psoc-6-mcu-cy8c62x8-cy8c62xa-datasheet-datasheet-en.pdf"
    )
    if not pdf.exists():
        pytest.skip("bundled PSoC datasheet not present")

    from pymupdf.layout.DocumentLayoutAnalyzer import get_model  # ty: ignore

    model = get_model()
    doc = pymupdf.open(str(pdf))
    try:
        # Sample every 6th page: the ONNX pass costs ~0.95s/page.
        sampled = list(range(0, len(doc), 6))
        page_keys = []
        for i in range(len(doc)):
            blocks = _extract_page_blocks(doc[i])
            page_keys.append(
                {
                    normalize_key(t)
                    for t, banded in blocks
                    if banded and is_candidate(t)
                }
            )
        furniture = detect_furniture(page_keys, len(doc))
        assert furniture, "no furniture detected on a document that has it"

        agreed = 0
        total = 0
        for pno in sampled:
            page = doc[pno]
            height = page.rect.height
            regions = model.predict(page)
            for b in _ordered_blocks(page):
                text = b[_BLOCK_TEXT_INDEX]
                if not (
                    _is_banded(b, height)
                    and is_candidate(text)
                    and normalize_key(text) in furniture
                ):
                    continue
                total += 1
                cx = (b[0] + b[2]) / 2
                cy = (b[1] + b[3]) / 2
                label = "none"
                for x0, y0, x1, y1, name in regions:
                    if x0 - 2 <= cx <= x1 + 2 and y0 - 2 <= cy <= y1 + 2:
                        label = name
                        break
                if label in ("page-header", "page-footer"):
                    agreed += 1
    finally:
        doc.close()

    assert total > 0, "sampled no dropped blocks; the sample is not exercising it"
    precision = agreed / total
    print(f"oracle precision: {agreed}/{total} = {precision:.3f}")
    assert precision >= 0.95, (
        f"only {agreed}/{total} dropped blocks are labelled page-header/"
        f"page-footer by the layout model. Below 0.95 this is a detector "
        f"defect -- fix the detector rather than lowering the bar."
    )
```

Add `_BLOCK_TEXT_INDEX = 4` as a module constant near the top of `tests/test_layout_integration.py`, and ensure the file imports `Path` (it already does) and `pytest` (it already does).

- [ ] **Step 4: Run the oracle test**

Run:
```bash
uv sync --extra layout
uv run pytest tests/test_layout_integration.py -q -s -k oracle
```
Expected: PASS, with the printed precision. **If precision is below 0.95, stop and treat it as a detector defect** — report the disagreeing blocks rather than lowering the threshold. If it passes comfortably above 0.95, leave the bar at 0.95; do not ratchet it to the observed value, because the ONNX model is a cross-check rather than ground truth and small sampling shifts should not turn it red.

- [ ] **Step 5: Restore the default lane and run everything**

Run:
```bash
uv sync
uv run pytest -q 2>&1 | tail -5
```
Expected: all pass, layout tests skipped.

- [ ] **Step 6: Commit**

```bash
git add tests/test_textfile.py tests/test_layout_integration.py
git commit -m "test: pin the real-document result and check precision vs the ONNX oracle

The layout model is too slow and too optional to ship in the text path but
makes a good independent cross-check. Precision is asserted at 0.95; recall
is reported only, since we knowingly detect less than it does."
```

---

### Task 6: Documentation and the version bump

**Files:**
- Modify: `CHANGELOG.md`, `README.md:232-233`, `docs/datasheetindex_architecture.md`, `CLAUDE.md`, `pyproject.toml:3`

**Interfaces:**
- Consumes: everything above. No code changes.

- [ ] **Step 1: Bump the version**

In `pyproject.toml` line 3, change `version = "0.32.0"` to `version = "0.33.0"`.

- [ ] **Step 2: Add the CHANGELOG entry**

Insert immediately after the `All notable changes...` line in `CHANGELOG.md`:

```markdown
## [0.33.0] - 2026-08-11

### Changed
- **The page-matched text file no longer carries running headers and footers.** Every page of a datasheet repeats a header naming the part and a footer with the document title, a revision string and a page number, and all of it reached `search_text`, `get_section_text` and the LLM ToC fallback. The cost was search precision rather than tokens: on the bundled PSoC 6, `search_text("PSOC")` returned 209 matches -- over the 200 cap an agent sees, so genuine hits were evicted -- of which 133 were the header, and `Datasheet` returned 138 of which 133 were furniture. Those now fall to 76 and 6. The text file shrinks 200,584 -> 193,020 characters (3.8%, ~1,891 tokens) across 265 dropped blocks.
- **Detection is native PyMuPDF, on the default lane, with no new dependency.** A block is dropped when it lies wholly inside the top or bottom 20% of its page, is at most 200 characters, does not open with a caption keyword, and its whitespace-collapsed digit-masked text recurs on at least `max(3, ceil(0.5 * pages))` pages. This is a simplified Lin page-association (SPIE 2003) -- the standard method, not a new one.
- **`Table N` captions are safe by construction.** An earlier line-level design would have deleted them: `Table #` recurs 89 times on the PSoC. Working at block granularity puts captions outside the band, and the caption-keyword rule is a second guard -- load-bearing, because `figures.caption_entries` reads the stripped text and `TocNode.continued_tables` is built from `Table N (continued)`.
- **The preamble is unchanged, byte for byte.** `preamble.py` reads `_extract_page_text`, which stays unstripped; the stripping lives in `scan_pages`. The documented "raw text, zero heuristics" contract still holds, with no flag and no second code path.

### Added
- **`DATASHEETINDEX_FURNITURE=0`** disables stripping entirely (also `false`, `no`, `off`), matching `DATASHEETINDEX_PARALLEL`'s spellings and its reasoning: an escape hatch that ignores `=false` looks broken to whoever most needs it.
- **`core/furniture.py`**, a pure module with no PyMuPDF, no environment and no I/O, so the decision logic is testable without a PDF.
- **An ONNX-oracle precision test.** `pymupdf.layout` classifies `page-header`/`page-footer` directly but costs ~0.95s/page against an ~8s build and sits behind the optional `[layout]` extra, so it cannot serve the text path -- but it makes an independent cross-check. Blocks we drop are asserted to be furniture by the model at >= 0.95 precision. Recall is reported only: we knowingly detect less than it does.

### Compatibility at a glance
- **The text file changes** for any document with running furniture; that is the point. Page markers, page ranges and section boundaries are untouched, so `get_section_text` and page alignment are unaffected, and the figure index keeps its per-page ordering. `artifact_cache` fingerprints on the version, so stale text files invalidate automatically.
- **Not detected, by design:** furniture whose *letters* vary per page, such as a per-chapter running title, and TI-style headers that alternate by odd/even page and so sit near a third of pages. Both fail safe by keeping text. Lowering the threshold to reach them was measured and is worse -- at 0.33 the PSoC starts losing `6 Electrical specifications`, a running section heading.
- **Considered and rejected:** an existing library (the only zero-dependency candidate, `refinedoc`, deletes `Table N (continued)` captions because it works on text lines with no coordinates) and LLM-driven detection (cheap, and higher recall, but on tcan1044a-q1 `qwen3.6-27b` flags 73 of 198 candidates including a table header row and several section headings -- and the text file must build without credentials). Both are recorded in the design spec.
```

- [ ] **Step 3: Update the README**

In `README.md`, the deliverables/tools section around line 232, add a clause to the text-file description. Find the bullet describing the page-matched text file and append:

```markdown
  (running headers and footers are omitted; set `DATASHEETINDEX_FURNITURE=0` to keep them)
```

- [ ] **Step 4: Update the architecture doc**

In `docs/datasheetindex_architecture.md`, in the `core/` module tree, add after the `textfile.py` entry:

```
│   ├── furniture.py       # Running header/footer decision logic
│   │                      #   normalized-key recurrence within a page-edge band
```

And in the "What we add" list, add:

```markdown
- **Running header/footer stripping** — a block inside the top/bottom 20% band whose whitespace-collapsed, digit-masked text recurs on at least half the pages is dropped from the page-matched text file. A simplified Lin page-association; block granularity and a caption-keyword guard are what keep `Table N (continued)` captions intact. The preamble keeps raw text.
```

- [ ] **Step 5: Update CLAUDE.md**

Add to the "Two deliverables" section, item 2, after the existing description of the text file:

```markdown
   Running headers/footers are stripped (`core/furniture.py`): banded + recurring
   + digit-masked key. `DATASHEETINDEX_FURNITURE=0` disables it. The **preamble
   is deliberately exempt** — it calls `_extract_page_text`, which is unstripped,
   so its "zero heuristics" contract survives. Do not move stripping into
   `_extract_page_text`.
```

- [ ] **Step 6: Verify and commit**

Run:
```bash
uv run pytest -q 2>&1 | tail -3
uv run ruff check src/ tests/ && uv run ty check
```
Expected: all pass.

```bash
git add -A
git commit -m "docs: changelog, README, architecture and CLAUDE.md for 0.33.0"
```

- [ ] **Step 7: Final verification on both lanes**

```bash
uv run pytest -q 2>&1 | tail -3          # default lane
uv sync --extra layout
uv run pytest -q 2>&1 | tail -3          # layout lane
uv sync                                   # restore the default lane
git status --short                        # must be clean
```

Both lanes must be green and the tree clean before this plan is considered done. Do not push or tag; releasing is a separate, explicitly-requested step (see CLAUDE.md — the tag is the only release trigger and a published version can never be re-uploaded).
