# locate_text Source Grounding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `locate_text` primitive that maps a query string on a PDF page to bounding boxes (percentages + PDF points), exposed on both tool-server surfaces.

**Architecture:** A stateless core function (`core/locate.py`) resolves coordinates on demand via PyMuPDF `page.search_for` (verbatim fast path) with a normalized word-level token fallback (`page.get_text("words")`). Shared normalization helpers move to `core/_textmatch.py` so the new code and the existing `search_text` ladder share one implementation. The function is wired into `DatasheetTools` and both server registries.

**Tech Stack:** Python 3.11+, PyMuPDF (`pymupdf`), pytest, uv, ruff, ty. Design spec: `docs/superpowers/specs/2026-06-15-locate-text-source-grounding-design.md`.

---

## File Structure

- **Create** `src/datasheetindex/core/_textmatch.py` — shared normalization + token-matching helpers, extracted from `textfile.py`.
- **Modify** `src/datasheetindex/core/textfile.py` — delete the moved helpers; import them from `_textmatch`.
- **Create** `src/datasheetindex/core/locate.py` — `locate_text`, `TextLocation`, `_Box`, geometry + matching helpers.
- **Modify** `src/datasheetindex/tools/registry.py` — `DatasheetTools.locate_text` + Agent SDK `@tool`.
- **Modify** `src/datasheetindex/mcp_server.py` — `locate_text_tool` + registration + `instructions` update + `inspect_page_tool` guard fix.
- **Create** `tests/test_locate.py` — core unit tests.
- **Modify** `tests/test_registry.py` — exact-tool-set assertion + `DatasheetTools.locate_text` test.
- **Modify** `tests/test_mcp_server.py` — exact-tool-set assertion + smoke + `inspect_page_tool` guard test.
- **Modify** `docs/datasheetindex_architecture.md` — document the new tool.
- **Modify** `CHANGELOG.md`, `pyproject.toml` — entry + version bump.

Parity note: like `search_text`/`TextSearchMatch`, `locate_text`/`TextLocation` are **not** added to `src/datasheetindex/__init__.py` `__all__`.

---

## Task 1: Extract shared text-matching helpers to `_textmatch.py`

Behavior-preserving refactor, guarded by the existing suite.

**Files:**
- Create: `src/datasheetindex/core/_textmatch.py`
- Modify: `src/datasheetindex/core/textfile.py`

- [ ] **Step 1: Establish a green baseline**

Run: `uv run pytest tests/test_textfile.py tests/test_registry.py -q`
Expected: PASS (this is the baseline the refactor must preserve).

- [ ] **Step 2: Create `_textmatch.py`**

```python
"""Shared text-normalization and token-matching helpers.

Extracted from ``textfile.py`` so both the page-text search ladder and the
``locate_text`` coordinate primitive share one normalization implementation
instead of reaching across modules into private names.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from functools import cache, lru_cache
from typing import NamedTuple

# Normalize Unicode hyphen/dash/minus code points to ASCII "-" so a hyphen
# query matches a datasheet that uses an en-dash, figure dash, or minus sign.
# Code points: U+2010..U+2015 (hyphen/dashes) and U+2212 (minus sign).
_DASH_TRANSLATION = str.maketrans(
    {chr(cp): "-" for cp in (0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2015, 0x2212)}
)

_TOKEN_EDGE_PUNCTUATION = ".,;:!?"

_TOKEN_RE = re.compile(r"\S+")


class _TokenSpan(NamedTuple):
    value: str
    start: int
    end: int


@lru_cache(maxsize=256)
def _translate_search_text(text: str) -> str:
    return text.translate(_DASH_TRANSLATION)


def _normalize_token(token: str, *, case_sensitive: bool) -> str:
    normalized = _translate_search_text(token).strip(_TOKEN_EDGE_PUNCTUATION)
    return normalized if case_sensitive else normalized.casefold()


def _match_query_tokens(
    page_tokens: Sequence[_TokenSpan],
    query_tokens: Sequence[str],
    start_index: int,
    *,
    max_gap_tokens: int,
) -> list[int] | None:
    if page_tokens[start_index].value != query_tokens[0]:
        return None

    @cache
    def _search(query_index: int, previous_token_index: int) -> tuple[int, ...] | None:
        if query_index >= len(query_tokens):
            return ()

        expected = query_tokens[query_index]
        search_start = previous_token_index + 1
        search_end = min(len(page_tokens), previous_token_index + max_gap_tokens + 2)
        for token_index in range(search_start, search_end):
            if page_tokens[token_index].value != expected:
                continue
            suffix = _search(query_index + 1, token_index)
            if suffix is not None:
                return (token_index, *suffix)
        return None

    suffix = _search(1, start_index)
    if suffix is None:
        return None
    return [start_index, *suffix]
```

- [ ] **Step 3: Delete the moved definitions from `textfile.py` and import them**

In `src/datasheetindex/core/textfile.py`:

1. Change the functools import (line 7) — `cache` is no longer used here:

```python
from functools import lru_cache
```

2. **Delete** these definitions (they now live in `_textmatch.py`): `_TOKEN_RE`, `_DASH_TRANSLATION`, `_TOKEN_EDGE_PUNCTUATION`, the `_TokenSpan` class, `_translate_search_text`, `_normalize_token`, and `_match_query_tokens`.

3. Add this import directly below the existing `if TYPE_CHECKING:` block (only the names `textfile.py` still references):

```python
from datasheetindex.core._textmatch import (
    _TOKEN_RE,
    _TokenSpan,
    _match_query_tokens,
    _normalize_token,
    _translate_search_text,
)
```

Note: `_DASH_TRANSLATION` and `_TOKEN_EDGE_PUNCTUATION` are intentionally **not** imported — after the move nothing in `textfile.py` references them directly (only the moved functions did).

- [ ] **Step 4: Verify no regression and clean imports**

Run: `uv run pytest tests/test_textfile.py tests/test_registry.py -q && uv run ruff check src/datasheetindex/core/textfile.py src/datasheetindex/core/_textmatch.py`
Expected: tests PASS; ruff reports no errors (no unused imports).

- [ ] **Step 5: Commit**

```bash
git add src/datasheetindex/core/_textmatch.py src/datasheetindex/core/textfile.py
git commit -m "refactor: extract shared text-matching helpers into _textmatch"
```

---

## Task 2: Core `locate_text` — types, geometry, fast path, list/dedup

The fallback is a stub here; Task 3 implements it.

**Files:**
- Create: `src/datasheetindex/core/locate.py`
- Test: `tests/test_locate.py`

- [ ] **Step 1: Write the failing fast-path tests**

Create `tests/test_locate.py`:

```python
"""Tests for locate_text coordinate grounding."""

from __future__ import annotations

import base64

import pymupdf
import pytest

from datasheetindex.core.locate import locate_text
from datasheetindex.tools.vision import inspect_page


def _doc_with(text_at: list[tuple[float, float, str]]) -> pymupdf.Document:
    """One-page PDF with each (x, y, text) drawn at that baseline point."""
    doc = pymupdf.open()
    page = doc.new_page()
    writer = pymupdf.TextWriter(page.rect)
    for x, y, text in text_at:
        writer.append((x, y), text)
    writer.write_text(page)
    return doc


def test_fast_path_exact_hit():
    doc = _doc_with([(72, 72, "Hello world")])
    results = locate_text(doc, "Hello", page=1)
    doc.close()

    assert len(results) == 1
    loc = results[0]
    assert loc["page"] == 1
    assert loc["match_method"] == "search_for"
    assert len(loc["boxes"]) == 1
    assert loc["region"] == loc["boxes"][0]
    assert "pattern" not in loc  # single-string query is untagged


def test_not_found_returns_empty():
    doc = _doc_with([(72, 72, "Hello world")])
    assert locate_text(doc, "absent", page=1) == []
    doc.close()


def test_page_out_of_range_raises():
    doc = _doc_with([(72, 72, "Hello")])
    with pytest.raises(ValueError, match="between 1 and"):
        locate_text(doc, "Hello", page=5)
    doc.close()


def test_empty_query_raises():
    doc = _doc_with([(72, 72, "Hello")])
    with pytest.raises(ValueError, match="must not be empty"):
        locate_text(doc, "   ", page=1)
    doc.close()


def test_pct_points_consistency():
    doc = _doc_with([(72, 72, "Hello world")])
    page_rect = doc[0].rect
    loc = locate_text(doc, "Hello", page=1)[0]
    doc.close()

    box = loc["boxes"][0]
    assert loc["page_width"] == pytest.approx(page_rect.width)
    assert loc["page_height"] == pytest.approx(page_rect.height)
    assert box["pct"]["left"] * loc["page_width"] == pytest.approx(
        box["points"]["x0"] - page_rect.x0
    )
    assert box["pct"]["bottom"] * loc["page_height"] == pytest.approx(
        box["points"]["y1"] - page_rect.y0
    )


def test_round_trip_into_inspect_page():
    doc = _doc_with([(72, 72, "Hello world")])
    loc = locate_text(doc, "Hello", page=1)[0]
    cropped = inspect_page(doc, page=1, region=loc["region"]["pct"])
    full = inspect_page(doc, page=1)
    doc.close()

    assert cropped[0]["type"] == "image"
    assert len(base64.b64decode(cropped[0]["data"])) < len(
        base64.b64decode(full[0]["data"])
    )


def test_list_query_tags_and_caps():
    doc = _doc_with([(72, 72, "Hello world")])
    tagged = locate_text(doc, ["Hello", "world"], page=1)
    capped = locate_text(doc, ["Hello", "world"], page=1, max_results=1)
    deduped = locate_text(doc, ["Hello", "Hello"], page=1)
    doc.close()

    assert {r["pattern"] for r in tagged} == {"Hello", "world"}
    assert len(capped) == 1
    assert len(deduped) == 1  # same box found twice collapses; first pattern wins
    assert deduped[0]["pattern"] == "Hello"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_locate.py -q`
Expected: FAIL with `ModuleNotFoundError: datasheetindex.core.locate`.

- [ ] **Step 3: Create `core/locate.py` (fast path + stub fallback)**

```python
"""locate_text -- map a query string to its bounding box(es) on a PDF page.

The missing edge between ``search_text`` (find text -> char offset) and
``inspect_page`` (render a region): given a string and a page, return where it
sits, as normalized percentages (for the ``inspect_page`` round-trip) and raw
PDF points (for PDF-native annotation).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, NotRequired, TypedDict

if TYPE_CHECKING:
    import pymupdf

_Rect = tuple[float, float, float, float]


class _Box(TypedDict):
    pct: dict[str, float]  # {"top","bottom","left","right"}, each 0.0-1.0
    points: dict[str, float]  # {"x0","y0","x1","y1"}, PDF points


class TextLocation(TypedDict):
    page: int  # 1-indexed
    match_method: str  # "search_for" | "tokens"
    page_width: float  # PDF points
    page_height: float  # PDF points
    region: _Box  # union of boxes; the inspect_page round-trip input
    boxes: list[_Box]  # >= 1; a multi-line match yields one box per line
    pattern: NotRequired[str]  # which query produced this hit (list queries only)


def _box_from_rect(rect: _Rect, page_rect: pymupdf.Rect) -> _Box:
    x0, y0, x1, y1 = rect
    width = page_rect.width
    height = page_rect.height
    return {
        "pct": {
            "left": (x0 - page_rect.x0) / width,
            "right": (x1 - page_rect.x0) / width,
            "top": (y0 - page_rect.y0) / height,
            "bottom": (y1 - page_rect.y0) / height,
        },
        "points": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
    }


def _union_region(boxes: list[_Box], page_rect: pymupdf.Rect) -> _Box:
    return _box_from_rect(
        (
            min(b["points"]["x0"] for b in boxes),
            min(b["points"]["y0"] for b in boxes),
            max(b["points"]["x1"] for b in boxes),
            max(b["points"]["y1"] for b in boxes),
        ),
        page_rect,
    )


def _search_for_occurrences(page: pymupdf.Page, query: str) -> list[list[_Rect]]:
    """Fast path: each verbatim ``search_for`` hit rect is one single-box occurrence."""
    return [[(r.x0, r.y0, r.x1, r.y1)] for r in page.search_for(query)]


def _token_locations(page: pymupdf.Page, query: str) -> list[list[_Rect]]:
    """Normalized word-level fallback. Implemented in Task 3."""
    return []


def _dedup_key(page: int, boxes: list[_Box]) -> tuple:
    return (
        page,
        tuple(
            sorted(
                (
                    round(b["points"]["x0"]),
                    round(b["points"]["y0"]),
                    round(b["points"]["x1"]),
                    round(b["points"]["y1"]),
                )
                for b in boxes
            )
        ),
    )


def locate_text(
    doc: pymupdf.Document,
    query: str | Sequence[str],
    *,
    page: int | None = None,
    max_results: int = 20,
) -> list[TextLocation]:
    """Map a query string (or list of strings) to bounding boxes on a page.

    Returns one ``TextLocation`` per occurrence; grounding is string-level, not
    hit-level (see the design spec). Not found -> ``[]``.
    """
    if isinstance(query, str):
        patterns = [query]
        tag_pattern = False
    else:
        patterns = list(query)
        tag_pattern = True

    cleaned = [p.strip() for p in patterns]
    if not cleaned or not any(cleaned):
        raise ValueError("query must not be empty")
    if max_results < 1:
        raise ValueError("max_results must be at least 1")

    total_pages = len(doc)
    if page is not None and (page < 1 or page > total_pages):
        raise ValueError(f"page must be between 1 and {total_pages}")
    target_pages = [page] if page is not None else range(1, total_pages + 1)

    results: list[TextLocation] = []
    seen: set[tuple] = set()
    for pattern in cleaned:
        if not pattern:
            continue
        if len(results) >= max_results:
            break
        for page_number in target_pages:
            if len(results) >= max_results:
                break
            page_obj = doc[page_number - 1]
            page_rect = page_obj.rect

            occurrences = _search_for_occurrences(page_obj, pattern)
            method = "search_for"
            if not occurrences:
                occurrences = _token_locations(page_obj, pattern)
                method = "tokens"

            for occurrence in occurrences:
                boxes = [_box_from_rect(rect, page_rect) for rect in occurrence]
                if not boxes:
                    continue
                key = _dedup_key(page_number, boxes)
                if key in seen:
                    continue
                seen.add(key)
                region = (
                    boxes[0] if len(boxes) == 1 else _union_region(boxes, page_rect)
                )
                location: TextLocation = {
                    "page": page_number,
                    "match_method": method,
                    "page_width": page_rect.width,
                    "page_height": page_rect.height,
                    "region": region,
                    "boxes": boxes,
                }
                if tag_pattern:
                    location["pattern"] = pattern
                results.append(location)
                if len(results) >= max_results:
                    break
    return results
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_locate.py -q && uv run ruff check src/datasheetindex/core/locate.py && uv run ty check src/datasheetindex/core/locate.py`
Expected: tests PASS; ruff and ty report no errors.

- [ ] **Step 5: Commit**

```bash
git add src/datasheetindex/core/locate.py tests/test_locate.py
git commit -m "feat: add locate_text fast path (search_for) with list/dedup"
```

---

## Task 3: Word-level token fallback

**Files:**
- Modify: `src/datasheetindex/core/locate.py`
- Test: `tests/test_locate.py`

- [ ] **Step 1: Write the failing fallback + real-fixture tests**

First add `from pathlib import Path` to the imports at the top of `tests/test_locate.py`, and these module-level constants below the imports (matching `tests/test_vision.py` / `tests/test_index.py`):

```python
DATA2PAGE_DIR = Path(__file__).resolve().parent.parent.parent / "data2page"
TLE9350_PATH = DATA2PAGE_DIR / "Infineon-TLE9350BSJ-DataSheet-v01_00-EN.pdf"
```

Then append these tests:

```python
def test_dash_mismatch_falls_back_to_tokens():
    # PDF text uses a Unicode minus (U+2212); the ASCII-hyphen query only
    # matches via the normalizing token fallback.
    minus = chr(0x2212)
    doc = _doc_with([(72, 72, f"{minus}0.3")])
    results = locate_text(doc, "-0.3", page=1)
    doc.close()

    assert len(results) == 1
    assert results[0]["match_method"] == "tokens"
    assert len(results[0]["boxes"]) == 1


def test_multi_line_phrase_unions_boxes_via_tokens():
    # A phrase wrapping across two ADJACENT lines: "range -9" then "stop" one
    # line below. The Unicode minus forces the token path, which groups the
    # matched words by (block_no, line_no) into one box per line.
    minus = chr(0x2212)
    doc = _doc_with([(72, 72, f"range {minus}9"), (72, 94, "stop")])
    results = locate_text(doc, "range -9 stop", page=1)
    doc.close()

    assert len(results) == 1
    loc = results[0]
    assert loc["match_method"] == "tokens"
    assert len(loc["boxes"]) == 2
    # region is the union: it spans from the top line to the bottom line.
    assert loc["region"]["points"]["y0"] == pytest.approx(
        min(b["points"]["y0"] for b in loc["boxes"])
    )
    assert loc["region"]["points"]["y1"] == pytest.approx(
        max(b["points"]["y1"] for b in loc["boxes"])
    )
    assert loc["region"] != loc["boxes"][0]


def test_real_fixture_locates_part_number():
    # Real-world text/word structure (spec testing-plan case 10). The part
    # number is in the document title, so it is present on page 1.
    if not TLE9350_PATH.exists():
        pytest.skip("Test PDF not found")
    doc = pymupdf.open(str(TLE9350_PATH))
    results = locate_text(doc, "TLE9350", page=1)
    doc.close()

    assert results, "expected to locate the part number on page 1"
    loc = results[0]
    assert loc["page"] == 1
    assert len(loc["boxes"]) >= 1
    assert 0.0 <= loc["region"]["pct"]["left"] <= 1.0
    assert 0.0 <= loc["region"]["pct"]["bottom"] <= 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_locate.py::test_dash_mismatch_falls_back_to_tokens tests/test_locate.py::test_multi_line_phrase_unions_boxes_via_tokens -q`
Expected: FAIL (the stub returns `[]`, so both find nothing).

- [ ] **Step 3: Implement the fallback**

In `src/datasheetindex/core/locate.py`, add this import below the `from collections.abc import Sequence` line:

```python
from datasheetindex.core._textmatch import (
    _TOKEN_RE,
    _TokenSpan,
    _match_query_tokens,
    _normalize_token,
)
```

Replace the `_token_locations` stub with:

```python
def _group_words_by_line(words: list[tuple]) -> list[_Rect]:
    """Group matched words into one rect per (block_no, line_no), in match order."""
    groups: dict[tuple[int, int], list[float]] = {}
    order: list[tuple[int, int]] = []
    for word in words:
        key = (word[5], word[6])  # (block_no, line_no); line_no is block-scoped
        if key not in groups:
            groups[key] = [word[0], word[1], word[2], word[3]]
            order.append(key)
        else:
            box = groups[key]
            box[0] = min(box[0], word[0])
            box[1] = min(box[1], word[1])
            box[2] = max(box[2], word[2])
            box[3] = max(box[3], word[3])
    return [tuple(groups[key]) for key in order]


def _token_locations(page: pymupdf.Page, query: str) -> list[list[_Rect]]:
    """Normalized word-level fallback: dash/case/whitespace-tolerant matching."""
    query_tokens = [
        token
        for token in (
            _normalize_token(raw, case_sensitive=False)
            for raw in _TOKEN_RE.findall(query)
        )
        if token
    ]
    if not query_tokens:
        return []

    page_spans: list[_TokenSpan] = []
    word_refs: list[tuple] = []
    for word in page.get_text("words"):
        normalized = _normalize_token(word[4], case_sensitive=False)
        if not normalized:
            continue
        index = len(page_spans)
        page_spans.append(_TokenSpan(normalized, index, index + 1))
        word_refs.append(word)

    # Same short-circuits as the search ladder, minus the "< 3 tokens" guard.
    if len(page_spans) < len(query_tokens):
        return []
    page_values = {span.value for span in page_spans}
    if not set(query_tokens).issubset(page_values):
        return []

    max_gap_tokens = max(8, len(query_tokens) * 2)
    occurrences: list[list[_Rect]] = []
    for start in range(len(page_spans)):
        if page_spans[start].value != query_tokens[0]:
            continue
        matched = _match_query_tokens(
            page_spans, query_tokens, start, max_gap_tokens=max_gap_tokens
        )
        if matched is None:
            continue
        occurrences.append(
            _group_words_by_line([word_refs[index] for index in matched])
        )
    return occurrences
```

- [ ] **Step 4: Run to verify pass (full file, no regressions)**

Run: `uv run pytest tests/test_locate.py -q && uv run ruff check src/datasheetindex/core/locate.py && uv run ty check src/datasheetindex/core/locate.py`
Expected: all PASS; ruff and ty clean.

- [ ] **Step 5: Commit**

```bash
git add src/datasheetindex/core/locate.py tests/test_locate.py
git commit -m "feat: add locate_text word-level token fallback"
```

---

## Task 4: Wire `locate_text` into `DatasheetTools` and the Agent SDK server

**Files:**
- Modify: `src/datasheetindex/tools/registry.py`
- Test: `tests/test_registry.py:289` (exact tool set) and a new method test

- [ ] **Step 1: Update the failing exact-set test + add a method test**

In `tests/test_registry.py`, update the assertion inside `test_create_server_registers_tools` to include the new tool:

```python
    assert set(server.tools) == {
        "build_datasheet",
        "get_section_text",
        "search_text",
        "inspect_page",
        "extract_table_markdown",
        "locate_text",
    }
```

Also update that test's docstring count if present ("register 5 agent-ready tools" -> "register 6 agent-ready tools").

Then, inside `test_create_server_registers_tools`, after the `inspect_result` assertion, smoke-test the new handler through its registered async wrapper (the synthetic PDF in that test contains "Registry MCP test", and `build_datasheet` has already bound the document):

```python
    import json

    locate_result = asyncio.run(
        server.tools["locate_text"]({"query": "Registry", "page": 1})
    )
    assert locate_result["is_error"] is False
    locate_payload = json.loads(locate_result["content"][0]["text"])
    assert locate_payload["results"], "SDK locate_text returned no results"
    assert locate_payload["results"][0]["match_method"] == "search_for"
```

Append a new test:

```python
def test_datasheet_tools_locate_text_without_build(tmp_path):
    pdf_path = tmp_path / "locate.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    writer = pymupdf.TextWriter(page.rect)
    writer.append((72, 72), "Hello world")
    writer.write_text(page)
    doc.save(str(pdf_path))
    doc.close()

    tools = DatasheetTools(str(pdf_path))
    results = tools.locate_text("Hello")  # no build_datasheet first
    tools.close()

    assert len(results) == 1
    assert results[0]["page"] == 1
    assert results[0]["match_method"] == "search_for"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_registry.py::test_create_server_registers_tools tests/test_registry.py::test_datasheet_tools_locate_text_without_build -q`
Expected: FAIL (`locate_text` not in the set; `DatasheetTools` has no `locate_text`).

- [ ] **Step 3: Implement the method and SDK tool**

In `src/datasheetindex/tools/registry.py`:

1. Add to the imports near the top (after the existing `from datasheetindex.core.textfile import ...` lines):

```python
from datasheetindex.core.locate import locate_text as locate_text_core
```

2. Add this method to `DatasheetTools`, directly after `inspect_page`:

```python
    def locate_text(
        self,
        query: str | list[str],
        *,
        page: int | None = None,
        max_results: int = 20,
    ) -> list[dict]:
        """Map a string to its bounding box(es) on a page.

        Works off the live PDF (`self.doc`); unlike `search_text`/`get_section_text`
        it does NOT require `build_datasheet` to have been called.
        """
        return locate_text_core(
            self.doc, query, page=page, max_results=max_results
        )
```

3. Inside `create_datasheet_tools_server`, register a new tool (place it after the `inspect_page` `@tool` block, before `extract_table_markdown`):

```python
    @tool(
        "locate_text",
        "Map a piece of text to its bounding-box coordinates on a page, for "
        "highlighting or precise visual inspection. Returns one result per "
        "occurrence; each has `region` (the union rectangle) and `boxes` "
        "(one per line). Feed region['pct'] into inspect_page(region=...) to "
        "crop to the exact spot; use region['points'] (PDF points) to annotate "
        "the PDF. Pass `page` when you know it (e.g. from a search_text hit) to "
        "stay cheap; omit it to scan all pages. `query` may be a single string "
        "or a list of strings.",
        {
            "type": "object",
            "properties": {
                "query": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                    "description": "A single pattern or a list of patterns.",
                },
                "page": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "1-indexed page to locate on. Omit to scan all.",
                },
                "max_results": {"type": "integer"},
            },
            "required": ["query"],
        },
    )
    async def locate_text(args: dict[str, Any]) -> dict[str, Any]:
        try:
            results = _require().locate_text(
                args["query"],
                page=args.get("page"),
                max_results=args.get("max_results", 20),
            )
            return _ok({"query": args["query"], "results": results})
        except Exception as exc:
            return _err(str(exc))
```

4. Add `locate_text` to the `tools=[...]` list in `create_sdk_mcp_server(...)`:

```python
        tools=[
            build_datasheet,
            get_section_text,
            search_text,
            inspect_page,
            locate_text,
            extract_table_markdown,
        ],
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_registry.py -q && uv run ruff check src/datasheetindex/tools/registry.py && uv run ty check src/datasheetindex/tools/registry.py`
Expected: all PASS; ruff and ty clean.

- [ ] **Step 5: Commit**

```bash
git add src/datasheetindex/tools/registry.py tests/test_registry.py
git commit -m "feat: expose locate_text on DatasheetTools and the Agent SDK server"
```

---

## Task 5: Wire `locate_text` into the local MCP server + fix `inspect_page_tool` guard

**Files:**
- Modify: `src/datasheetindex/mcp_server.py`
- Test: `tests/test_mcp_server.py`

- [ ] **Step 1: Update the failing tests**

In `tests/test_mcp_server.py`, inside `test_create_local_mcp_server_registers_inspect_page`:

1. Add `locate_text` to the exact-set assertion:

```python
    assert set(server.registered_tools) == {
        "build_datasheet",
        "get_section_text",
        "inspect_page",
        "search_text",
        "extract_table_markdown",
        "locate_text",
    }
```

2. Add a `locate_text` entry to the `fake_tools` `SimpleNamespace` (alongside the others):

```python
        locate_text=lambda query, page=None, max_results=20: [
            {
                "page": 1,
                "match_method": "search_for",
                "page_width": 612.0,
                "page_height": 792.0,
                "region": {
                    "pct": {"top": 0.0, "bottom": 0.1, "left": 0.0, "right": 0.1},
                    "points": {"x0": 0.0, "y0": 0.0, "x1": 61.2, "y1": 79.2},
                },
                "boxes": [
                    {
                        "pct": {"top": 0.0, "bottom": 0.1, "left": 0.0, "right": 0.1},
                        "points": {"x0": 0.0, "y0": 0.0, "x1": 61.2, "y1": 79.2},
                    }
                ],
            }
        ],
```

3. After the existing `extract_table_markdown` assertion block, add a smoke check:

```python
    locate_result = server.registered_tools["locate_text"]["func"](
        query="Hello", page=1, ctx=ctx
    )
    assert locate_result["query"] == "Hello"
    assert locate_result["results"][0]["match_method"] == "search_for"
```

Then add a new test for the `inspect_page_tool` guard fix:

```python
def test_inspect_page_tool_without_datasheet_raises(monkeypatch):
    _install_fake_mcp(monkeypatch)

    from datasheetindex.mcp_server import create_local_mcp_server

    server = create_local_mcp_server()
    server_ctx = types.SimpleNamespace(tools=None)
    ctx = types.SimpleNamespace(
        request_context=types.SimpleNamespace(lifespan_context=server_ctx)
    )
    func = server.registered_tools["inspect_page"]["func"]
    with pytest.raises(RuntimeError, match="No datasheet loaded"):
        func(page=1, ctx=ctx)
```

(If `pytest` is not already imported at the top of the file, add `import pytest`.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_mcp_server.py -q`
Expected: FAIL (`locate_text` missing from the set; new tests error; current `inspect_page_tool` raises `AttributeError`, not the matched `RuntimeError`).

- [ ] **Step 3: Implement in `mcp_server.py`**

In `src/datasheetindex/mcp_server.py`:

1. Fix `inspect_page_tool` — replace its `if ctx is None: ...` guard and direct dereference with `_require_tools`:

```python
        blocks = _require_tools(ctx).inspect_page(
            page, region=region, dpi=dpi, detail=detail
        )
```

(Delete the preceding `if ctx is None:\n    raise RuntimeError("MCP context was not provided")` lines in that function — `_require_tools` already handles `ctx is None`.)

2. Add a `locate_text_tool` nested function (place it after `search_text_tool`):

```python
    def locate_text_tool(
        query: str | list[str],
        page: int | None = None,
        max_results: int = 20,
        ctx: Context[ServerSession, _ServerContext] | None = None,
    ) -> dict[str, object]:
        """Map a string to bounding-box coordinates on a page."""
        tools = _require_tools(ctx)
        return {
            "query": query,
            "results": tools.locate_text(
                query, page=page, max_results=max_results
            ),
        }
```

3. Register it (place after the `inspect_page` `server.tool(...)` block):

```python
    server.tool(
        name="locate_text",
        description=(
            "Map a piece of text to its bounding-box coordinates on a page, "
            "for highlighting or precise visual inspection. Returns one result "
            "per occurrence, each with 'region' (the union rectangle) and "
            "'boxes' (one per line), in both percentages and PDF points. Feed "
            "region['pct'] into inspect_page(region=...) to crop to the exact "
            "spot; use region['points'] to annotate the PDF. Pass 'page' when "
            "you know it (e.g. from a search_text hit) to stay cheap; omit to "
            "scan all pages."
        ),
    )(locate_text_tool)
```

4. Extend the `instructions=` string in `create_local_mcp_server` (append before the closing paren of the string) so the surface advertises the tool:

```python
            " Use locate_text to get the bounding-box coordinates of a string "
            "on a page (for highlighting or to crop inspect_page precisely)."
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_mcp_server.py -q && uv run ruff check src/datasheetindex/mcp_server.py && uv run ty check src/datasheetindex/mcp_server.py`
Expected: all PASS; ruff and ty clean.

- [ ] **Step 5: Commit**

```bash
git add src/datasheetindex/mcp_server.py tests/test_mcp_server.py
git commit -m "feat: register locate_text on local MCP server; harden inspect_page guard"
```

---

## Task 6: Document the tool in the architecture doc

**Files:**
- Modify: `docs/datasheetindex_architecture.md`

- [ ] **Step 1: Reword the "Agent Tools" intro (drop the hardcoded count)**

In `docs/datasheetindex_architecture.md`, in the `## Agent Tools` section, replace:

```markdown
The agent has one custom tool beyond the built-in file reading capabilities of the Claude Agent SDK.
```

with a count-free phrasing (so it does not go stale again as tools are added):

```markdown
Beyond the built-in file reading capabilities of the Claude Agent SDK, the agent has custom PDF-native inspection tools: `inspect_page` (visual inspection) and `locate_text` (text-to-coordinate grounding).
```

- [ ] **Step 2: Add a `locate_text` subsection under "Agent Tools"**

After the `inspect_page` section (before "What the Library Does NOT Do"), add:

```markdown
### `locate_text`

Maps a query string to its bounding box(es) on a page — the bridge between
`search_text` (find text) and `inspect_page` (render a region). It returns one
result per occurrence, each carrying `region` (the union rectangle) and `boxes`
(one per line), expressed in **both** normalized percentages (0.0-1.0, so they
feed straight into `inspect_page(region=...)`) and raw PDF points (for
annotating the PDF directly), plus page dimensions.

Matching is hybrid: `page.search_for` on the verbatim query (fast path), with a
normalized word-level fallback (`page.get_text("words")`) that tolerates the
dash/case/whitespace variation endemic to datasheets (`-0.3` vs `−0.3`, `±2%`).
It is stateless and works off the live PDF — no `build_datasheet` required.

Grounding is string-level, not hit-level: a string appearing multiple times on a
page returns multiple candidate results; disambiguate with a more specific query.
```

- [ ] **Step 3: Qualify the "Why only one tool?" heading**

That rationale is specifically about the dropped vision/table tools. Replace:

```markdown
**Why only one tool?** We evaluated and dropped three other tools during design:
```

with:

```markdown
**Why only one *vision/table* tool?** We evaluated and dropped three other tools during design:
```

- [ ] **Step 4: Update the registry wiring prose (MCP / SDK Integration section)**

Replace:

```markdown
`DatasheetTools` instance and registers `build_datasheet`, `get_section_text`,
`search_text`, `inspect_page`, and `extract_table_markdown` on a `ToolServer`.
```

with:

```markdown
`DatasheetTools` instance and registers `build_datasheet`, `get_section_text`,
`search_text`, `inspect_page`, `locate_text`, and `extract_table_markdown` on a
`ToolServer`.
```

- [ ] **Step 5: Update the "What we add" agent-tools summary bullet**

Replace:

```markdown
- **Agent tools** — `build_datasheet`, `get_section_text`, `search_text`,
  `inspect_page`, and `extract_table_markdown`, with text-first navigation,
```

with:

```markdown
- **Agent tools** — `build_datasheet`, `get_section_text`, `search_text`,
  `inspect_page`, `locate_text`, and `extract_table_markdown`, with text-first navigation,
```

- [ ] **Step 6: Add a bullet under "What we add"**

In the "What we add" list (in the "Building on PageIndex" section), add:

```markdown
- **`locate_text`** — text-to-coordinate source grounding (bounding boxes as
  percentages + PDF points), so an agent or review UI can turn a located string
  into a precise highlight or a tightly cropped `inspect_page` call
```

- [ ] **Step 7: Update the Phase 2 roadmap summary**

In the "Implementation Priority" section, under `### Phase 2: Agent Tools` (which currently lists only `inspect_page`), add a bullet:

```markdown
- `locate_text` — text-to-coordinate grounding (bounding boxes for highlighting)
```

- [ ] **Step 8: Update the module structure diagram**

In the `## Module Structure` tree, add these two lines under `core/` (immediately after the `textfile.py` entry), keeping the existing tree-drawing characters aligned:

```
│   ├── _textmatch.py      # Shared dash/token normalization + matcher
│   ├── locate.py          # locate_text: text -> bounding-box coordinates
```

- [ ] **Step 9: Mention `locate_text` in the reference system-prompt block**

In the "Agent System Prompt Guidance" block, replace:

```
3. MCP tools that can build and read the artifacts, search the extracted text,
   and call `inspect_page` for visual inspection
```

with:

```
3. MCP tools that can build and read the artifacts, search the extracted text,
   locate a string's coordinates (`locate_text`), and call `inspect_page` for
   visual inspection
```

- [ ] **Step 10: Commit**

```bash
git add docs/datasheetindex_architecture.md
git commit -m "docs: document locate_text in the architecture doc"
```

---

## Task 7: CHANGELOG entry, version bump, full-suite verification

**Files:**
- Modify: `CHANGELOG.md`, `pyproject.toml`

- [ ] **Step 1: Bump the version**

In `pyproject.toml`, change:

```toml
version = "0.15.0"
```

- [ ] **Step 2: Add the CHANGELOG entry**

In `CHANGELOG.md`, insert directly below the `# Changelog` preamble (above `## [0.14.0] - 2026-06-02`):

```markdown
## [0.15.0] - 2026-06-15

### Added
- **`locate_text` source grounding.** New tool that maps a query string to its bounding box(es) on a page, returning one result per occurrence with `region` (union rectangle) and `boxes` (one per line) in both normalized percentages and PDF points, plus page dimensions. Matching is hybrid: verbatim `page.search_for` with a normalized word-level token fallback (dash/case/whitespace tolerant). Stateless and works off the live PDF (no `build_datasheet` required). Exposed on both the Agent SDK and local MCP tool surfaces.

### Changed
- **`inspect_page` on the local MCP server now raises a clean "No datasheet loaded" error** (via `_require_tools`) instead of an `AttributeError` when called before `build_datasheet`.
- **Shared text normalization extracted to `core/_textmatch.py`** (dash translation, token normalization, subsequence matcher), used by both `search_text` and `locate_text`. No behavior change to `search_text`.
```

- [ ] **Step 3: Run the full suite and all checks**

Run: `uv run pytest -q 2>&1 | tee /tmp/locate-full-suite.log && uv run ruff check . && uv run ty check`
Expected: entire suite PASSES; ruff and ty clean.

- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md pyproject.toml
git commit -m "chore: bump version to 0.15.0 and update changelog"
```

---

## Self-Review Checklist (completed during planning)

- **Spec coverage:** locate_text core (Tasks 2-3), pct+points+dims contract (Task 2), hybrid match with verbatim fast path + normalized fallback (Tasks 2-3), occurrence grouping by `(block_no, line_no)` and union `region` (Task 3), `_textmatch` extraction with `_TOKEN_RE` and preserved gap/subset short-circuits minus the `<3` guard (Tasks 1, 3), both tool surfaces + both exact-set assertions + both handler smoke tests — SDK in Task 4, local MCP in Task 5, `inspect_page_tool` guard cleanup (Task 5), no-build behavior (Task 4), dedup key (Task 2), error handling (Task 2), **real-fixture smoke test (`TLE9350_PATH`, Task 3, spec case 10)**, architecture-doc de-staling of all tool counts/enumerations + CHANGELOG/version (Tasks 6-7). All covered.
- **Placeholder scan:** none — every code/step has concrete content.
- **Type consistency:** `locate_text`, `TextLocation`, `_Box`, `region`/`boxes`/`match_method`/`page_width`/`page_height`/`pattern`, `_token_locations`, `_search_for_occurrences`, `_box_from_rect`, `_union_region`, `_group_words_by_line`, `_dedup_key`, `locate_text_core` are consistent across tasks.
