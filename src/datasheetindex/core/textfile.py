"""PDF to page-matched text file generation."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, NamedTuple, NotRequired, TypedDict

from datasheetindex.core._textmatch import (
    _TOKEN_RE,
    _match_query_tokens,
    _normalize_token,
    _TokenSpan,
    _translate_search_text,
)
from datasheetindex.core.figures import (
    DEFAULT_MIN_AREA_PCT,
    caption_entries,
    raster_regions,
)
from datasheetindex.core.furniture import (
    detect_furniture,
    is_candidate,
    normalize_key,
)

if TYPE_CHECKING:
    import pymupdf

logger = logging.getLogger(__name__)


class TextSearchMatch(TypedDict):
    """A text match located within a page of the extracted text artifact."""

    page: int
    start: int
    end: int
    snippet: str
    # Set only in multi-pattern searches: which query pattern produced this hit.
    pattern: NotRequired[str]
    # Attached by the tool layer: ToC breadcrumb of the section containing the page.
    breadcrumb: NotRequired[str]


_PAGE_MARKER_RE = re.compile(r"--- PAGE (\d+) ---")


class _PageText(NamedTuple):
    page: int
    text: str


class _CollapsedText(NamedTuple):
    text: str
    index_map: tuple[int, ...]


class _TokenIndex(NamedTuple):
    spans: tuple[_TokenSpan, ...]
    values: frozenset[str]


# ---------------------------------------------------------------------------
# Block-level text extraction with column detection
# ---------------------------------------------------------------------------

# A text block tuple from page.get_text("blocks"):
# (x0, y0, x1, y1, text, block_no, block_type)
_BLOCK_X0 = 0
_BLOCK_Y0 = 1
_BLOCK_X1 = 2
_BLOCK_Y1 = 3
_BLOCK_TEXT = 4
_BLOCK_TYPE = 6

# Column detection thresholds (fractions of page width)
_MIN_COL_WIDTH_FRAC = 0.25  # each column >= 25% of page width
_MAX_COL_WIDTH_FRAC = 0.55  # each column <= 55% of page width
_MAX_GUTTER_FRAC = 0.20  # gutter <= 20% of page width
_WIDE_BLOCK_FRAC = 0.60  # blocks > 60% are "wide" (not column content)
_GUTTER_TOLERANCE_FRAC = 0.10  # gutter positions must agree within 10%

# Minimum gap in points between two blocks to qualify as a gutter
_MIN_GUTTER_PTS = 10

# Fraction of a page's height, at each edge, within which a block may be
# running furniture. Applied per page, so landscape and mixed-size pages need
# no special case. This band is what separates a running header from a
# "Table N" caption: measured on the PSoC, "Table #" recurs 89 times but never
# inside the band.
_FURNITURE_BAND_FRAC = 0.20

# Height thresholds for confidence tiers (in points; 1 pt ~ 1/72 inch)
_HIGH_CONFIDENCE_HEIGHT = 80  # ~7 lines at 11pt
_MEDIUM_CONFIDENCE_HEIGHT = 40  # ~3-4 lines at 11pt
_MEDIUM_CONFIDENCE_MIN_PAIRS = 2


def _detect_columns(
    text_blocks: list[tuple[Any, ...]],
    page_width: float,
) -> tuple[float, float, float] | None:
    """Detect a two-column layout from text block positions.

    Returns ``(gutter_x, col_top, col_bottom)`` when a two-column layout
    is detected, or ``None`` for single-column pages.
    """
    min_w = page_width * _MIN_COL_WIDTH_FRAC
    max_w = page_width * _MAX_COL_WIDTH_FRAC
    max_gutter = page_width * _MAX_GUTTER_FRAC
    gutter_tol = page_width * _GUTTER_TOLERANCE_FRAC

    pairs: list[tuple[float, float, float, float]] = []  # (gutter_x, min_h, top, bot)

    for i, b1 in enumerate(text_blocks):
        w1 = b1[_BLOCK_X1] - b1[_BLOCK_X0]
        if w1 < min_w or w1 > max_w:
            continue
        for b2 in text_blocks[i + 1 :]:
            w2 = b2[_BLOCK_X1] - b2[_BLOCK_X0]
            if w2 < min_w or w2 > max_w:
                continue
            # Y overlap required
            y_overlap = min(b1[_BLOCK_Y1], b2[_BLOCK_Y1]) - max(
                b1[_BLOCK_Y0], b2[_BLOCK_Y0]
            )
            if y_overlap <= 0:
                continue
            # Determine left/right and check gutter
            if b1[_BLOCK_X1] < b2[_BLOCK_X0] - _MIN_GUTTER_PTS:
                gap = b2[_BLOCK_X0] - b1[_BLOCK_X1]
                gutter_x = (b1[_BLOCK_X1] + b2[_BLOCK_X0]) / 2
            elif b2[_BLOCK_X1] < b1[_BLOCK_X0] - _MIN_GUTTER_PTS:
                gap = b1[_BLOCK_X0] - b2[_BLOCK_X1]
                gutter_x = (b2[_BLOCK_X1] + b1[_BLOCK_X0]) / 2
            else:
                continue
            if gap > max_gutter:
                continue
            min_h = min(
                b1[_BLOCK_Y1] - b1[_BLOCK_Y0],
                b2[_BLOCK_Y1] - b2[_BLOCK_Y0],
            )
            top = min(b1[_BLOCK_Y0], b2[_BLOCK_Y0])
            bot = max(b1[_BLOCK_Y1], b2[_BLOCK_Y1])
            pairs.append((gutter_x, min_h, top, bot))

    if not pairs:
        return None

    # Confidence tiers
    tall = [p for p in pairs if p[1] >= _HIGH_CONFIDENCE_HEIGHT]
    medium = [p for p in pairs if p[1] >= _MEDIUM_CONFIDENCE_HEIGHT]

    if tall:
        selected = tall
    elif len(medium) >= _MEDIUM_CONFIDENCE_MIN_PAIRS:
        selected = medium
    else:
        return None

    # Gutter consistency check
    gutters = [p[0] for p in selected]
    avg_gutter = sum(gutters) / len(gutters)
    if not all(abs(g - avg_gutter) < gutter_tol for g in gutters):
        return None

    col_top = min(p[2] for p in selected)
    col_bottom = max(p[3] for p in selected)
    return (avg_gutter, col_top, col_bottom)


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
    return [(b[_BLOCK_TEXT], _is_banded(b, page_height)) for b in _ordered_blocks(page)]


def furniture_enabled_by_env() -> bool:
    """Whether DATASHEETINDEX_FURNITURE permits header/footer stripping.

    Accepts the spellings a user actually reaches for, for the reason
    ``structure._parallel_enabled_by_env`` records: matching only the literal
    "0" would silently ignore ``DATASHEETINDEX_FURNITURE=false``, leaving the
    escape hatch looking broken to the person who most needs it.

    Public despite living beside private helpers, because it has a second
    consumer outside this module: ``tools/bound.py`` puts its answer in
    ``_BuildOptions``, so the setting participates in both artifact-reuse
    keys. Without that, flipping the hatch served the previously built text
    file from cache and the hatch looked inert.
    """
    value = os.environ.get("DATASHEETINDEX_FURNITURE", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _is_furniture_block(text: str, banded: bool, furniture: frozenset[str]) -> bool:
    """Whether one block is running furniture, and so dropped from the text.

    The drop decision in **one** place. It was previously spelled out inline
    in ``scan_pages`` and hand-copied into the ONNX oracle precision test,
    which is this feature's principal safety evidence -- so a change to the
    gate left the oracle measuring the old rule and staying green. The test
    now calls this.

    ``furniture`` empty means detection found nothing, or the escape hatch is
    set; either way nothing is dropped.
    """
    if not furniture or not banded:
        return False
    return is_candidate(text) and normalize_key(text) in furniture


@dataclass(frozen=True)
class PageScan:
    """One pass over the document: the text file plus the figure index.

    ``text`` is byte-identical to what ``generate_text`` produced before this
    type existed; the seven stubs in ``tests/test_index.py`` depend on that.
    """

    text: str
    figures: list[dict[str, object]]
    excluded_below_min_area: int


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
    stripping = furniture_enabled_by_env()

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
            if stripping
            else set()
        )
        rasters, page_excluded = raster_regions(page, min_area_pct=min_area_pct)
        page_rasters.append(rasters)
        excluded += page_excluded

    furniture = detect_furniture(page_keys, total_pages) if stripping else frozenset()

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
            if _is_furniture_block(text, banded, furniture):
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


def generate_text(doc: pymupdf.Document) -> str:
    """Generate page-matched text with PAGE markers.

    Retained wrapper: the page-matched text file alone. Each page's text is
    preceded by a ``--- PAGE N ---`` marker where N is 1-indexed (matching
    human PDF page numbers).
    """
    return scan_pages(doc).text


def extract_section_text(text_content: str, start_page: int, end_page: int) -> str:
    """Extract text between two page markers (inclusive).

    Looks for ``--- PAGE N ---`` markers and returns everything from
    the start_page marker to the end_page+1 marker (or end of text).
    """
    start_pattern = f"--- PAGE {start_page} ---"
    start_idx = text_content.find(start_pattern)
    if start_idx == -1:
        return ""

    end_pattern = f"--- PAGE {end_page + 1} ---"
    end_idx = text_content.find(end_pattern, start_idx)
    if end_idx == -1:
        return text_content[start_idx:]

    return text_content[start_idx:end_idx]


def extract_page_text(text_content: str, page: int) -> str:
    """Extract text for a single page marker, excluding the marker itself."""
    section = extract_section_text(text_content, page, page)
    if not section:
        return ""

    marker = f"--- PAGE {page} ---"
    if section.startswith(marker):
        section = section[len(marker) :]
    return section.lstrip("\r\n").rstrip()


@lru_cache(maxsize=4)
def _iter_page_text(text_content: str) -> tuple[_PageText, ...]:
    matches = list(_PAGE_MARKER_RE.finditer(text_content))
    pages: list[_PageText] = []
    for index, match in enumerate(matches):
        page = int(match.group(1))
        start = match.end()
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(text_content)
        )
        pages.append(_PageText(page, text_content[start:end].lstrip("\r\n").rstrip()))
    return tuple(pages)


def _build_snippet(page_text: str, start: int, end: int, context_chars: int) -> str:
    snippet_start = max(0, start - context_chars)
    snippet_end = min(len(page_text), end + context_chars)
    snippet = " ".join(page_text[snippet_start:snippet_end].split())
    if snippet_start > 0:
        snippet = f"...{snippet}"
    if snippet_end < len(page_text):
        snippet = f"{snippet}..."
    return snippet


def _compile_literal_pattern(query: str, *, case_sensitive: bool) -> re.Pattern[str]:
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(re.escape(_translate_search_text(query)), flags)


def _find_pattern_spans(
    searchable_text: str,
    pattern: re.Pattern[str],
    *,
    max_results: int,
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in pattern.finditer(searchable_text):
        spans.append((match.start(), match.end()))
        if len(spans) >= max_results:
            break
    return spans


def _find_literal_spans(
    page_text: str,
    pattern: re.Pattern[str],
    *,
    max_results: int,
) -> list[tuple[int, int]]:
    return _find_pattern_spans(
        _translate_search_text(page_text),
        pattern,
        max_results=max_results,
    )


@lru_cache(maxsize=256)
def _collapse_whitespace(text: str) -> _CollapsedText:
    collapsed_chars: list[str] = []
    index_map: list[int] = []
    in_whitespace = False
    for index, char in enumerate(text):
        if char.isspace():
            if in_whitespace:
                continue
            collapsed_chars.append(" ")
            index_map.append(index)
            in_whitespace = True
            continue

        collapsed_chars.append(char)
        index_map.append(index)
        in_whitespace = False

    return _CollapsedText("".join(collapsed_chars), tuple(index_map))


def _find_collapsed_whitespace_spans(
    page_text: str,
    pattern: re.Pattern[str],
    *,
    max_results: int,
) -> list[tuple[int, int]]:
    collapsed_text = _collapse_whitespace(page_text)
    spans: list[tuple[int, int]] = []
    searchable_text = _translate_search_text(collapsed_text.text)
    for match in pattern.finditer(searchable_text):
        raw_start = collapsed_text.index_map[match.start()]
        raw_end = (
            collapsed_text.index_map[match.end()]
            if match.end() < len(collapsed_text.index_map)
            else len(page_text)
        )
        spans.append((raw_start, raw_end))
        if len(spans) >= max_results:
            break
    return spans


@lru_cache(maxsize=512)
def _iter_token_spans(
    text: str,
    *,
    case_sensitive: bool,
) -> tuple[_TokenSpan, ...]:
    tokens: list[_TokenSpan] = []
    for match in _TOKEN_RE.finditer(text):
        normalized = _normalize_token(match.group(0), case_sensitive=case_sensitive)
        if not normalized:
            continue
        tokens.append(_TokenSpan(normalized, match.start(), match.end()))
    return tuple(tokens)


@lru_cache(maxsize=512)
def _build_token_index(
    text: str,
    *,
    case_sensitive: bool,
) -> _TokenIndex:
    spans = _iter_token_spans(text, case_sensitive=case_sensitive)
    return _TokenIndex(spans, frozenset(token.value for token in spans))


def _find_token_sequence_spans(
    page_index: _TokenIndex,
    query_index: _TokenIndex,
    *,
    max_results: int,
) -> list[tuple[int, int]]:
    if len(query_index.spans) < 3:
        return []

    if len(page_index.spans) < len(query_index.spans):
        return []
    if not query_index.values.issubset(page_index.values):
        return []

    query_tokens = [token.value for token in query_index.spans]
    max_gap_tokens = max(8, len(query_tokens) * 2)
    spans: list[tuple[int, int]] = []
    for index, token in enumerate(page_index.spans):
        if token.value != query_tokens[0]:
            continue
        matched_indices = _match_query_tokens(
            page_index.spans,
            query_tokens,
            index,
            max_gap_tokens=max_gap_tokens,
        )
        if matched_indices is None:
            continue
        spans.append(
            (
                page_index.spans[matched_indices[0]].start,
                page_index.spans[matched_indices[-1]].end,
            )
        )
        if len(spans) >= max_results:
            break

    return spans


def _add_spans(
    results: list[TextSearchMatch],
    *,
    page_number: int,
    page_text: str,
    spans: Sequence[tuple[int, int]],
    context_chars: int,
) -> None:
    for start, end in spans:
        results.append(
            {
                "page": page_number,
                "start": start,
                "end": end,
                "snippet": _build_snippet(page_text, start, end, context_chars),
            }
        )


def search_text(
    text_content: str,
    query: str | Sequence[str],
    *,
    page: int | None = None,
    case_sensitive: bool = False,
    max_results: int = 20,
    context_chars: int = 80,
) -> list[TextSearchMatch]:
    """Search extracted page text and return page-aware snippets.

    ``query`` may be a single pattern or a list of patterns. With a single
    string the result is the plain page-aware match list. With a list, each
    pattern is searched in turn, every match is tagged with the ``pattern``
    that produced it, results are deduplicated by ``(page, start, end)``
    (first pattern wins), and ``max_results`` is a global cap across patterns.
    """
    if isinstance(query, str):
        return _search_single(
            text_content,
            query,
            page=page,
            case_sensitive=case_sensitive,
            max_results=max_results,
            context_chars=context_chars,
        )

    patterns = [p.strip() for p in query]
    if not patterns or not any(patterns):
        raise ValueError("query must not be empty")
    if max_results < 1:
        raise ValueError("max_results must be at least 1")

    results: list[TextSearchMatch] = []
    seen: set[tuple[int, int, int]] = set()
    for pattern in patterns:
        if not pattern:
            continue
        if len(results) >= max_results:
            break
        # Fetch up to the full cap per pattern (not just the remaining slots):
        # the cross-pattern dedup below may drop a pattern's leading matches as
        # duplicates, and a per-slot budget would discard its unique hits before
        # they are ever returned.
        for match in _search_single(
            text_content,
            pattern,
            page=page,
            case_sensitive=case_sensitive,
            max_results=max_results,
            context_chars=context_chars,
        ):
            key = (match["page"], match["start"], match["end"])
            if key in seen:
                continue
            seen.add(key)
            match["pattern"] = pattern
            results.append(match)
            if len(results) >= max_results:
                break
    return results


def _search_single(
    text_content: str,
    query: str,
    *,
    page: int | None = None,
    case_sensitive: bool = False,
    max_results: int = 20,
    context_chars: int = 80,
) -> list[TextSearchMatch]:
    """Search extracted page text for a single pattern."""
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    if max_results < 1:
        raise ValueError("max_results must be at least 1")
    if context_chars < 0:
        raise ValueError("context_chars must be non-negative")

    candidate_pages = tuple(
        page_text
        for page_text in _iter_page_text(text_content)
        if page is None or page_text.page == page
    )
    literal_pattern = _compile_literal_pattern(query, case_sensitive=case_sensitive)
    results: list[TextSearchMatch] = []
    for page_number, page_text in candidate_pages:
        remaining_results = max_results - len(results)
        spans = _find_literal_spans(
            page_text,
            literal_pattern,
            max_results=remaining_results,
        )
        _add_spans(
            results,
            page_number=page_number,
            page_text=page_text,
            spans=spans,
            context_chars=context_chars,
        )

        if len(results) >= max_results:
            break

    if results or not any(char.isspace() for char in query):
        return results

    collapsed_query = " ".join(query.split())
    if collapsed_query:
        collapsed_pattern = _compile_literal_pattern(
            collapsed_query,
            case_sensitive=case_sensitive,
        )
        for page_number, page_text in candidate_pages:
            remaining_results = max_results - len(results)
            spans = _find_collapsed_whitespace_spans(
                page_text,
                collapsed_pattern,
                max_results=remaining_results,
            )
            _add_spans(
                results,
                page_number=page_number,
                page_text=page_text,
                spans=spans,
                context_chars=context_chars,
            )

            if len(results) >= max_results:
                break

    if results:
        return results

    query_index = _build_token_index(query, case_sensitive=case_sensitive)
    for page_number, page_text in candidate_pages:
        remaining_results = max_results - len(results)
        spans = _find_token_sequence_spans(
            _build_token_index(page_text, case_sensitive=case_sensitive),
            query_index,
            max_results=remaining_results,
        )
        _add_spans(
            results,
            page_number=page_number,
            page_text=page_text,
            spans=spans,
            context_chars=context_chars,
        )

        if len(results) >= max_results:
            break

    return results
