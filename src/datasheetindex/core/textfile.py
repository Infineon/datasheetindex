"""PDF to page-matched text file generation."""

from __future__ import annotations

import re
from collections.abc import Sequence
from functools import cache, lru_cache
from typing import TYPE_CHECKING, Any, NamedTuple, TypedDict

if TYPE_CHECKING:
    import pymupdf


class TextSearchMatch(TypedDict):
    """A text match located within a page of the extracted text artifact."""

    page: int
    start: int
    end: int
    snippet: str


_PAGE_MARKER_RE = re.compile(r"--- PAGE (\d+) ---")
_TOKEN_RE = re.compile(r"\S+")
_DASH_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
    }
)
_TOKEN_EDGE_PUNCTUATION = ".,;:!?"


class _TokenSpan(NamedTuple):
    value: str
    start: int
    end: int


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


def _extract_page_text(page: pymupdf.Page) -> str:
    """Extract text from a page with column-aware reading order.

    Uses ``page.get_text("blocks")`` to detect two-column layouts and
    reorder blocks so the left column is read before the right column.
    Falls back to standard top-to-bottom, left-to-right ordering when
    no column structure is detected.
    """
    raw_blocks = page.get_text("blocks")
    text_blocks = [b for b in raw_blocks if b[_BLOCK_TYPE] == 0]

    if not text_blocks:
        return ""

    page_width = page.rect.width
    result = _detect_columns(text_blocks, page_width)

    if result is None:
        # No columns detected -- standard reading order
        text_blocks.sort(key=lambda b: (b[_BLOCK_Y0], b[_BLOCK_X0]))
        return "\n".join(b[_BLOCK_TEXT] for b in text_blocks)

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

    ordered = (
        sorted(above, key=lambda b: (b[_BLOCK_Y0], b[_BLOCK_X0]))
        + sorted(left_col, key=lambda b: b[_BLOCK_Y0])
        + sorted(right_col, key=lambda b: b[_BLOCK_Y0])
        + sorted(below, key=lambda b: (b[_BLOCK_Y0], b[_BLOCK_X0]))
    )
    return "\n".join(b[_BLOCK_TEXT] for b in ordered)


def generate_text(doc: pymupdf.Document) -> str:
    """Generate page-matched text with PAGE markers.

    Each page's text is preceded by a ``--- PAGE N ---`` marker where N is
    1-indexed (matching human PDF page numbers).
    """
    parts: list[str] = []
    for page_idx in range(len(doc)):
        page_num = page_idx + 1
        page = doc[page_idx]
        text = _extract_page_text(page)
        parts.append(f"--- PAGE {page_num} ---")
        parts.append(text)
    return "\n".join(parts)


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


@lru_cache(maxsize=256)
def _translate_search_text(text: str) -> str:
    return text.translate(_DASH_TRANSLATION)


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


def _normalize_token(token: str, *, case_sensitive: bool) -> str:
    normalized = _translate_search_text(token).strip(_TOKEN_EDGE_PUNCTUATION)
    return normalized if case_sensitive else normalized.casefold()


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
    query: str,
    *,
    page: int | None = None,
    case_sensitive: bool = False,
    max_results: int = 20,
    context_chars: int = 80,
) -> list[TextSearchMatch]:
    """Search extracted page text and return page-aware snippets."""
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
