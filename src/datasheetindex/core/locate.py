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

from datasheetindex.core._textmatch import (
    _TOKEN_RE,
    _match_query_tokens,
    _normalize_token,
    _TokenSpan,
)

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
    return [(b[0], b[1], b[2], b[3]) for b in (groups[key] for key in order)]


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
