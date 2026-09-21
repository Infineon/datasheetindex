"""locate_text -- map a query string to its bounding box(es) on a PDF page.

The missing edge between ``search_text`` (find text -> char offset) and
``inspect_page`` (render a region): given a string and a page, return where it
sits, as normalized percentages clamped to the page (for the ``inspect_page``
round-trip) and raw, unclamped PDF points (for PDF-native annotation).
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
    # {"top","bottom","left","right"}, clamped to 0.0-1.0, of the *displayed*
    # page (page.rect, what inspect_page crops) -- rotation applied.
    pct: dict[str, float]
    # {"x0","y0","x1","y1"}, raw PDF points, unclamped, in the page's
    # *unrotated* coordinate space (get_text/search_for), top-left origin.
    points: dict[str, float]


class TextLocation(TypedDict):
    page: int  # 1-indexed
    match_method: str  # "search_for" | "tokens"
    page_width: float  # PDF points, displayed page (rotation applied)
    page_height: float  # PDF points, displayed page (rotation applied)
    region: _Box  # union of boxes; the inspect_page round-trip input
    boxes: list[_Box]  # >= 1; a multi-line match yields one box per line
    pattern: NotRequired[str]  # which query produced this hit (list queries only)


def _clamp01(value: float) -> float:
    """Clamp a normalized coordinate into ``[0.0, 1.0]``."""
    return min(1.0, max(0.0, value))


def _box_from_rect(
    rect: _Rect, page_rect: pymupdf.Rect, rotation_matrix: pymupdf.Matrix
) -> _Box:
    """Normalize a match rectangle against the page.

    ``pct`` is clamped to the page; ``points`` deliberately is not.

    A glyph can sit partly outside the page rect -- a descender crossing the
    bottom edge, or a CropBox whose origin falls inside the match -- and the
    normalized value is then just outside ``[0, 1]``. That matters because
    ``pct`` is documented as the ``inspect_page(region=...)`` input and
    ``inspect_page`` *raises* on an out-of-range region, so an unclamped box
    makes this function emit a region its own documented consumer rejects.
    Clamping keeps a genuine match usable instead of unrenderable.

    The two are also in different spaces on a rotated page, deliberately.
    ``search_for`` reports the unrotated page, and ``points`` keeps that: it is
    what a PDF-native consumer (an annotation, or pdf.js's page transform, which
    applies the rotation itself) expects. ``pct`` is mapped through
    ``rotation_matrix`` first, because ``page.rect`` and ``inspect_page``
    describe the displayed page; normalizing unrotated points against it put a
    90-degree page's box on the wrong part of the page, or clamped it flat.

    ``points`` stays raw because it means something different: PDF-native
    coordinates for annotation and highlighting, where a glyph that really does
    cross the page edge should be described where it actually sits. So the
    ``pct * page_width == points - page_rect.x0`` identity holds for every box
    inside an unrotated page and is intentionally broken for one that overflows.
    """
    import pymupdf

    displayed = pymupdf.Rect(rect) * rotation_matrix
    x0, y0, x1, y1 = displayed.x0, displayed.y0, displayed.x1, displayed.y1
    width = page_rect.width
    height = page_rect.height
    return {
        "pct": {
            "left": _clamp01((x0 - page_rect.x0) / width),
            "right": _clamp01((x1 - page_rect.x0) / width),
            "top": _clamp01((y0 - page_rect.y0) / height),
            "bottom": _clamp01((y1 - page_rect.y0) / height),
        },
        "points": {"x0": rect[0], "y0": rect[1], "x1": rect[2], "y1": rect[3]},
    }


def _union_region(
    boxes: list[_Box], page_rect: pymupdf.Rect, rotation_matrix: pymupdf.Matrix
) -> _Box:
    return _box_from_rect(
        (
            min(b["points"]["x0"] for b in boxes),
            min(b["points"]["y0"] for b in boxes),
            max(b["points"]["x1"] for b in boxes),
            max(b["points"]["y1"] for b in boxes),
        ),
        page_rect,
        rotation_matrix,
    )


def _search_for_occurrences(page: pymupdf.Page, query: str) -> list[list[_Rect]]:
    """Fast path: verbatim ``search_for`` hits, one list of rects per occurrence.

    ``search_for`` returns one rect per line fragment of a hit, not one per hit:
    a phrase that wraps, a table row (one rect per cell) or a sub/superscript
    (``R_DS(on)``, a trademark sign) comes back as several consecutive rects.
    Reporting each as its own occurrence turned one match into a tie. PyMuPDF
    does not say which rects belong together, so they are grouped by counting:
    a hit covers exactly the query's non-whitespace characters, because MuPDF's
    search matches case-insensitively and lets any whitespace run match any
    other. When the count does not line up, fall back to one rect per
    occurrence -- the pre-grouping behaviour, never worse than before.
    """
    rects = [(r.x0, r.y0, r.x1, r.y1) for r in page.search_for(query)]
    per_rect = [[rect] for rect in rects]
    if len(rects) < 2:
        return per_rect
    target = sum(1 for ch in query if not ch.isspace())
    centers = _glyph_centers(page)
    occurrences: list[list[_Rect]] = []
    current: list[_Rect] = []
    count = 0
    for rect in rects:
        x0, y0, x1, y1 = rect
        current.append(rect)
        count += sum(1 for cx, cy in centers if x0 <= cx < x1 and y0 <= cy < y1)
        if count == target:
            occurrences.append(current)
            current, count = [], 0
        elif count > target:
            return per_rect
    return per_rect if current else occurrences


def _glyph_centers(page: pymupdf.Page) -> list[tuple[float, float]]:
    """Centre point of every non-whitespace glyph on the page (unrotated space)."""
    centers: list[tuple[float, float]] = []
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", ()):
            for span in line["spans"]:
                for char in span["chars"]:
                    if char["c"].isspace():
                        continue
                    x0, y0, x1, y1 = char["bbox"]
                    centers.append(((x0 + x1) / 2, (y0 + y1) / 2))
    return centers


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
                boxes = [
                    _box_from_rect(rect, page_rect, page_obj.rotation_matrix)
                    for rect in occurrence
                ]
                if not boxes:
                    continue
                key = _dedup_key(page_number, boxes)
                if key in seen:
                    continue
                seen.add(key)
                region = (
                    boxes[0]
                    if len(boxes) == 1
                    else _union_region(boxes, page_rect, page_obj.rotation_matrix)
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
