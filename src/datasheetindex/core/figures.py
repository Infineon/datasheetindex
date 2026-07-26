"""Per-page figure entries: raster regions and text-layer captions.

Raster enumeration is *exact*: ``get_image_info`` reads the PDF's image
XObjects, so nothing is inferred and there is no false-positive rate to
calibrate. Vector figures are deliberately out of scope -- they leak their text
into the extraction, so they are not invisible the way a raster region is.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pymupdf

#: Placements smaller than this share of the page are excluded. Measured across
#: a 14-document corpus: 73 of 168 placements fall below it, 61 below 0.5%, and
#: 4 of 14 documents repeat one image XObject across pages (a vendor logo). Set
#: low on purpose -- excluding real content is the expensive error.
DEFAULT_MIN_AREA_PCT = 1.0

#: Figure numbers are plain (``12``) or section-relative (``10-1``), with either
#: a hyphen or an en-dash. Measured across 14 documents: the plain-only pattern
#: found captions in 2 of them; this one finds them in 11.
_FIGURE_NUMBER = r"(\d+(?:[-–]\d+)?)"

#: A line that is *only* a figure label. Its title is the next non-empty line.
_SPLIT_FORM = re.compile(rf"^(?:Figure|Fig\.)\s+{_FIGURE_NUMBER}\s*[.:]?\s*$")

#: Label, a **mandatory** ``.`` or ``:``, then the title on the same line. The
#: separator is the whole prose filter: in "Figure 2 shows the major
#: subsystems" it is a space, so that line matches neither form and emits
#: nothing.
_SAME_LINE_FORM = re.compile(rf"^(?:Figure|Fig\.)\s+{_FIGURE_NUMBER}\s*[.:]\s+(\S.*)$")


def raster_regions(
    page: pymupdf.Page, *, min_area_pct: float = DEFAULT_MIN_AREA_PCT
) -> tuple[list[dict[str, object]], int]:
    """Return ``(entries, excluded_below_min_area)`` for one page.

    The bbox ``get_image_info`` reports is the *placement* rectangle and can
    extend past the page. It is intersected with ``page.rect`` first, and
    ``region``, ``bbox`` and ``page_area_pct`` all describe the **visible**
    rectangle -- an unclipped normalization lands outside ``0..1``, which
    ``inspect_page`` rejects outright. A placement with an empty intersection is
    dropped rather than emitted with a degenerate region.
    """
    import pymupdf

    rect = page.rect
    page_area = rect.width * rect.height
    if page_area <= 0:
        return [], 0

    page_number = page.number
    assert page_number is not None, "page must be bound to an open document"
    page_text_chars = len(page.get_text())
    entries: list[dict[str, object]] = []
    excluded = 0

    for info in page.get_image_info():
        visible = pymupdf.Rect(info["bbox"]) & rect
        if visible.is_empty:
            continue
        area_pct = 100.0 * (visible.width * visible.height) / page_area
        if area_pct < min_area_pct:
            excluded += 1
            continue
        entries.append(
            {
                "page": page_number + 1,
                "kind": "raster",
                "region": {
                    "left": (visible.x0 - rect.x0) / rect.width,
                    "right": (visible.x1 - rect.x0) / rect.width,
                    "top": (visible.y0 - rect.y0) / rect.height,
                    "bottom": (visible.y1 - rect.y0) / rect.height,
                },
                "bbox": [visible.x0, visible.y0, visible.x1, visible.y1],
                "pixels": [info.get("width", 0), info.get("height", 0)],
                "page_area_pct": round(area_pct, 2),
                "page_text_chars": page_text_chars,
                "caption": None,
                "caption_source": None,
            }
        )
    return entries, excluded


def caption_entries(page_number: int, page_text: str) -> list[dict[str, object]]:
    """Return caption entries found in one page's column-aware text.

    ``page_text`` must come from ``_extract_page_text`` -- the same
    column-aware extraction the text file carries. On ``page.get_text()`` a
    two-column page can interleave a split-form number with an unrelated
    title, silently producing a wrong caption.

    A ``Figure N`` mention matching neither accepted form emits nothing. An
    entry claiming page 12 merely *mentions* Figure 3 is noise the consumer
    would have to filter, and every entry in this array is meant to be worth
    acting on.
    """
    lines = page_text.splitlines()
    entries: list[dict[str, object]] = []

    for index, raw in enumerate(lines):
        line = raw.strip()

        same_line = _SAME_LINE_FORM.match(line)
        if same_line is not None:
            entries.append(_caption_entry(page_number, same_line.group(1), line))
            continue

        split = _SPLIT_FORM.match(line)
        if split is None:
            continue
        title = next((nxt.strip() for nxt in lines[index + 1 :] if nxt.strip()), None)
        if title is None:
            # A bare label as the last line of a page: no title to pair it
            # with, and reaching onto the next page would invent one.
            continue
        entries.append(_caption_entry(page_number, split.group(1), f"{line} {title}"))

    return entries


def _caption_entry(
    page_number: int, figure_number: str, caption: str
) -> dict[str, object]:
    return {
        "page": page_number,
        "kind": "caption",
        "region": None,
        "bbox": None,
        # A string always. Section-relative numbers are not integers, and a
        # union type costs every consumer a branch for no benefit -- this is an
        # identifier to display and match on, never an arithmetic value.
        "figure_number": figure_number,
        "caption": caption,
        "caption_source": "text",
    }
