"""inspect_page - page rendering as image for visual inspection."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import pymupdf

_VALID_REGION_KEYS = {"top", "bottom", "left", "right"}

#: Detail tiers map to render dpi. Sized so the resulting page-image's
#: vision-token cost (Anthropic ``(W*H)/750``; OpenAI similar) lands at
#: three useful operating points for agentic loops:
#:
#:   * "low" ~ 75 dpi  → ~650 tokens for a US-letter page; layout overview
#:     ("which quadrant has the table"), readable for large headings only
#:   * "medium" ~ 100 dpi → ~1150 tokens; readable table cells, body text,
#:     most diagrams. The recommended default for agentic loops.
#:   * "high" ~ 150 dpi → ~2580 tokens; required for sub-7pt footnotes,
#:     subscripts, dense schematic captions. Today's library default.
#:
#: The agent can pick per call; smaller-context models save proportional
#: input-token cost. The explicit ``dpi`` argument still wins if supplied.
_DETAIL_TO_DPI: dict[str, int] = {
    "low": 75,
    "medium": 100,
    "high": 150,
}

Detail = Literal["low", "medium", "high"]


def inspect_page(
    doc: pymupdf.Document,
    page: int,
    region: dict[str, float] | None = None,
    dpi: int | None = None,
    detail: Detail = "high",
) -> list[dict]:
    """Render a PDF page as a PNG image for visual inspection.

    Args:
        doc: An open PyMuPDF document.
        page: 1-indexed page number.
        region: Optional percentage-based crop region with keys
            ``top``, ``bottom``, ``left``, ``right`` (each 0.0-1.0).
        dpi: Explicit render resolution. Overrides ``detail`` when set.
            Default ``None`` -> falls back to ``detail``.
        detail: Vision-token-cost tier. "low" (75 dpi, ~650 tokens),
            "medium" (100 dpi, ~1150 tokens), or "high" (150 dpi,
            ~2580 tokens). Defaults to "high" for backward compatibility
            with pre-detail-arg callers; agent-surface wrappers typically
            override to "medium" to halve vision cost on long loops.

    Returns:
        A list with a single content block:
        ``[{"type": "image", "data": <base64_png>, "mime_type": "image/png"}]``

    Raises:
        ValueError: If page number is out of range or detail is unknown.
    """
    # Validate ``detail`` even when ``dpi`` is supplied. Otherwise a typo
    # in ``detail`` is silently ignored as long as the caller also passes
    # an explicit ``dpi``, then erupts the moment the override is removed.
    if detail not in _DETAIL_TO_DPI:
        raise ValueError(
            f"detail must be one of {sorted(_DETAIL_TO_DPI)}; got {detail!r}"
        )
    if dpi is None:
        dpi = _DETAIL_TO_DPI[detail]
    total_pages = len(doc)
    if page < 1 or page > total_pages:
        raise ValueError(
            f"Page {page} out of range. Document has {total_pages} pages (1-indexed)."
        )

    # Convert 1-indexed to 0-indexed
    page_obj = doc[page - 1]

    if region is not None:
        import pymupdf as _pymupdf

        unknown = set(region) - _VALID_REGION_KEYS
        if unknown:
            unknown_text = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown region keys: {unknown_text}")

        rect = page_obj.rect
        top = region.get("top", 0.0)
        bottom = region.get("bottom", 1.0)
        left = region.get("left", 0.0)
        right = region.get("right", 1.0)

        if not (0.0 <= top <= 1.0 and 0.0 <= bottom <= 1.0):
            raise ValueError("Region 'top' and 'bottom' must be between 0.0 and 1.0")
        if not (0.0 <= left <= 1.0 and 0.0 <= right <= 1.0):
            raise ValueError("Region 'left' and 'right' must be between 0.0 and 1.0")
        if top >= bottom:
            raise ValueError("Region requires top < bottom")
        if left >= right:
            raise ValueError("Region requires left < right")

        clip = _pymupdf.Rect(
            rect.x0 + left * rect.width,
            rect.y0 + top * rect.height,
            rect.x0 + right * rect.width,
            rect.y0 + bottom * rect.height,
        )
        pix = page_obj.get_pixmap(dpi=dpi, clip=clip)
    else:
        pix = page_obj.get_pixmap(dpi=dpi)

    png_bytes = pix.tobytes("png")
    b64_data = base64.b64encode(png_bytes).decode("ascii")

    return [{"type": "image", "data": b64_data, "mime_type": "image/png"}]
