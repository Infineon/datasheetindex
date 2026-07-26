"""Name raster figure regions with a VLM.

Every raster region above ``min_area_pct`` is a candidate. Two triage rules
were designed and measured and both failed: comparing per-page caption and
raster counts is the caption-to-region association section 4 of the spec
refuses to make, reached by arithmetic; thresholding page text does not
discriminate, because a real datasheet's raster pages are *more* text-starved
than the pathological document's. So the cap, not triage, is the cost control
-- and unlike triage a cap cannot silently skip a blind region.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from datasheetindex.tools.vision import inspect_page

if TYPE_CHECKING:
    import pymupdf

    from datasheetindex.llm.client import VisionLlmCallable

logger = logging.getLogger(__name__)

#: Per-document ceiling on VLM calls. Corpus median is 4 candidates per
#: document and the max is 29, so this binds only on outliers. It exists for
#: the shape neither fixture has: a scanned document whose every page is one
#: full-page image over an empty text layer.
DEFAULT_MAX_FIGURE_CAPTIONS = 20

#: Four rather than one-per-candidate: an unbounded pool at the default cap
#: opens twenty simultaneous gateway connections, which is how a shared
#: deployment starts rate-limiting.
_DISPATCH_WORKERS = 4

CAPTION_SYSTEM_PROMPT = (
    "You are labelling a figure extracted from an electronics datasheet. "
    "In one sentence, name the kind of content (table, schematic, plot, "
    "photo, block diagram, pinout) and its subject. Do NOT transcribe any "
    "values, cell contents, or numbers. This is a navigation label, not data."
)


def _candidate_order(entry: dict[str, object]) -> tuple[float, int]:
    """Largest visible area first, page as the tie-break.

    The cap then retains the most substantive regions, and equal-area regions
    keep a stable, reproducible order. The casts are safe by construction:
    ``core.figures.raster_regions`` writes both fields, and the dict is only
    ``object``-valued because the figure entry is heterogeneous.
    """
    return (-cast("float", entry["page_area_pct"]), cast("int", entry["page"]))


@dataclass(frozen=True)
class CaptionOutcome:
    """What the captioning pass achieved.

    ``failed`` and ``pending`` mean different things and must not be merged.
    ``failed`` is transient -- a call raised or returned nothing -- and marks
    the artifact incomplete so a gateway blip is not cached forever.
    ``pending`` is stable: no vision client was available, which on a default
    ``uv sync`` (no ``[llm]`` extra) is simply the environment.
    """

    captioned: int
    pending: int
    excluded_above_max: int
    failed: bool


def caption_figures_in_place(
    doc: pymupdf.Document,
    figures: list[dict[str, object]],
    *,
    vision_client: VisionLlmCallable | None,
    max_figure_captions: int = DEFAULT_MAX_FIGURE_CAPTIONS,
) -> CaptionOutcome:
    """Caption eligible raster regions, mutating ``figures`` in place."""
    candidates = [entry for entry in figures if entry["kind"] == "raster"]
    # Sorted on a copy: the caller's array stays in document order.
    candidates.sort(key=_candidate_order)

    eligible = candidates[: max(0, max_figure_captions)]
    excluded_above_max = len(candidates) - len(eligible)

    if not eligible:
        return CaptionOutcome(0, 0, excluded_above_max, False)

    if vision_client is None:
        return CaptionOutcome(0, len(eligible), excluded_above_max, False)

    # Render serially: PyMuPDF is not thread-safe for concurrent page work.
    rendered: list[tuple[dict[str, object], str]] = []
    failed = False
    for entry in eligible:
        try:
            blocks = inspect_page(
                doc,
                page=cast("int", entry["page"]),
                # Already normalized and clipped to the page upstream.
                # inspect_page rejects anything outside 0.0-1.0, so this must
                # arrive unmodified.
                region=cast("dict[str, float]", entry["region"]),
            )
        except Exception:
            logger.warning(
                "Could not render figure region on page %s for captioning",
                entry["page"],
                exc_info=True,
            )
            failed = True
            continue
        rendered.append((entry, cast("str", blocks[0]["data"])))

    # Dispatch concurrently: network I/O, safe to overlap.
    def describe(payload: tuple[dict[str, object], str]) -> str | None:
        entry, image_base64 = payload
        try:
            reply = vision_client.describe_image(CAPTION_SYSTEM_PROMPT, image_base64)
        except Exception:
            logger.warning(
                "Figure caption failed on page %s", entry["page"], exc_info=True
            )
            return None
        return (reply or "").strip() or None

    with ThreadPoolExecutor(max_workers=_DISPATCH_WORKERS) as pool:
        # map preserves input order, so results are applied in candidate order
        # rather than completion order -- the figures array must hash the same
        # on every build or artifact reuse is defeated.
        replies = list(pool.map(describe, rendered))

    captioned = 0
    for (entry, _), reply in zip(rendered, replies, strict=True):
        if reply is None:
            failed = True
            continue
        entry["caption"] = reply
        entry["caption_source"] = "llm"
        captioned += 1

    return CaptionOutcome(captioned, 0, excluded_above_max, failed)
