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

import base64
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pymupdf

from datasheetindex.llm.client import is_permanent_llm_failure
from datasheetindex.tools.vision import inspect_page

if TYPE_CHECKING:
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

#: A region rendered as exactly one distinct colour has nothing in it to
#: describe -- not a tuned heuristic, a tautology. Measured over 38 real
#: raster regions across three plot-heavy TI datasheets (tlv9061, opa2134,
#: ads1115), rendered at dpi=150 -- the resolution inspect_page's own
#: ``detail="high"`` default uses: 6 of 38 measured exactly 1.0, all
#: confirmed blank XObject fragments on ti-tlv9061 p46 (the page behind the
#: hallucination this guard fixes: a blank region captioned as "a schematic
#: diagram ... optocoupler component"). The nearest real-content region
#: measured 0.9877, and one near-miss blank fragment measured 0.999743 and is
#: deliberately left uncaught here. Do NOT loosen this to 0.99 or similar --
#: a characteristic-curve plot is mostly white with thin lines, and 0.9877 is
#: closer to 1.0 than to 0.99, so a loosened threshold would start skipping
#: exactly the plot figures this feature exists to caption.
_BLANK_REGION_TOPUSAGE = 1.0


def _is_blank_region(image_base64: str) -> bool:
    """True when the rendered region carries a single distinct colour.

    Runs *after* ``inspect_page`` has already rendered the region, so a blank
    region still consumes one of the caller's ``max_figure_captions`` slots --
    the cap selects candidates before this check ever runs. That is accepted
    as the cost of a minimal guard rather than reworking candidate selection.
    """
    pix = pymupdf.Pixmap(base64.b64decode(image_base64))
    fraction, _colour = pix.color_topusage()
    return fraction >= _BLANK_REGION_TOPUSAGE


CAPTION_SYSTEM_PROMPT = (
    "You are labelling a figure from an electronics datasheet so an agent "
    "can decide whether it holds what it is looking for. Name the kind of "
    "content (table, schematic, plot, photo, block diagram, pinout), then "
    "IMMEDIATELY name the figure's most identifying labels: for a table its "
    "row labels first, then its column headings; for a plot its axes and "
    "plotted quantity; for a diagram its labelled blocks. Your text may be "
    "truncated, so identifying labels must come before any description of "
    "structure. Name only labels you can actually read; if they are "
    "illegible, say so rather than guessing. Do NOT transcribe cell values, "
    "measurements or numbers. Under 60 words. Do not begin with 'This is'."
)


def validate_max_figure_captions(max_figure_captions: object) -> None:
    """Raise ``ValueError`` unless the cap is an integer ``>= 0``.

    One definition, two callers, because the second one has a side effect to
    order against: ``DatasheetIndex.build`` validates at its own entry, but
    ``DatasheetTools.build_datasheet`` must reject a bad cap *before* it removes
    a valid sidecar, or a rejected call destroys a usable cache on its way to
    raising.

    ``bool`` is rejected explicitly (it is an ``int`` subclass, so ``True``
    would silently become a cap of 1), and a non-int is rejected here rather
    than reaching ``candidates[:2.5]`` and raising ``TypeError`` deep inside the
    captioning pass.
    """
    if not isinstance(max_figure_captions, int) or isinstance(
        max_figure_captions, bool
    ):
        raise ValueError("max_figure_captions must be an integer >= 0")
    if max_figure_captions < 0:
        raise ValueError("max_figure_captions must be an integer >= 0")


def _candidate_order(entry: dict[str, object]) -> tuple[float, int]:
    """Largest visible area first, page as the tie-break.

    The cap then retains the most substantive regions, and equal-area regions
    keep a stable, reproducible order. The casts are safe by construction:
    ``core.figures.raster_regions`` writes both fields, and the dict is only
    ``object``-valued because the figure entry is heterogeneous.
    """
    return (-cast("float", entry["page_area_pct"]), cast("int", entry["page"]))


def _raster_candidates(figures: list[dict[str, object]]) -> list[dict[str, object]]:
    """Every raster region, largest visible area first.

    Built on a copy: the caller's array stays in document order.
    """
    candidates = [entry for entry in figures if entry["kind"] == "raster"]
    candidates.sort(key=_candidate_order)
    return candidates


def _image_identity(entry: dict[str, object]) -> int | None:
    """The XObject this placement draws, or ``None`` when unknown.

    ``None`` is not a shared identity. A missing or zero ``xref`` -- an
    artifact built before the field existed, or a hand-built entry -- means the
    document never told us what this region is, and grouping unknowns together
    would hand one picture's caption to every unidentified figure in the
    document. ``bool`` is excluded because it is an ``int`` subclass and
    ``True`` would otherwise read as xref 1.
    """
    xref = entry.get("xref")
    if isinstance(xref, bool) or not isinstance(xref, int) or xref <= 0:
        return None
    return xref


def _image_groups(figures: list[dict[str, object]]) -> list[list[dict[str, object]]]:
    """Raster placements grouped by picture, largest picture first.

    A PDF draws a repeated logo as one image XObject placed once per page, so
    the placements are N entries showing the same picture. Describing that
    picture once and sharing the answer is exact, not an approximation: an
    XObject's content cannot vary between placements, only its scale.

    Each group is ordered with its largest placement first -- that is the
    representative the pass renders, giving the vision model the most legible
    copy -- and the groups themselves follow that representative's candidate
    order, so the cap keeps the most substantive *pictures*. Both orders come
    from a single pass over the already-sorted candidate list, which is what
    keeps the result byte-stable across builds; artifact reuse fingerprints
    these captions, so an order that varied between runs would defeat it.
    """
    groups: list[list[dict[str, object]]] = []
    by_xref: dict[int, list[dict[str, object]]] = {}

    for entry in _raster_candidates(figures):
        xref = _image_identity(entry)
        if xref is None:
            groups.append([entry])
            continue
        group = by_xref.get(xref)
        if group is None:
            group = []
            by_xref[xref] = group
            groups.append(group)
        group.append(entry)

    return groups


def eligible_caption_count(
    figures: list[dict[str, object]],
    max_figure_captions: int = DEFAULT_MAX_FIGURE_CAPTIONS,
) -> int:
    """How many VLM calls ``caption_figures_in_place`` would attempt.

    Exists so a caller deciding whether constructing a vision client can pay
    for itself asks the captioning pass what a candidate is instead of keeping
    a second copy of that definition -- a copy that could drift and silently
    stop a client from ever being built. Counts distinct pictures, matching
    what the pass actually dispatches: a document whose only regions are four
    placements of one logo is one call's worth of work, not four.
    """
    return min(len(_image_groups(figures)), max(0, max_figure_captions))


@dataclass(frozen=True)
class CaptionOutcome:
    """What the captioning pass achieved.

    ``failed`` and ``pending`` mean different things and must not be merged.
    ``failed`` is transient -- a call raised or returned nothing -- and marks
    the artifact incomplete so a gateway blip is not cached forever.
    ``pending`` is stable: no vision client was available, which on a default
    ``uv sync`` (no ``[llm]`` extra) is simply the environment.

    ``blank`` is a third, distinct kind of stable outcome: a region skipped
    because ``_is_blank_region`` found nothing in it. It must join neither of
    the above. Not ``failed`` -- a region blank today renders blank on every
    future build of the same document, so marking it would rebuild that
    document forever for a fact that never changes, exactly the cache-
    poisoning bug class ``failed`` exists to avoid. Not ``pending`` either --
    a vision client *was* available; there was simply nothing captionable to
    hand it.

    **The units differ between fields, deliberately.** ``captioned``,
    ``pending`` and ``blank`` count *pictures* -- one per distinct image
    XObject, so each is one VLM call made, skipped or owed. ``shared`` and
    ``excluded_above_max`` count *entries*: ``shared`` is how many extra
    placements received a caption their picture had already earned, and
    ``excluded_above_max`` is how many entries the cap left with no caption at
    all. The second pair is in entries because that is what their consumers
    see -- ``excluded_above_max`` is published as
    ``figure_captions_excluded.above_max`` in the artifact, where the question
    being answered is "how many figure entries here lack a caption".
    """

    captioned: int
    pending: int
    excluded_above_max: int
    failed: bool
    blank: int = 0
    shared: int = 0
    #: **Every** attempted caption failed for a reason that will not change on
    #: retry -- a rejected certificate, or credentials the gateway refuses
    #: (401/403). Totality is the whole meaning: one rejected call among
    #: successes is a blip, and a document that carries real captions must
    #: never be published as blocked.
    #:
    #: A *narrowing* of ``failed``, not a replacement: ``failed`` is still set,
    #: so reuse behaves exactly as before and a misconfigured build is never
    #: cached as complete. This exists so the failure can be REPORTED
    #: differently -- once, naming the cause -- because the expensive part of
    #: this state is how long it lasts, not what one build of it costs.
    #: Deliberately conservative: anything unrecognised stays plain ``failed``,
    #: since telling an operator to fix a healthy gateway is the worse error.
    blocked: bool = False


def caption_figures_in_place(
    doc: pymupdf.Document,
    figures: list[dict[str, object]],
    *,
    vision_client: VisionLlmCallable | None,
    max_figure_captions: int = DEFAULT_MAX_FIGURE_CAPTIONS,
) -> CaptionOutcome:
    """Caption eligible raster regions, mutating ``figures`` in place.

    The unit of work is a *picture*, not a placement: repeated placements of
    one image XObject are described once and every placement receives the
    answer (see ``_image_groups``). The cap therefore bounds VLM calls, which
    is what it has always claimed to bound.
    """
    groups = _image_groups(figures)

    eligible = groups[: max(0, max_figure_captions)]
    excluded_above_max = sum(len(group) for group in groups[len(eligible) :])

    if not eligible:
        return CaptionOutcome(0, 0, excluded_above_max, False)

    if vision_client is None:
        return CaptionOutcome(0, len(eligible), excluded_above_max, False)

    # Render serially: PyMuPDF is not thread-safe for concurrent page work.
    # One render per group, of its largest placement -- the most legible copy
    # of the picture, and the only one the model ever sees.
    rendered: list[tuple[list[dict[str, object]], str]] = []
    failed = False
    blank = 0
    for group in eligible:
        entry = group[0]
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
        image_base64 = cast("str", blocks[0]["data"])
        if _is_blank_region(image_base64):
            blank += 1
            logger.info(
                "Skipping blank figure region on page %s: rendered region is a "
                "single colour, nothing to caption",
                entry["page"],
            )
            continue
        rendered.append((group, image_base64))

    # Collected across threads: `list.append` is atomic under the GIL, and only
    # the first entry is ever read. A set would work equally well; a list keeps
    # the first failure rather than an arbitrary one.
    permanent: list[BaseException] = []

    # Dispatch concurrently: network I/O, safe to overlap.
    def describe(payload: tuple[list[dict[str, object]], str]) -> str | None:
        entry = payload[0][0]
        image_base64 = payload[1]
        try:
            reply = vision_client.describe_image(CAPTION_SYSTEM_PROMPT, image_base64)
        except Exception as exc:
            if is_permanent_llm_failure(exc):
                permanent.append(exc)
            # Deliberately NOT carved out for LlmTlsVerificationError, unlike
            # the ToC fallback. Captioning runs at step 6b and the artifacts are
            # written at step 8, so raising here would abort the build and write
            # *nothing* -- for a document whose index is otherwise complete and
            # whose ToC may be perfectly good. An unusable artifact is a worse
            # outcome than uncaptioned figures, and the ToC fallback's argument
            # (an empty ToC is indistinguishable from a document with no
            # outline) does not transfer to a case where the outline is fine.
            #
            # Legibility still improves, without the raise: the warning below
            # now carries the named error's full remedy instead of openai's
            # "Connection error.". The re-captioning loop this would have
            # closed is pre-existing behaviour for any persistent caption
            # failure, and closing it belongs with `figure_captions_pending`'s
            # non-transient treatment in `reuse_blocker`, not here.
            logger.warning(
                "Figure caption failed on page %s", entry["page"], exc_info=True
            )
            return None
        caption = (reply or "").strip()
        if not caption:
            # A raise was always logged; an empty *reply* was not, and the two
            # are indistinguishable downstream -- both become ``failed``. That
            # silence is how a transport returning empty captions for half of
            # them hid until someone measured 16 regions five times. The client
            # logs *why* (model, finish_reason); this logs *where*.
            logger.warning("Figure caption came back empty on page %s", entry["page"])
        return caption or None

    with ThreadPoolExecutor(max_workers=_DISPATCH_WORKERS) as pool:
        # map preserves input order, so results are applied in candidate order
        # rather than completion order -- the figures array must hash the same
        # on every build or artifact reuse is defeated.
        replies = list(pool.map(describe, rendered))

    captioned = 0
    shared = 0
    for (group, _), reply in zip(rendered, replies, strict=True):
        if reply is None:
            failed = True
            continue
        # Every placement of the picture, not just the one rendered. Leaving
        # the others null would make pages 2-N of a repeated figure look like
        # pages with an uncaptionable region on them.
        for placement in group:
            placement["caption"] = reply
            placement["caption_source"] = "llm"
        captioned += 1
        shared += len(group) - 1

    # Totality, not presence. One rejected call among successes is a blip -- a
    # 401 during a key rotation, a 403 on one oversized image -- and claiming
    # the gateway is misconfigured there would send an operator to fix
    # something that just served the other captions, as well as publishing
    # `figure_captions_blocked` on a document that carries real ones. The
    # `rendered` guard matters on its own: with no figures both lists are
    # empty and `len(permanent) == len(rendered)` is vacuously true.
    blocked = bool(rendered) and len(permanent) == len(rendered)
    if blocked:
        # ERROR, once, naming the cause -- alongside the per-figure warnings
        # above, which stay: they are the only signal on a PARTIAL permanent
        # failure, which this branch deliberately does not fire for.
        # Nothing else changes: `failed` still marks the artifact incomplete, so
        # reuse behaves exactly as before. What this fixes is the *duration* of
        # the breakage. While it lasts, `figure_caption_failed` blocks artifact
        # reuse, so every build_datasheet call re-scans the whole PDF (86.5s
        # measured on the 134-page PSoC 6) to fail the same way again. That cost
        # is only worth paying attention to because the operator could not
        # previously tell what to fix: the one actionable line was buried in
        # each per-figure traceback.
        logger.error(
            "Figure captioning is misconfigured, not merely failing: all %d "
            "attempted captions were rejected for a reason that will not change "
            "on retry. Until it is fixed, this document is rebuilt from scratch "
            "on every request. Cause: %s",
            len(rendered),
            permanent[0],
        )
    return CaptionOutcome(
        captioned, 0, excluded_above_max, failed, blank, shared, blocked
    )
