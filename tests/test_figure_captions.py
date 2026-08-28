"""Tests for the VLM figure-captioning pass."""

from __future__ import annotations

import threading
import time

import pymupdf
import pytest

from datasheetindex.llm.figure_captions import (
    CAPTION_SYSTEM_PROMPT,
    DEFAULT_MAX_FIGURE_CAPTIONS,
    CaptionOutcome,
    caption_figures_in_place,
)


class RecordingVision:
    """A vision client that records calls and returns a canned line."""

    def __init__(self, reply="a table of device attributes", fail_on=()):
        self.calls = []
        self.systems = []
        self._reply = reply
        self._fail_on = set(fail_on)
        # Dispatch is concurrent, so the call index a call fails on has to be
        # decided under a lock; otherwise two threads can both read the list
        # length after both appends and neither fails.
        self._lock = threading.Lock()

    def describe_image(self, system, image_base64, *, media_type="image/png"):
        with self._lock:
            self.calls.append(image_base64)
            self.systems.append(system)
            index = len(self.calls)
        if index in self._fail_on:
            raise RuntimeError("gateway exploded")
        return self._reply


def _content_pixmap(color=(10, 20, 30)):
    """A 20x20 pixmap that renders as non-uniform, real content.

    A flat single colour would trip the blank-region guard
    (``color_topusage`` == 1.0) and be skipped before ever reaching the vision
    client -- these fixtures are exercising dispatch behaviour, not blank
    detection, so a thin accent stripe is added to keep every existing
    fixture below the guard's threshold, the same way a real plot's thin
    lines keep it below 1.0.
    """
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20))
    pix.set_rect(pix.irect, color)
    pix.set_rect(pymupdf.IRect(0, 0, 20, 1), (200, 200, 200))
    return pix


def _distinct_pixmap(index):
    """A content pixmap no other index produces.

    Load-bearing, not cosmetic. PyMuPDF folds identical image bytes into a
    single XObject, so reusing one pixmap builds a document of N placements of
    ONE picture -- which the captioning pass now describes once. A fixture
    meant to exercise N candidates has to contain N different pictures, or it
    silently becomes a test of the dedup path instead.
    """
    return _content_pixmap(color=(10 + index * 9, 20, 30))


def _doc_with_images(count, *, pages=1):
    doc = pymupdf.open()
    for page_index in range(pages):
        page = doc.new_page(width=595, height=842)
        for i in range(count):
            top = 50 + i * 30
            page.insert_image(
                pymupdf.Rect(50, top, 500, top + 25),
                pixmap=_distinct_pixmap(page_index * count + i),
            )
    return doc


def _doc_with_one_shrinking_image_per_page(pages):
    """One image per page, strictly decreasing in area down the document.

    Distinct widths make each rendered region distinct bytes, so a fake vision
    client can tell which candidate an image belongs to -- and distinct pixmaps
    make each page a distinct picture, so the pass treats them as three
    candidates rather than one repeated three times.
    """
    doc = pymupdf.open()
    for index in range(pages):
        page = doc.new_page(width=595, height=842)
        page.insert_image(
            pymupdf.Rect(50, 50, 550 - index * 60, 400), pixmap=_distinct_pixmap(index)
        )
    return doc


def _doc_with_single_raster(paint):
    """One-page document with one raster region, painted by ``paint(pix)``.

    1000x1000 source so a single accent row scales down to a small enough
    fraction of the rendered region to land the ``color_topusage`` fraction
    just under 1.0 rather than well under it -- see
    ``_paint_near_blank_with_a_thin_line``.
    """
    doc = pymupdf.open()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 1000, 1000))
    paint(pix)
    page = doc.new_page(width=595, height=842)
    page.insert_image(pymupdf.Rect(50, 50, 500, 400), pixmap=pix)
    return doc


def _paint_blank(pix):
    pix.set_rect(pix.irect, (255, 255, 255))


def _paint_near_blank_with_a_thin_line(pix):
    """Mostly white with a single one-pixel accent row.

    A stand-in for a real plot region (mostly white, thin lines). Measured
    through the full ``inspect_page`` render path (dpi=150, ``detail="high"``,
    the region shape ``_doc_with_single_raster`` produces): the rendered
    ``color_topusage`` fraction is 0.9972 -- below the 1.0 blank threshold,
    but above a naively loosened 0.99, which is exactly the false-positive
    this fixture is built to catch.
    """
    pix.set_rect(pix.irect, (255, 255, 255))
    pix.set_rect(pymupdf.IRect(0, 500, 1000, 501), (0, 0, 0))


def _figures(doc):
    from datasheetindex.core.textfile import scan_pages

    return scan_pages(doc).figures


def test_every_raster_region_is_captioned():
    doc = _doc_with_images(2)
    figures = _figures(doc)
    vision = RecordingVision()
    outcome = caption_figures_in_place(
        doc, figures, vision_client=vision, max_figure_captions=20
    )
    doc.close()

    raster = [f for f in figures if f["kind"] == "raster"]
    assert len(vision.calls) == len(raster)
    assert vision.systems == [CAPTION_SYSTEM_PROMPT] * len(raster)
    assert outcome == CaptionOutcome(
        captioned=len(raster), pending=0, excluded_above_max=0, failed=False
    )
    for entry in raster:
        assert entry["caption"] == "a table of device attributes"
        assert entry["caption_source"] == "llm"


def test_a_region_on_a_captioned_page_is_still_captioned():
    # Pins the no-triage decision. Skipping "already captioned" regions is
    # section 4's forbidden caption-to-region association by another route.
    doc = _doc_with_images(1)
    writer = pymupdf.TextWriter(doc[0].rect)
    writer.append((72, 800), "Figure 4. Package outline")
    writer.write_text(doc[0])
    figures = _figures(doc)
    vision = RecordingVision()
    caption_figures_in_place(doc, figures, vision_client=vision, max_figure_captions=20)
    doc.close()

    assert len(vision.calls) == 1
    assert any(f["caption_source"] == "llm" for f in figures)
    assert any(f["caption_source"] == "text" for f in figures)


def test_cap_keeps_the_largest_regions_and_discloses_the_rest():
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    # Three regions of strictly decreasing area, and three distinct pictures --
    # one pixmap placed three times is one picture, which the cap now counts
    # once (see test_the_cap_counts_distinct_images_not_placements).
    page.insert_image(pymupdf.Rect(50, 50, 550, 400), pixmap=_distinct_pixmap(0))
    page.insert_image(pymupdf.Rect(50, 420, 400, 600), pixmap=_distinct_pixmap(1))
    page.insert_image(pymupdf.Rect(50, 620, 250, 700), pixmap=_distinct_pixmap(2))
    figures = _figures(doc)
    vision = RecordingVision()
    outcome = caption_figures_in_place(
        doc, figures, vision_client=vision, max_figure_captions=2
    )
    doc.close()

    assert outcome.captioned == 2
    assert outcome.excluded_above_max == 1
    assert outcome.failed is False  # a disclosed cap is not a failure
    captioned = [f for f in figures if f["caption_source"] == "llm"]
    areas = sorted((f["page_area_pct"] for f in captioned), reverse=True)
    smallest = min(f["page_area_pct"] for f in figures if f["kind"] == "raster")
    assert smallest not in areas


def test_zero_cap_captions_nothing():
    doc = _doc_with_images(2)
    figures = _figures(doc)
    vision = RecordingVision()
    outcome = caption_figures_in_place(
        doc, figures, vision_client=vision, max_figure_captions=0
    )
    doc.close()

    assert vision.calls == []
    assert outcome.captioned == 0
    assert outcome.pending == 0
    assert outcome.failed is False


def test_no_vision_client_reports_pending_not_failed():
    doc = _doc_with_images(2)
    figures = _figures(doc)
    outcome = caption_figures_in_place(
        doc, figures, vision_client=None, max_figure_captions=20
    )
    doc.close()

    raster = [f for f in figures if f["kind"] == "raster"]
    assert outcome.pending == len(raster)
    assert outcome.captioned == 0
    assert outcome.failed is False  # absence of a client is stable, not a defect
    assert all(f["caption"] is None for f in raster)


def test_no_vision_client_and_a_zero_cap_reports_nothing_pending():
    # Counting all candidates instead of eligible ones would mark work pending
    # that the caller declined, and reuse would then rebuild forever on any
    # machine with a key.
    doc = _doc_with_images(2)
    figures = _figures(doc)
    outcome = caption_figures_in_place(
        doc, figures, vision_client=None, max_figure_captions=0
    )
    doc.close()

    assert outcome.pending == 0


def test_a_raising_call_leaves_the_build_successful_but_failed_flagged():
    doc = _doc_with_images(2)
    figures = _figures(doc)
    vision = RecordingVision(fail_on={1})
    outcome = caption_figures_in_place(
        doc, figures, vision_client=vision, max_figure_captions=20
    )
    doc.close()

    assert outcome.failed is True
    assert outcome.pending == 0  # a client existed; this is transient, not absent
    assert outcome.captioned == len(vision.calls) - 1
    # The figures the call did not cover are untouched, not half-written.
    uncaptioned = [f for f in figures if f["caption_source"] is None]
    assert len(uncaptioned) == 1
    assert uncaptioned[0]["caption"] is None


@pytest.mark.parametrize("reply", ["", "   \n  ", "\n"])
def test_an_empty_response_is_a_failure_not_a_caption(reply):
    doc = _doc_with_images(1)
    figures = _figures(doc)
    vision = RecordingVision(reply=reply)
    outcome = caption_figures_in_place(
        doc, figures, vision_client=vision, max_figure_captions=20
    )
    doc.close()

    raster = [f for f in figures if f["kind"] == "raster"]
    assert outcome.failed is True
    assert outcome.captioned == 0
    # None, not "" -- an empty string in a published artifact reads as "the
    # model said this figure has no description".
    assert raster[0]["caption"] is None
    assert raster[0]["caption_source"] is None


def test_captions_are_stripped():
    doc = _doc_with_images(1)
    figures = _figures(doc)
    caption_figures_in_place(
        doc,
        figures,
        vision_client=RecordingVision(reply="  a pinout diagram\n"),
        max_figure_captions=20,
    )
    doc.close()

    assert figures[0]["caption"] == "a pinout diagram"


def test_rendering_is_serial_on_the_calling_thread_and_precedes_dispatch(monkeypatch):
    # PyMuPDF is not thread-safe for concurrent page work; the parallel table
    # scan already carries that scar with measured wrong counts. "All renders
    # before any dispatch" is not enough on its own -- a render pool would
    # satisfy it -- so the thread the renders run on is pinned too.
    import datasheetindex.llm.figure_captions as mod

    events = []
    rendered_regions = []
    render_threads = []
    real_inspect = mod.inspect_page

    def recording_inspect(doc, page, region=None, detail="high"):
        events.append("render")
        rendered_regions.append(region)
        render_threads.append(threading.get_ident())
        return real_inspect(doc, page, region=region, detail=detail)

    monkeypatch.setattr(mod, "inspect_page", recording_inspect)

    class Dispatcher:
        def describe_image(self, system, image_base64, *, media_type="image/png"):
            events.append("dispatch")
            return "x"

    doc = _doc_with_images(3)
    figures = _figures(doc)
    caption_figures_in_place(
        doc, figures, vision_client=Dispatcher(), max_figure_captions=20
    )
    doc.close()

    assert "dispatch" in events and "render" in events
    assert events.index("dispatch") > len([e for e in events if e == "render"]) - 1, (
        "a dispatch was interleaved with rendering"
    )
    assert events[: events.index("dispatch")] == ["render"] * events.index("dispatch")
    # Serial, not merely earlier: every render ran on the thread that called
    # caption_figures_in_place, so no render pool slipped in.
    assert len(render_threads) == 3
    assert set(render_threads) == {threading.get_ident()}
    # The regions were clipped and normalized upstream and inspect_page rejects
    # anything outside 0.0-1.0: they must arrive exactly as indexed, neither
    # re-clamped nor rounded.
    raster = [f for f in figures if f["kind"] == "raster"]
    assert rendered_regions == [entry["region"] for entry in raster]


def test_results_are_applied_in_candidate_order_not_completion_order(monkeypatch):
    # The figures array is hashed as part of the artifact; completion order
    # would change the bytes on every build and defeat reuse.
    import datasheetindex.llm.figure_captions as mod

    doc = _doc_with_one_shrinking_image_per_page(3)
    figures = _figures(doc)

    page_of_image: dict[str, int] = {}
    real_inspect = mod.inspect_page

    def recording_inspect(doc, page, region=None, detail="high"):
        blocks = real_inspect(doc, page, region=region, detail=detail)
        page_of_image[blocks[0]["data"]] = page
        return blocks

    monkeypatch.setattr(mod, "inspect_page", recording_inspect)

    class CompletesInReverse:
        """Answers correctly, but the first dispatch is the last to return."""

        def __init__(self):
            self._lock = threading.Lock()
            self._dispatched = 0

        def describe_image(self, system, image_base64, *, media_type="image/png"):
            with self._lock:
                self._dispatched += 1
                position = self._dispatched
            time.sleep(0.05 * (4 - position))
            return f"page {page_of_image[image_base64]}"

    caption_figures_in_place(
        doc, figures, vision_client=CompletesInReverse(), max_figure_captions=20
    )
    doc.close()

    # Each region carries the answer for its own image, so a result zipped back
    # in completion order lands on the wrong figure.
    assert [f["caption"] for f in figures] == ["page 1", "page 2", "page 3"]


def test_the_figures_array_is_not_reordered():
    # Sorting candidates by area must not reach the caller's list: the ToC JSON
    # publishes figures in document order.
    doc = pymupdf.open()
    for index in range(3):
        page = doc.new_page(width=595, height=842)
        # Area grows with the page number, so candidate order is the reverse of
        # document order.
        page.insert_image(
            pymupdf.Rect(50, 50, 200 + index * 150, 400), pixmap=_distinct_pixmap(index)
        )
    figures = _figures(doc)
    before = [f["page"] for f in figures]
    caption_figures_in_place(
        doc, figures, vision_client=RecordingVision(), max_figure_captions=20
    )
    doc.close()

    assert before == [1, 2, 3]
    assert [f["page"] for f in figures] == before


def test_default_cap_is_twenty():
    assert DEFAULT_MAX_FIGURE_CAPTIONS == 20


def test_a_blank_region_is_not_dispatched_and_reports_as_blank():
    # A confirmed hallucination on ti-tlv9061.pdf page 46 named a blank
    # region as "a schematic diagram ... optocoupler component". The guard
    # must stop it before the vision client is ever called.
    doc = _doc_with_single_raster(_paint_blank)
    figures = _figures(doc)
    vision = RecordingVision()
    outcome = caption_figures_in_place(
        doc, figures, vision_client=vision, max_figure_captions=20
    )
    doc.close()

    raster = [f for f in figures if f["kind"] == "raster"]
    assert len(raster) == 1
    assert vision.calls == []
    assert raster[0]["caption"] is None
    assert raster[0]["caption_source"] is None
    assert outcome.blank == 1
    assert outcome.captioned == 0


def test_a_blank_skip_sets_neither_failed_nor_pending():
    # Regression guard for the cache-poisoning hazard: `failed` propagates to
    # `llm_enrichment_incomplete`, which would rebuild a document forever for
    # a page that is blank on every build, since blank-ness never changes.
    # `pending` would claim no vision client was available, which is false --
    # a client existed and simply had nothing worth calling it for.
    doc = _doc_with_single_raster(_paint_blank)
    figures = _figures(doc)
    vision = RecordingVision()
    outcome = caption_figures_in_place(
        doc, figures, vision_client=vision, max_figure_captions=20
    )
    doc.close()

    # The discriminator that ties this to the blank-skip path specifically:
    # without the guard the region is dispatched normally, gets a canned
    # reply, and *also* leaves failed/pending False -- so asserting only
    # those two would pass by accident. Zero calls is what only the guard
    # produces.
    assert vision.calls == []
    assert outcome.failed is False
    assert outcome.pending == 0


def test_a_near_blank_region_with_real_content_is_still_dispatched():
    # False-positive guard: a region that is mostly but not entirely one
    # colour -- a thin line on white, what a real plot region looks like --
    # must still reach the vision client. This is the test a loosened
    # threshold (e.g. 0.99) would break.
    doc = _doc_with_single_raster(_paint_near_blank_with_a_thin_line)
    figures = _figures(doc)
    vision = RecordingVision()
    outcome = caption_figures_in_place(
        doc, figures, vision_client=vision, max_figure_captions=20
    )
    doc.close()

    raster = [f for f in figures if f["kind"] == "raster"]
    assert len(vision.calls) == 1
    assert outcome.blank == 0
    assert outcome.captioned == 1
    assert raster[0]["caption_source"] == "llm"


def _doc_with_one_image_repeated(*, pages):
    """One image XObject drawn once per page -- a header logo, in effect.

    PyMuPDF folds identical bytes into a single XObject, so this is what a
    repeated vendor logo looks like on disk: N placements, one picture.
    """
    doc = pymupdf.open()
    pix = _content_pixmap()
    for _ in range(pages):
        page = doc.new_page(width=595, height=842)
        page.insert_image(pymupdf.Rect(50, 50, 550, 400), pixmap=pix)
    return doc


def test_one_image_placed_many_times_costs_one_call():
    """Measured on a real corpus: 9 of 86 caption candidates (10%) are repeats.

    On onsemi's PCNs every candidate above the area threshold is the same
    header logo, so before this the document spent its whole caption budget
    describing one picture four times.
    """
    doc = _doc_with_one_image_repeated(pages=4)
    figures = _figures(doc)
    vision = RecordingVision()
    outcome = caption_figures_in_place(
        doc, figures, vision_client=vision, max_figure_captions=20
    )
    doc.close()

    raster = [f for f in figures if f["kind"] == "raster"]
    assert len(raster) == 4
    assert len(vision.calls) == 1
    assert outcome == CaptionOutcome(
        captioned=1, pending=0, excluded_above_max=0, failed=False, shared=3
    )
    # Every placement still carries the caption: the index must not go quiet
    # about pages 2-4 just because the picture was described once.
    for entry in raster:
        assert entry["caption"] == "a table of device attributes"
        assert entry["caption_source"] == "llm"


def test_the_cap_counts_distinct_images_not_placements():
    """The point of the dedup: repeats stop crowding out real content."""
    doc = _doc_with_one_image_repeated(pages=3)
    page = doc.new_page(width=595, height=842)
    page.insert_image(
        pymupdf.Rect(50, 450, 500, 700), pixmap=_content_pixmap((9, 9, 9))
    )
    figures = _figures(doc)
    vision = RecordingVision()
    outcome = caption_figures_in_place(
        doc, figures, vision_client=vision, max_figure_captions=2
    )
    doc.close()

    # Two distinct pictures, four placements, a cap of 2 -- and the second
    # picture is reached, which it never would be if placements were counted.
    assert len(vision.calls) == 2
    assert outcome.captioned == 2
    assert outcome.shared == 2
    assert outcome.excluded_above_max == 0
    assert all(f["caption"] for f in figures if f["kind"] == "raster")


def test_excluded_above_max_counts_placements_denied_a_caption():
    """The artifact's `above_max` answers "how many entries lack a caption".

    Counting distinct images instead would under-report it: one dropped
    picture placed three times leaves three entries uncaptioned, and the
    consumer of `figure_captions_excluded` is looking at entries.
    """
    doc = _doc_with_one_image_repeated(pages=3)
    page = doc.new_page(width=595, height=842)
    # Strictly larger, so it sorts ahead and the repeated image is what the
    # cap of 1 drops.
    page.insert_image(pymupdf.Rect(20, 20, 575, 800), pixmap=_content_pixmap((9, 9, 9)))
    figures = _figures(doc)
    vision = RecordingVision()
    outcome = caption_figures_in_place(
        doc, figures, vision_client=vision, max_figure_captions=1
    )
    doc.close()

    assert len(vision.calls) == 1
    assert outcome.captioned == 1
    assert outcome.shared == 0
    assert outcome.excluded_above_max == 3


def test_entries_without_a_known_xref_never_group():
    """An unknown identity is not a shared identity.

    Artifacts built before the xref field, and any hand-built entry, carry no
    usable identity. Treating those as equal would hand one caption to every
    unrelated figure in the document -- silently wrong, and wrong in the
    direction that produces confident nonsense.
    """
    doc = _doc_with_images(2)
    figures = _figures(doc)
    for entry in figures:
        if entry["kind"] == "raster":
            entry["xref"] = 0
    vision = RecordingVision()
    outcome = caption_figures_in_place(
        doc, figures, vision_client=vision, max_figure_captions=20
    )
    doc.close()

    assert len(vision.calls) == 2
    assert outcome.captioned == 2
    assert outcome.shared == 0


def test_eligible_count_counts_distinct_images():
    """Gate and pass must agree, or a client is built for work that never runs.

    ``index.build`` asks this to decide whether constructing a vision client
    can pay for itself. If it counted placements while the pass counts
    pictures, the two definitions would drift apart in exactly the case this
    dedup creates.
    """
    from datasheetindex.llm.figure_captions import eligible_caption_count

    doc = _doc_with_one_image_repeated(pages=4)
    figures = _figures(doc)
    doc.close()

    assert eligible_caption_count(figures, 20) == 1
    assert eligible_caption_count(figures, 0) == 0


def test_a_tls_failure_is_absorbed_here_rather_than_destroying_the_artifact():
    """The counterpart to the ToC fallback, and deliberately the other way round.

    Captioning runs at step 6b of ``index.build``; the artifacts are written at
    step 8. Raising here would abort the build and write *nothing* -- for a
    document whose index, and possibly whose ToC, is otherwise complete. An
    unusable artifact is worse than uncaptioned figures, so this one failure
    stays absorbed. Only the logged message improves, because the named error
    carries the remedy that ``openai``'s "Connection error." did not.
    """
    from datasheetindex.llm.client import LlmTlsVerificationError

    class _TlsVision:
        def describe_image(self, *_args, **_kwargs):
            raise LlmTlsVerificationError("add the CA to the trust store")

    doc = _doc_with_images(2)
    figures = _figures(doc)
    try:
        outcome = caption_figures_in_place(
            doc, figures, vision_client=_TlsVision(), max_figure_captions=20
        )
    finally:
        doc.close()

    assert outcome.failed is True
    assert outcome.captioned == 0
