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


def _doc_with_images(count, *, pages=1):
    doc = pymupdf.open()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20))
    pix.set_rect(pix.irect, (10, 20, 30))
    for _ in range(pages):
        page = doc.new_page(width=595, height=842)
        for i in range(count):
            top = 50 + i * 30
            page.insert_image(pymupdf.Rect(50, top, 500, top + 25), pixmap=pix)
    return doc


def _doc_with_one_shrinking_image_per_page(pages):
    """One image per page, strictly decreasing in area down the document.

    Distinct widths make each rendered region distinct bytes, so a fake vision
    client can tell which candidate an image belongs to.
    """
    doc = pymupdf.open()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20))
    pix.set_rect(pix.irect, (10, 20, 30))
    for index in range(pages):
        page = doc.new_page(width=595, height=842)
        page.insert_image(pymupdf.Rect(50, 50, 550 - index * 60, 400), pixmap=pix)
    return doc


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
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20))
    pix.set_rect(pix.irect, (1, 2, 3))
    # Three regions of strictly decreasing area.
    page.insert_image(pymupdf.Rect(50, 50, 550, 400), pixmap=pix)
    page.insert_image(pymupdf.Rect(50, 420, 400, 600), pixmap=pix)
    page.insert_image(pymupdf.Rect(50, 620, 250, 700), pixmap=pix)
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
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20))
    pix.set_rect(pix.irect, (10, 20, 30))
    for index in range(3):
        page = doc.new_page(width=595, height=842)
        # Area grows with the page number, so candidate order is the reverse of
        # document order.
        page.insert_image(pymupdf.Rect(50, 50, 200 + index * 150, 400), pixmap=pix)
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
