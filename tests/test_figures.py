"""Tests for per-page figure entry construction."""

from __future__ import annotations

from typing import cast

import pymupdf
import pytest

from datasheetindex.core.figures import (
    DEFAULT_MIN_AREA_PCT,
    caption_entries,
    raster_regions,
)
from datasheetindex.tools.vision import inspect_page


def _page_with_image(rect: pymupdf.Rect, *, page_size=(595, 842)) -> pymupdf.Document:
    """One-page PDF with a solid PNG placed at `rect`."""
    doc = pymupdf.open()
    page = doc.new_page(width=page_size[0], height=page_size[1])
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 20))
    pix.set_rect(pix.irect, (255, 0, 0))
    page.insert_image(rect, pixmap=pix, keep_proportion=False)
    return doc


def test_one_image_yields_one_entry_with_geometry():
    doc = _page_with_image(pymupdf.Rect(100, 200, 400, 500))
    entries, excluded = raster_regions(doc[0])
    doc.close()

    assert excluded == 0
    assert len(entries) == 1
    entry = entries[0]
    assert entry["page"] == 1
    assert entry["kind"] == "raster"
    assert entry["caption"] is None
    assert entry["caption_source"] is None
    assert entry["bbox"] == pytest.approx([100, 200, 400, 500], abs=0.5)
    # 300x300 points of a 595x842 page
    assert entry["page_area_pct"] == pytest.approx(
        100.0 * (300 * 300) / (595 * 842), abs=0.1
    )
    assert entry["pixels"] == [40, 20]


def test_region_round_trips_back_to_bbox():
    doc = _page_with_image(pymupdf.Rect(100, 200, 400, 500))
    page_rect = doc[0].rect
    entry = raster_regions(doc[0])[0][0]
    doc.close()

    region = cast("dict[str, float]", entry["region"])
    assert region["left"] * page_rect.width == pytest.approx(100, abs=0.5)
    assert region["right"] * page_rect.width == pytest.approx(400, abs=0.5)
    assert region["top"] * page_rect.height == pytest.approx(200, abs=0.5)
    assert region["bottom"] * page_rect.height == pytest.approx(500, abs=0.5)


def test_region_is_accepted_by_the_real_inspect_page():
    # The coordinate contract, asserted against the real consumer rather than
    # by inspection, so the two cannot drift.
    doc = _page_with_image(pymupdf.Rect(100, 200, 400, 500))
    entry = raster_regions(doc[0])[0][0]
    rendered = inspect_page(
        doc, page=1, region=cast("dict[str, float]", entry["region"])
    )
    doc.close()

    assert rendered[0]["type"] == "image"


def test_image_overflowing_the_page_is_clipped_not_dropped():
    # ti-tlv9061 has 9 real placements like this. Unclipped, inspect_page raises.
    doc = _page_with_image(pymupdf.Rect(400, 700, 900, 1200))
    entries, _ = raster_regions(doc[0])
    entry = entries[0]
    region = cast("dict[str, float]", entry["region"])
    rendered = inspect_page(doc, page=1, region=region)
    doc.close()

    for edge, value in region.items():
        assert 0.0 <= value <= 1.0, f"{edge}={value} outside [0, 1]"
    bbox = cast("list[float]", entry["bbox"])
    assert bbox[2] == pytest.approx(595, abs=0.5)
    assert bbox[3] == pytest.approx(842, abs=0.5)
    assert rendered[0]["type"] == "image"


def test_image_entirely_off_page_is_dropped():
    doc = _page_with_image(pymupdf.Rect(700, 900, 800, 1000))
    entries, excluded = raster_regions(doc[0])
    doc.close()

    assert entries == []
    assert excluded == 0  # dropped as invisible, not as below-threshold


def test_region_is_normalized_against_a_non_zero_page_origin():
    doc = _page_with_image(pymupdf.Rect(100, 100, 200, 200))
    page = doc[0]
    page.set_cropbox(pymupdf.Rect(50, 50, 550, 800))
    page = doc.reload_page(page)
    entries, _ = raster_regions(page)
    region = cast("dict[str, float]", entries[0]["region"])
    rendered = inspect_page(doc, page=1, region=region)
    doc.close()

    rect = pymupdf.Rect(50, 50, 550, 800)
    assert region["left"] == pytest.approx((100 - rect.x0) / rect.width, abs=0.01)
    assert rendered[0]["type"] == "image"


def test_images_below_min_area_pct_are_excluded_and_counted():
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 10, 10))
    pix.set_rect(pix.irect, (0, 0, 255))
    page.insert_image(pymupdf.Rect(10, 10, 30, 30), pixmap=pix)  # ~0.08%
    page.insert_image(pymupdf.Rect(100, 100, 400, 400), pixmap=pix)  # ~1.8%
    entries, excluded = raster_regions(page, min_area_pct=DEFAULT_MIN_AREA_PCT)
    doc.close()

    assert len(entries) == 1
    assert excluded == 1


def test_page_without_images_yields_nothing():
    doc = pymupdf.open()
    doc.new_page(width=595, height=842)
    entries, excluded = raster_regions(doc[0])
    doc.close()

    assert entries == []
    assert excluded == 0


def test_split_form_joins_the_next_non_empty_line():
    entries = caption_entries(7, "intro\nFigure 2\nBlock diagram\nmore body")

    assert len(entries) == 1
    assert entries[0] == {
        "page": 7,
        "kind": "caption",
        "region": None,
        "bbox": None,
        "figure_number": "2",
        "caption": "Figure 2 Block diagram",
        "caption_source": "text",
    }


def test_same_line_form_requires_its_separator():
    entries = caption_entries(3, "Figure 3. Package outline")

    assert len(entries) == 1
    assert entries[0]["figure_number"] == "3"
    assert entries[0]["caption"] == "Figure 3. Package outline"


def test_section_relative_numbering_in_both_forms():
    # 404 of the corpus's 492 captions take the same-line hyphenated form.
    same_line = caption_entries(9, "Figure 10-1. Reset Logic")
    split = caption_entries(9, "Figure 3-2\nClock tree\n")
    en_dash = caption_entries(9, "Figure 10–1. Reset Logic")

    assert same_line[0]["figure_number"] == "10-1"
    assert same_line[0]["caption"] == "Figure 10-1. Reset Logic"
    assert split[0]["figure_number"] == "3-2"
    assert split[0]["caption"] == "Figure 3-2 Clock tree"
    assert en_dash[0]["figure_number"] == "10–1"


def test_figure_number_is_a_string_even_when_plain():
    entries = caption_entries(1, "Figure 4. Timing")

    assert entries[0]["figure_number"] == "4"
    assert isinstance(entries[0]["figure_number"], str)


def test_fig_abbreviation_is_accepted():
    entries = caption_entries(2, "Fig. 10. Enable and disable times")

    assert entries[0]["figure_number"] == "10"


@pytest.mark.parametrize(
    "text",
    [
        "as Figure 5 shows the limit is 3V",
        "Figure 2 shows the major subsystems of the device",
        "Figure 6-2 shows the structure of the 32 general purpose registers",
        "See Figure 3",
    ],
)
def test_prose_is_not_a_caption(text):
    # The mandatory separator rejected 70 prose lines across the corpus while
    # admitting 404 real same-line captions. Widening the number pattern for
    # section-relative numbering widened the prose surface with it.
    assert caption_entries(1, text) == []


def test_bare_figure_number_as_the_last_line_yields_nothing():
    assert caption_entries(1, "body text\nFigure 9\n") == []


def test_page_with_no_captions_yields_nothing():
    assert caption_entries(1, "Absolute Maximum Ratings\nVCC 5.5 V") == []


def _doc_with_repeated_image(placements: int) -> pymupdf.Document:
    """One image XObject placed once per page, `placements` pages.

    PyMuPDF folds identical image bytes into a single XObject, which is also
    what a real vendor logo in a page header looks like on disk.
    """
    doc = pymupdf.open()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 20))
    pix.set_rect(pix.irect, (255, 0, 0))
    for _ in range(placements):
        page = doc.new_page(width=595, height=842)
        page.insert_image(pymupdf.Rect(100, 100, 400, 400), pixmap=pix)
    return doc


def test_repeated_placements_of_one_image_share_an_xref():
    """The identity that lets the captioning pass avoid paying twice.

    Without it, a logo in a page header is a fresh figure on every page: N
    placements, N VLM calls, N identical captions.
    """
    doc = _doc_with_repeated_image(3)
    xrefs = [raster_regions(page)[0][0]["xref"] for page in doc]
    doc.close()

    assert all(isinstance(x, int) and x > 0 for x in xrefs)
    assert len(set(xrefs)) == 1


def test_distinct_images_do_not_share_an_xref():
    """The other half of the identity: different pictures stay different."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    for index, colour in enumerate([(255, 0, 0), (0, 255, 0)]):
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 20))
        pix.set_rect(pix.irect, colour)
        page.insert_image(
            pymupdf.Rect(100, 100 + index * 310, 400, 400 + index * 310), pixmap=pix
        )
    entries, _ = raster_regions(page)
    doc.close()

    assert len(entries) == 2
    assert entries[0]["xref"] != entries[1]["xref"]


def test_blocked_captioning_is_published_in_the_json(tmp_path, monkeypatch):
    """An agent must be able to tell "nothing to caption" from "captioning is broken".

    Both leave every figure uncaptioned, and only the builder knows which
    happened. Left unpublished, an agent keeps asking for captions that can
    never arrive.
    """
    import pymupdf

    from datasheetindex import DatasheetIndex
    from datasheetindex.llm.client import LlmTlsVerificationError

    pdf = tmp_path / "figs.pdf"
    doc = pymupdf.open()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 400, 400))
    pix.set_rect(pix.irect, (255, 255, 255))
    pix.set_rect(pymupdf.IRect(0, 200, 400, 201), (0, 0, 0))
    page = doc.new_page(width=595, height=842)
    page.insert_image(pymupdf.Rect(50, 50, 500, 400), pixmap=pix)
    doc.save(str(pdf))
    doc.close()

    class _TlsVision:
        def describe_image(self, *_args, **_kwargs):
            raise LlmTlsVerificationError("certificate verify failed")

    monkeypatch.setattr(
        "datasheetindex.index.get_vision_client", lambda _c: _TlsVision()
    )
    monkeypatch.setattr(
        DatasheetIndex, "_try_create_default_llm_client", lambda _self: object()
    )

    with DatasheetIndex(str(pdf)) as idx:
        artifacts = idx.build(output_dir=str(tmp_path / "out"))

    assert artifacts.json_data["figure_captions_blocked"] is True
