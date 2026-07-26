"""Tests for per-page figure entry construction."""

from __future__ import annotations

from typing import cast

import pymupdf
import pytest

from datasheetindex.core.figures import DEFAULT_MIN_AREA_PCT, raster_regions
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
