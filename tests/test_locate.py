"""Tests for locate_text coordinate grounding."""

from __future__ import annotations

import base64

import pymupdf
import pytest

from datasheetindex.core.locate import locate_text
from datasheetindex.tools.vision import inspect_page


def _doc_with(text_at: list[tuple[float, float, str]]) -> pymupdf.Document:
    """One-page PDF with each (x, y, text) drawn at that baseline point."""
    doc = pymupdf.open()
    page = doc.new_page()
    writer = pymupdf.TextWriter(page.rect)
    for x, y, text in text_at:
        writer.append((x, y), text)
    writer.write_text(page)
    return doc


def test_fast_path_exact_hit():
    doc = _doc_with([(72, 72, "Hello world")])
    results = locate_text(doc, "Hello", page=1)
    doc.close()

    assert len(results) == 1
    loc = results[0]
    assert loc["page"] == 1
    assert loc["match_method"] == "search_for"
    assert len(loc["boxes"]) == 1
    assert loc["region"] == loc["boxes"][0]
    assert "pattern" not in loc  # single-string query is untagged


def test_not_found_returns_empty():
    doc = _doc_with([(72, 72, "Hello world")])
    assert locate_text(doc, "absent", page=1) == []
    doc.close()


def test_page_out_of_range_raises():
    doc = _doc_with([(72, 72, "Hello")])
    with pytest.raises(ValueError, match="between 1 and"):
        locate_text(doc, "Hello", page=5)
    doc.close()


def test_empty_query_raises():
    doc = _doc_with([(72, 72, "Hello")])
    with pytest.raises(ValueError, match="must not be empty"):
        locate_text(doc, "   ", page=1)
    doc.close()


def test_pct_points_consistency():
    doc = _doc_with([(72, 72, "Hello world")])
    page_rect = doc[0].rect
    loc = locate_text(doc, "Hello", page=1)[0]
    doc.close()

    box = loc["boxes"][0]
    assert loc["page_width"] == pytest.approx(page_rect.width)
    assert loc["page_height"] == pytest.approx(page_rect.height)
    assert box["pct"]["left"] * loc["page_width"] == pytest.approx(
        box["points"]["x0"] - page_rect.x0
    )
    assert box["pct"]["bottom"] * loc["page_height"] == pytest.approx(
        box["points"]["y1"] - page_rect.y0
    )


def test_round_trip_into_inspect_page():
    doc = _doc_with([(72, 72, "Hello world")])
    loc = locate_text(doc, "Hello", page=1)[0]
    cropped = inspect_page(doc, page=1, region=loc["region"]["pct"])
    full = inspect_page(doc, page=1)
    doc.close()

    assert cropped[0]["type"] == "image"
    assert len(base64.b64decode(cropped[0]["data"])) < len(
        base64.b64decode(full[0]["data"])
    )


def test_list_query_tags_and_caps():
    doc = _doc_with([(72, 72, "Hello world")])
    tagged = locate_text(doc, ["Hello", "world"], page=1)
    capped = locate_text(doc, ["Hello", "world"], page=1, max_results=1)
    deduped = locate_text(doc, ["Hello", "Hello"], page=1)
    doc.close()

    assert {r["pattern"] for r in tagged} == {"Hello", "world"}
    assert len(capped) == 1
    assert len(deduped) == 1  # same box found twice collapses; first pattern wins
    assert deduped[0]["pattern"] == "Hello"
