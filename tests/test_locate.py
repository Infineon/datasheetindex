"""Tests for locate_text coordinate grounding."""

from __future__ import annotations

import base64
from pathlib import Path

import pymupdf
import pytest

from datasheetindex.core.locate import locate_text
from datasheetindex.tools.vision import inspect_page

DATA2PAGE_DIR = Path(__file__).resolve().parent.parent.parent / "data2page"
TLE9350_PATH = DATA2PAGE_DIR / "Infineon-TLE9350BSJ-DataSheet-v01_00-EN.pdf"


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


def test_dash_mismatch_falls_back_to_tokens():
    # PDF text uses a Unicode minus (U+2212); the ASCII-hyphen query only
    # matches via the normalizing token fallback.
    minus = chr(0x2212)
    doc = _doc_with([(72, 72, f"{minus}0.3")])
    results = locate_text(doc, "-0.3", page=1)
    doc.close()

    assert len(results) == 1
    assert results[0]["match_method"] == "tokens"
    assert len(results[0]["boxes"]) == 1


def test_multi_line_phrase_unions_boxes_via_tokens():
    # A phrase wrapping across two ADJACENT lines: "range -9" then "stop" one
    # line below. The Unicode minus forces the token path, which groups the
    # matched words by (block_no, line_no) into one box per line.
    minus = chr(0x2212)
    doc = _doc_with([(72, 72, f"range {minus}9"), (72, 94, "stop")])
    results = locate_text(doc, "range -9 stop", page=1)
    doc.close()

    assert len(results) == 1
    loc = results[0]
    assert loc["match_method"] == "tokens"
    assert len(loc["boxes"]) == 2
    # region is the union: it spans from the top line to the bottom line.
    assert loc["region"]["points"]["y0"] == pytest.approx(
        min(b["points"]["y0"] for b in loc["boxes"])
    )
    assert loc["region"]["points"]["y1"] == pytest.approx(
        max(b["points"]["y1"] for b in loc["boxes"])
    )
    assert loc["region"] != loc["boxes"][0]


def test_real_fixture_locates_part_number():
    # Real-world text/word structure (spec testing-plan case 10). The part
    # number is in the document title, so it is present on page 1.
    if not TLE9350_PATH.exists():
        pytest.skip("Test PDF not found")
    doc = pymupdf.open(str(TLE9350_PATH))
    results = locate_text(doc, "TLE9350", page=1)
    doc.close()

    assert results, "expected to locate the part number on page 1"
    loc = results[0]
    assert loc["page"] == 1
    assert len(loc["boxes"]) >= 1
    assert 0.0 <= loc["region"]["pct"]["left"] <= 1.0
    assert 0.0 <= loc["region"]["pct"]["bottom"] <= 1.0


def test_max_results_zero_raises():
    doc = _doc_with([(72, 72, "Hello")])
    with pytest.raises(ValueError, match="at least 1"):
        locate_text(doc, "Hello", page=1, max_results=0)
    doc.close()


def test_page_none_scans_all_pages():
    # Two-page doc; the query appears only on page 2. page=None must find it.
    doc = pymupdf.open()
    p1 = doc.new_page()
    w1 = pymupdf.TextWriter(p1.rect)
    w1.append((72, 72), "first page only")
    w1.write_text(p1)
    p2 = doc.new_page()
    w2 = pymupdf.TextWriter(p2.rect)
    w2.append((72, 72), "needle here")
    w2.write_text(p2)
    results = locate_text(doc, "needle", page=None)
    doc.close()

    assert len(results) == 1
    assert results[0]["page"] == 2
