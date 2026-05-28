"""Tests for text file generation."""

import re
from pathlib import Path

import pymupdf
import pytest

import datasheetindex.core.textfile as textfile_module
from datasheetindex.core.textfile import extract_page_text, generate_text, search_text

DATA2PAGE_DIR = Path(__file__).resolve().parent.parent.parent / "data2page"
TLE9350_PATH = DATA2PAGE_DIR / "Infineon-TLE9350BSJ-DataSheet-v01_00-EN.pdf"


def test_markers_start_at_page_1():
    """PAGE markers should be 1-indexed."""
    doc = pymupdf.open()
    doc.new_page()
    doc.new_page()
    text = generate_text(doc)
    doc.close()
    assert "--- PAGE 1 ---" in text
    assert "--- PAGE 2 ---" in text
    assert "--- PAGE 0 ---" not in text


def test_marker_count_matches_pages():
    """Number of PAGE markers should match number of pages."""
    doc = pymupdf.open()
    for _ in range(5):
        doc.new_page()
    text = generate_text(doc)
    doc.close()
    markers = re.findall(r"--- PAGE \d+ ---", text)
    assert len(markers) == 5


def test_markers_sequential():
    """PAGE markers should be in sequential order."""
    doc = pymupdf.open()
    for _ in range(3):
        doc.new_page()
    text = generate_text(doc)
    doc.close()
    numbers = [int(m) for m in re.findall(r"--- PAGE (\d+) ---", text)]
    assert numbers == [1, 2, 3]


def test_extract_page_text_returns_single_page_without_marker():
    text = "--- PAGE 1 ---\nFirst page\n--- PAGE 2 ---\nSecond page\n"

    assert extract_page_text(text, 2) == "Second page"


def test_search_text_returns_page_aware_matches():
    text = "--- PAGE 1 ---\nAlpha beta\n--- PAGE 2 ---\nGamma alpha delta\n"

    matches = search_text(text, "alpha")

    assert matches == [
        {"page": 1, "start": 0, "end": 5, "snippet": "Alpha beta"},
        {"page": 2, "start": 6, "end": 11, "snippet": "Gamma alpha delta"},
    ]


def test_search_text_respects_page_filter_and_validation():
    text = "--- PAGE 1 ---\nAlpha beta\n--- PAGE 2 ---\nGamma alpha delta\n"

    matches = search_text(text, "alpha", page=2, max_results=1)

    assert matches == [
        {"page": 2, "start": 6, "end": 11, "snippet": "Gamma alpha delta"}
    ]

    with pytest.raises(ValueError, match="query must not be empty"):
        search_text(text, "")


def test_search_text_matches_across_collapsed_whitespace():
    text = "--- PAGE 1 ---\nTransmitted recessive bit\nwidth at 5 Mbit/s\n"

    matches = search_text(text, "Transmitted recessive bit width at 5 Mbit/s")

    assert len(matches) == 1
    assert matches[0]["page"] == 1
    assert matches[0]["start"] == 0
    assert matches[0]["snippet"] == "Transmitted recessive bit width at 5 Mbit/s"


def test_search_text_matches_interleaved_table_labels():
    text = (
        "--- PAGE 1 ---\n"
        "Transmitted recessive bit tBit(Bus)_5M 155 170 210 ns width at 5 Mbit/s\n"
    )

    matches = search_text(text, "Transmitted recessive bit width at 5 Mbit/s")

    assert len(matches) == 1
    assert matches[0]["page"] == 1
    assert matches[0]["start"] == 0
    assert "Transmitted recessive bit" in matches[0]["snippet"]
    assert "tBit(Bus)_5M" in matches[0]["snippet"]
    assert "width at 5 Mbit/s" in matches[0]["snippet"]


def test_search_text_token_fallback_respects_case_sensitivity():
    text = "--- PAGE 1 ---\nAlpha spacer beta spacer gamma\n"

    assert search_text(text, "alpha beta gamma", case_sensitive=True) == []

    matches = search_text(text, "alpha beta gamma")
    assert len(matches) == 1
    assert matches[0]["page"] == 1
    assert matches[0]["snippet"] == "Alpha spacer beta spacer gamma"


def test_search_text_skips_expensive_fallback_when_literal_hits(monkeypatch):
    text = "--- PAGE 1 ---\nAlpha beta\n--- PAGE 2 ---\nAlpha beta gamma\n"

    def _unexpected_fallback(*args, **kwargs):
        raise AssertionError("fallback should not run when literal matches exist")

    monkeypatch.setattr(
        textfile_module,
        "_find_collapsed_whitespace_spans",
        _unexpected_fallback,
    )
    monkeypatch.setattr(
        textfile_module,
        "_find_token_sequence_spans",
        _unexpected_fallback,
    )

    matches = search_text(text, "Alpha beta")

    assert matches == [
        {"page": 1, "start": 0, "end": 10, "snippet": "Alpha beta"},
        {"page": 2, "start": 0, "end": 10, "snippet": "Alpha beta gamma"},
    ]


def test_two_column_page_reads_left_then_right():
    """Two-column text should be read left column first, then right."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_textbox(
        pymupdf.Rect(50, 80, 280, 300),
        "Left column first paragraph. "
        "This text should appear before the right column content "
        "in the extracted output.",
        fontsize=10,
    )
    page.insert_textbox(
        pymupdf.Rect(320, 80, 560, 300),
        "Right column second paragraph. "
        "This text should appear after the left column content "
        "in the extracted output.",
        fontsize=10,
    )
    text = generate_text(doc)
    doc.close()

    left_pos = text.index("Left column first")
    right_pos = text.index("Right column second")
    assert left_pos < right_pos


def test_mixed_layout_title_columns_table():
    """Title, two columns, then full-width table should be ordered correctly."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)

    # Full-width title
    page.insert_text((50, 50), "Section Title", fontsize=14)

    # Left column
    page.insert_textbox(
        pymupdf.Rect(50, 80, 280, 250),
        "Left column describes the static and dynamic electrical "
        "characteristics of the device under test conditions.",
        fontsize=9,
    )

    # Right column
    page.insert_textbox(
        pymupdf.Rect(320, 80, 560, 250),
        "Right column defines operating limits within which the "
        "device performs as specified in the datasheet.",
        fontsize=9,
    )

    # Full-width table row below columns
    page.insert_textbox(
        pymupdf.Rect(50, 280, 560, 300),
        "Parameter    Symbol    Min    Typ    Max    Unit",
        fontsize=9,
    )

    text = generate_text(doc)
    doc.close()

    title_pos = text.index("Section Title")
    left_pos = text.index("Left column describes")
    right_pos = text.index("Right column defines")
    table_pos = text.index("Parameter")

    assert title_pos < left_pos
    assert left_pos < right_pos
    assert right_pos < table_pos


def test_single_column_page_unchanged():
    """Single-column pages should produce the same output as sort=True."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)

    page.insert_text((50, 50), "Title of Section", fontsize=14)
    page.insert_textbox(
        pymupdf.Rect(50, 80, 560, 300),
        "This is a full-width paragraph that spans the entire page. "
        "It should appear in normal top-to-bottom reading order. "
        "No column detection should trigger here.",
        fontsize=10,
    )
    page.insert_textbox(
        pymupdf.Rect(50, 320, 560, 400),
        "Second full-width paragraph below the first one.",
        fontsize=10,
    )

    text = generate_text(doc)
    doc.close()

    title_pos = text.index("Title of Section")
    first_pos = text.index("full-width paragraph that spans")
    second_pos = text.index("Second full-width paragraph")
    assert title_pos < first_pos < second_pos


@pytest.mark.real_pdf
def test_real_pdf_no_false_column_detection():
    """Column detection should not trigger on single-column TLE9350 datasheet."""
    if not TLE9350_PATH.exists():
        pytest.skip("Test PDF not found")
    doc = pymupdf.open(str(TLE9350_PATH))
    text_with_columns = generate_text(doc)

    # Compare against sort=True baseline: for a single-column datasheet,
    # the page marker count and content should be identical.
    baseline_parts: list[str] = []
    for page_idx in range(len(doc)):
        page_num = page_idx + 1
        baseline_parts.append(f"--- PAGE {page_num} ---")
        baseline_parts.append(doc[page_idx].get_text(sort=True))
    baseline = "\n".join(baseline_parts)
    doc.close()

    # Both outputs should have the same page markers
    col_markers = re.findall(r"--- PAGE (\d+) ---", text_with_columns)
    base_markers = re.findall(r"--- PAGE (\d+) ---", baseline)
    assert col_markers == base_markers

    # Content should be substantively the same -- same characters per page.
    # Block-based extraction may split or join whitespace differently from
    # sort=True (e.g. table cells "0.64" and "-0.23" vs "0.64-0.23"), so
    # we compare the sorted non-whitespace characters rather than word sets.
    col_pages = re.split(r"--- PAGE \d+ ---\n?", text_with_columns)[1:]
    base_pages = re.split(r"--- PAGE \d+ ---\n?", baseline)[1:]
    assert len(col_pages) == len(base_pages)
    for i, (col_page, base_page) in enumerate(zip(col_pages, base_pages, strict=True)):
        col_chars = sorted(col_page.replace(" ", "").replace("\n", ""))
        base_chars = sorted(base_page.replace(" ", "").replace("\n", ""))
        assert col_chars == base_chars, f"Page {i + 1}: character content differs"


@pytest.mark.real_pdf
def test_real_pdf_has_text_between_markers():
    """Real PDF should have non-empty text between markers."""
    if not TLE9350_PATH.exists():
        pytest.skip("Test PDF not found")
    doc = pymupdf.open(str(TLE9350_PATH))
    text = generate_text(doc)
    page_count = len(doc)
    doc.close()

    markers = re.findall(r"--- PAGE (\d+) ---", text)
    assert len(markers) == page_count

    # Split on markers and check that most pages have content
    sections = re.split(r"--- PAGE \d+ ---\n?", text)
    # First element is empty (before PAGE 1)
    content_sections = sections[1:]
    non_empty = [s for s in content_sections if s.strip()]
    assert len(non_empty) > page_count * 0.8
