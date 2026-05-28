"""Tests for preamble extraction."""

from pathlib import Path

import pymupdf
import pytest

from datasheetindex.core.preamble import generate_preamble

DATA2PAGE_DIR = Path(__file__).resolve().parent.parent.parent / "data2page"
TLE9350_PATH = DATA2PAGE_DIR / "Infineon-TLE9350BSJ-DataSheet-v01_00-EN.pdf"


def test_single_page_doc():
    """Should handle a 1-page document without error."""
    doc = pymupdf.open()
    page = doc.new_page()
    writer = pymupdf.TextWriter(page.rect)
    writer.append((72, 72), "Single page content")
    writer.write_text(page)
    preamble = generate_preamble(doc)
    doc.close()
    assert "Single page" in preamble


def test_respects_max_chars():
    """Output should not exceed max_chars."""
    doc = pymupdf.open()
    for _ in range(2):
        page = doc.new_page()
        writer = pymupdf.TextWriter(page.rect)
        # Write enough text to exceed a small limit
        for y in range(72, 700, 14):
            writer.append((72, y), "A" * 80)
        writer.write_text(page)
    preamble = generate_preamble(doc, max_chars=200)
    doc.close()
    assert len(preamble) <= 200


def test_two_column_preamble_reads_left_then_right():
    """Preamble from a two-column page should read left column first."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_textbox(
        pymupdf.Rect(50, 80, 280, 300),
        "Left column overview of the product with full technical "
        "specifications and operating parameters listed here.",
        fontsize=10,
    )
    page.insert_textbox(
        pymupdf.Rect(320, 80, 560, 300),
        "Right column features including thermal protection and "
        "voltage regulation capabilities described in detail.",
        fontsize=10,
    )
    preamble = generate_preamble(doc)
    doc.close()

    left_pos = preamble.index("Left column overview")
    right_pos = preamble.index("Right column features")
    assert left_pos < right_pos


@pytest.mark.real_pdf
def test_real_pdf_contains_product_name():
    """Preamble from real datasheet should contain the product name."""
    if not TLE9350_PATH.exists():
        pytest.skip("Test PDF not found")
    doc = pymupdf.open(str(TLE9350_PATH))
    preamble = generate_preamble(doc)
    doc.close()
    assert len(preamble) > 0
    # The TLE9350 product name should appear in pages 1-2
    assert "TLE9350" in preamble


@pytest.mark.real_pdf
def test_real_pdf_respects_max_chars():
    """Real PDF preamble should respect max_chars."""
    if not TLE9350_PATH.exists():
        pytest.skip("Test PDF not found")
    doc = pymupdf.open(str(TLE9350_PATH))
    preamble = generate_preamble(doc, max_chars=500)
    doc.close()
    assert len(preamble) <= 500
