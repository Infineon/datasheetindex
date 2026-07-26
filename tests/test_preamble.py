"""Tests for preamble extraction."""

from pathlib import Path

import pymupdf
import pytest

from datasheetindex.core.preamble import (
    DEFAULT_MAX_CHARS,
    build_front_matter,
    generate_preamble,
)
from datasheetindex.core.textfile import _extract_page_text

DATA2PAGE_DIR = Path(__file__).resolve().parent.parent.parent / "data2page"
TLE9350_PATH = DATA2PAGE_DIR / "Infineon-TLE9350BSJ-DataSheet-v01_00-EN.pdf"


def _doc_with_lines(pages: int, lines: int = 6, width: int = 40):
    """A doc whose pages carry `lines` lines of `width` 'A' characters."""
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page()
        writer = pymupdf.TextWriter(page.rect)
        y = 72
        for _ in range(lines):
            writer.append((72, y), "A" * width)
            y += 14
        writer.write_text(page)
    return doc


def _framing_overhead(pages: int) -> int:
    """Marker + separator characters for pages 1..pages, per the spec."""
    return sum(13 + len(str(n)) for n in range(1, pages + 1)) + 2 * pages - 1


def test_page_markers_appear_once_per_page_in_order():
    doc = _doc_with_lines(2)
    fm = build_front_matter(doc)
    doc.close()

    assert fm.text.count("--- PAGE 1 ---") == 1
    assert fm.text.count("--- PAGE 2 ---") == 1
    assert fm.text.index("--- PAGE 1 ---") < fm.text.index("--- PAGE 2 ---")
    assert fm.text.startswith("--- PAGE 1 ---\n")


def test_front_matter_that_fits_is_emitted_whole():
    doc = _doc_with_lines(2)
    fm = build_front_matter(doc)
    doc.close()

    assert fm.chars_shown == fm.chars_extracted
    assert fm.char_truncated is False
    assert fm.pages_read == 2
    assert fm.total_pages == 2
    assert fm.pages_omitted == 0
    assert "NOTE" not in fm.text


def test_framing_overhead_matches_the_formula_on_two_pages():
    doc = _doc_with_lines(2)
    fm = build_front_matter(doc)
    doc.close()

    assert len(fm.text) - fm.chars_shown == _framing_overhead(2) == 31


def test_framing_overhead_matches_the_formula_at_three_digit_pages():
    """Blank pages extract to "", so the whole string is framing."""
    doc = pymupdf.open()
    for _ in range(105):
        doc.new_page()
    fm = build_front_matter(doc, max_pages=105)
    doc.close()

    assert fm.chars_shown == 0
    assert len(fm.text) == _framing_overhead(105)


def test_page_text_is_emitted_verbatim_without_rstrip():
    """The formula rests on this, so it gets its own test.

    Block text normally ends in a newline, so a page's segment must appear
    between its markers character for character -- no rstrip anywhere.
    """
    doc = _doc_with_lines(2)
    page_one = _extract_page_text(doc[0])
    fm = build_front_matter(doc)
    doc.close()

    assert f"--- PAGE 1 ---\n{page_one}\n--- PAGE 2 ---" in fm.text
    assert len(fm.text) - fm.chars_shown == _framing_overhead(2)


def test_max_pages_larger_than_document_reads_what_exists():
    doc = _doc_with_lines(1)
    fm = build_front_matter(doc, max_pages=9)
    doc.close()

    assert fm.pages_read == 1
    assert fm.total_pages == 1
    assert fm.pages_omitted == 0
    assert len(fm.pages) == 1
    assert fm.pages[0].page == 1
    assert fm.pages[0].chars == fm.chars_extracted


def test_max_chars_bounds_document_text_not_the_string():
    doc = _doc_with_lines(2, lines=45, width=80)
    fm = build_front_matter(doc, max_chars=200)
    doc.close()

    assert fm.chars_shown <= 200
    assert fm.char_truncated is True
    # The returned string legitimately exceeds the cap: markers are framing.
    assert len(fm.text) > fm.chars_shown


def test_generate_preamble_wraps_build_front_matter():
    doc = _doc_with_lines(1)
    text = generate_preamble(doc)
    expected = build_front_matter(doc).text
    doc.close()

    assert text == expected
    assert DEFAULT_MAX_CHARS == 5000


def test_pages_entries_cover_every_page_read_even_when_truncated():
    doc = _doc_with_lines(2, lines=45, width=80)
    fm = build_front_matter(doc, max_chars=200)
    doc.close()

    assert [p.page for p in fm.pages] == [1, 2]
    assert fm.chars_extracted == sum(p.chars for p in fm.pages)


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


def test_respects_max_chars_on_document_text():
    """max_chars bounds document text; markers and notes are framing.

    Rewritten in 0.26.0: this asserted len(preamble) <= max_chars on the whole
    returned string, a guarantee page markers deliberately give up.
    """
    doc = _doc_with_lines(2, lines=45, width=80)
    fm = build_front_matter(doc, max_chars=200)
    doc.close()
    assert fm.chars_shown <= 200


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
def test_real_pdf_respects_max_chars_on_document_text():
    """Real PDF: max_chars bounds document text, not the framed string."""
    if not TLE9350_PATH.exists():
        pytest.skip("Test PDF not found")
    doc = pymupdf.open(str(TLE9350_PATH))
    fm = build_front_matter(doc, max_chars=500)
    doc.close()
    assert fm.chars_shown <= 500
