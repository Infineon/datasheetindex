"""Tests for preamble extraction."""

from pathlib import Path

import pymupdf
import pytest

from datasheetindex.core.preamble import (
    DEFAULT_MAX_CHARS,
    _page_signals,
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


def test_char_truncation_note_carries_the_exact_counts():
    doc = _doc_with_lines(2, lines=45, width=80)
    fm = build_front_matter(doc, max_chars=200)
    doc.close()

    expected = (
        f"=== NOTE: preamble truncated at 200 characters; "
        f"{fm.chars_shown} of {fm.chars_extracted} characters from "
        f"pages 1-2 shown, ending mid-page on page 1 ==="
    )
    assert fm.text.endswith(expected)


def test_page_note_names_pages_read_and_total():
    doc = _doc_with_lines(4)
    fm = build_front_matter(doc)
    doc.close()

    assert fm.text.endswith(
        "=== NOTE: preamble covers pages 1-2 of 4; later pages were not examined ==="
    )
    # The caps are independent: the text fit, so no character note.
    assert "truncated at" not in fm.text


def test_both_notes_appear_with_the_character_note_first():
    doc = _doc_with_lines(4, lines=45, width=80)
    fm = build_front_matter(doc, max_chars=200)
    doc.close()

    notes = [ln for ln in fm.text.splitlines() if ln.startswith("=== NOTE:")]
    assert len(notes) == 2
    assert "truncated at 200 characters" in notes[0]
    assert "later pages were not examined" in notes[1]


def test_page_boundary_cut_omits_the_mid_page_clause():
    """A cap that lands exactly on a page boundary cut no page in half."""
    doc = _doc_with_lines(2, lines=4, width=40)
    page_one_chars = build_front_matter(doc, max_pages=1).chars_extracted
    fm = build_front_matter(doc, max_chars=page_one_chars)
    doc.close()

    assert fm.char_truncated is True
    assert "ending mid-page" not in fm.text
    assert fm.text.count("--- PAGE ") == 1


def test_a_page_that_does_not_fit_at_all_gets_no_marker():
    """An exhausted budget must not claim a page is empty.

    ``--- PAGE 2 ---`` with nothing after it reads as "page 2 holds no text",
    which is a claim about document content the budget does not license.
    """
    doc = _doc_with_lines(2, lines=4, width=40)
    page_one_chars = build_front_matter(doc, max_pages=1).chars_extracted
    fm = build_front_matter(doc, max_chars=page_one_chars + 5)
    doc.close()

    assert fm.char_truncated is True
    assert fm.text.count("--- PAGE ") == 1
    assert "--- PAGE 2 ---" not in fm.text
    assert "ending mid-page" not in fm.text


def test_a_cut_just_past_a_page_boundary_omits_the_mid_page_clause():
    """The boundary case away from the measure-zero exact-fit value.

    ``max_chars`` a few characters past page 1 takes a different branch than
    the exact fit tested above: page 2's first line does not fit, so the cut
    still lands on the page boundary and no page was ended mid-way.
    """
    doc = _doc_with_lines(2, lines=4, width=40)
    page_one_chars = build_front_matter(doc, max_pages=1).chars_extracted
    page_two_first_line = _extract_page_text(doc[1]).splitlines(keepends=True)[0]
    fm = build_front_matter(doc, max_chars=page_one_chars + 5)
    doc.close()

    assert 0 < 5 < len(page_two_first_line)
    assert fm.char_truncated is True
    assert fm.text.count("--- PAGE ") == 1
    assert "ending mid-page" not in fm.text


def test_max_chars_zero_emits_the_note_and_no_marker():
    """Nothing fits, so nothing is claimed -- and no stray leading newline."""
    doc = _doc_with_lines(2, lines=4, width=40)
    fm = build_front_matter(doc, max_chars=0)
    doc.close()

    assert "--- PAGE " not in fm.text
    assert not fm.text.startswith("\n")
    assert fm.chars_shown == 0
    assert fm.char_truncated is True
    assert "ending mid-page" not in fm.text
    assert fm.text == (
        "=== NOTE: preamble truncated at 0 characters; "
        f"0 of {fm.chars_extracted} characters from pages 1-2 shown ==="
    )


def test_negative_max_chars_raises():
    """A negative cap would otherwise be quoted verbatim in the NOTE line."""
    doc = _doc_with_lines(2)
    with pytest.raises(ValueError, match="max_chars"):
        build_front_matter(doc, max_chars=-50)
    doc.close()


def test_single_page_document_note_uses_singular_page_phrase():
    doc = _doc_with_lines(1, lines=45, width=80)
    fm = build_front_matter(doc, max_chars=200)
    doc.close()

    assert "characters from page 1 shown" in fm.text
    assert "pages 1-1" not in fm.text


def test_notes_are_framing_and_excluded_from_chars_shown():
    doc = _doc_with_lines(4, lines=45, width=80)
    fm = build_front_matter(doc, max_chars=200)
    doc.close()

    assert fm.chars_shown <= 200
    assert len(fm.text) > 200


def test_max_pages_zero_raises():
    doc = _doc_with_lines(3)
    with pytest.raises(ValueError, match="max_pages"):
        build_front_matter(doc, max_pages=0)
    doc.close()


def test_max_pages_bool_raises():
    doc = _doc_with_lines(3)
    with pytest.raises(ValueError, match="max_pages"):
        # bool is a subclass of int -- True would silently become a cap
        # of 1 if this were not rejected explicitly.
        build_front_matter(doc, max_pages=True)
    doc.close()


def test_signals_on_a_bulleted_features_page():
    """Glyphs go in as escapes -- the project bans literal Unicode in tests."""
    text = (
        "CY8C62x8\n"
        "General description\n"
        "The PSoC 6 MCU is a dual-core device.\n"
        "Features\n"
        "\u2022 32-bit Arm Cortex-M4F CPU at 150 MHz\n"
        "\u2022 2 MByte flash and 1 MByte SRAM\n"
        "- up to 102 programmable GPIOs\n"
        "\u25aa 12-bit 2-Msps SAR ADC\n"
    )
    signals = _page_signals(text)

    assert signals["bullets"] == 4
    assert signals["has_features_heading"] is True


def test_a_cover_letter_scores_no_bullets_and_no_features_heading():
    """A cover letter shows itself by what it lacks, not by what it says.

    The disclaimer prose is kept deliberately: it is realistic cover-letter
    text, and pinning that it yields no bullets and no heading is the whole
    claim now that no signal scores its vocabulary.
    """
    text = (
        "Product Change Notification\n"
        "TI requires acknowledgement of receipt of this notification "
        "within 30 days.\n"
        "TI makes no warranty and accepts no liability; see the trademark "
        "and copyright notices.\n"
    )
    signals = _page_signals(text)

    assert signals["bullets"] == 0
    assert signals["has_features_heading"] is False


def test_a_leading_hyphen_needs_whitespace_to_count_as_a_bullet():
    """Datasheets are full of temperature ranges; those are not bullets."""
    assert _page_signals("-40 to +85 degrees C\n")["bullets"] == 0
    assert _page_signals("- a real bullet\n")["bullets"] == 1


def test_a_lone_dash_on_its_own_line_counts_as_a_bullet():
    """Infineon's extraction puts most of its markers alone on a line.

    End of line satisfies the load-bearing requirement -- the dash is not
    immediately followed by a digit or a letter -- so it counts, while the
    ranges the whitespace rule exists for still do not.
    """
    text = "Features\n-\n32-bit CPU at 150 MHz\n-\n2 MByte flash\n"
    assert _page_signals(text)["bullets"] == 2
    assert _page_signals("-40 to +85 degrees C\n")["bullets"] == 0
    assert _page_signals("-1.5 V minimum\n")["bullets"] == 0
    assert _page_signals("\u201340 C\n")["bullets"] == 0


def test_features_heading_matches_a_whole_line_only():
    assert _page_signals("Features\n")["has_features_heading"] is True
    assert _page_signals("General Description:\n")["has_features_heading"] is True
    assert _page_signals("General description\n")["has_features_heading"] is True
    assert (
        _page_signals("Features of the analog subsystem\n")["has_features_heading"]
        is False
    )


@pytest.mark.parametrize(
    "line,expected",
    [
        # TI numbers its front-matter headings; all seven TI datasheets in the
        # corpus write "1 Features", and the whole-line comparison rejected
        # every one of them.
        ("1 Features", True),
        ("2 Features", True),
        ("1.1 Features", True),
        ("Features", True),
        ("1 General Description", True),
        # Nexperia's form, verbatim from the 74HC595: a period and two spaces.
        ("1.  General description", True),
        # A numbered heading whose title is not a heading is still not one.
        ("Features of the analog subsystem", False),
        ("5 Features of the analog subsystem", False),
    ],
)
def test_features_heading_ignores_a_leading_section_number(line, expected):
    assert _page_signals(line + "\n")["has_features_heading"] is expected


def test_build_front_matter_populates_signals_per_page():
    """Page 1 carries both signals; page 2 is prose that carries neither.

    The page-2 disclaimer sentence is now just prose -- nothing counts its
    vocabulary -- and it stays because a page with neither signal is what the
    per-page assertions need.
    """
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(
        pymupdf.Rect(50, 50, 500, 300),
        "Features\n- first feature\n- second feature",
        fontsize=10,
    )
    page = doc.new_page()
    page.insert_textbox(
        pymupdf.Rect(50, 50, 500, 300),
        "We disclaim all warranty and liability.",
        fontsize=10,
    )
    fm = build_front_matter(doc)
    doc.close()

    assert fm.pages[0].bullets == 2
    assert fm.pages[0].has_features_heading is True
    assert fm.pages[1].bullets == 0
    assert fm.pages[1].has_features_heading is False


def test_signals_reflect_the_whole_page_even_when_truncated():
    """Signals describe the page read, not the fragment shown."""
    doc = pymupdf.open()
    for _ in range(2):
        page = doc.new_page()
        writer = pymupdf.TextWriter(page.rect)
        y = 72
        for _ in range(40):
            writer.append((72, y), "- " + "A" * 60)
            y += 14
        writer.write_text(page)
    fm = build_front_matter(doc, max_chars=100)
    doc.close()

    assert fm.char_truncated is True
    assert fm.pages[1].bullets > 0


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
