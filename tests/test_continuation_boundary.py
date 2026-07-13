"""Tests for the page-boundary continuation signal.

This is a different concept from `continued_tables` (see test_continued_tables.py):
it answers "does content continue from page N onto page N+1?" for an arbitrary
page range, rather than "which tables in this section are captioned Continued".
"""

from datasheetindex.core.structure import continuation_at_boundary


def _make_text(*page_texts: str) -> str:
    """Build a text_content string from per-page text snippets."""
    parts: list[str] = []
    for i, text in enumerate(page_texts, start=1):
        parts.append(f"--- PAGE {i} ---")
        parts.append(text)
    return "\n".join(parts)


# Running headers that precede the marker on a real page, as PyMuPDF emits them.
TI_PAGE_5 = (
    "TCAN1044A-Q1\n"
    "SLLSFJ3D - AUGUST 2023\n"
    "www.ti.com\n"
    "6.4 Recommended Operating Conditions (continued)\n"
    "MIN NOM MAX UNIT\n"
    "TJ Operating junction temperature -40\n"
)
INFINEON_PAGE_21 = (
    "TLE9350BSJ\n"
    "High speed CAN FD transceiver\n"
    "Table 9\n"
    "Electrical characteristics transmitter (Continued)\n"
    "Vdiff_slope_rd 42 70 V/us\n"
)


def test_ti_style_marker_fires():
    text = _make_text(
        "6.4 Recommended Operating Conditions\nVCC 4.5 5 5.5 V", TI_PAGE_5
    )
    assert continuation_at_boundary(text, 1) == ["6.4 Recommended Operating Conditions"]


def test_infineon_style_marker_fires():
    first = "Table 9\nElectrical characteristics transmitter"
    text = _make_text(first, INFINEON_PAGE_21)
    assert continuation_at_boundary(text, 1) == [
        "Electrical characteristics transmitter"
    ]


def test_table_n_single_line_marker_fires():
    text = _make_text(
        "Table 1 Electrical Specs\ndata",
        "Table 1 Electrical Specs (Continued)\nmore",
    )
    assert continuation_at_boundary(text, 1) == ["Table 1 Electrical Specs"]


def test_cont_abbreviation_fires():
    text = _make_text("Table 2 Timing\ndata", "Table 2 Timing (Cont.)\nmore")
    assert continuation_at_boundary(text, 1) == ["Table 2 Timing"]


def test_marker_below_the_opening_block_is_ignored():
    """The NOTES: shape -- a mid-page (continued) block on a drawing page."""
    body = "\n".join(f"dimension line {i}" for i in range(20))
    text = _make_text("page one", f"{body}\nNOTES: (continued)\nmore notes")
    assert continuation_at_boundary(text, 1) == []


def test_no_marker_is_silent():
    text = _make_text("page one", "page two with no marker")
    assert continuation_at_boundary(text, 1) == []


def test_last_page_is_silent():
    text = _make_text("page one", "page two")
    assert continuation_at_boundary(text, 2) == []


def test_out_of_range_page_is_silent():
    """page < 1 makes the head-cut probe at start_page == 1 well-defined."""
    text = _make_text("Table 1 Specs (Continued)\ndata", "page two")
    assert continuation_at_boundary(text, 0) == []
    assert continuation_at_boundary(text, -3) == []


def test_marker_at_last_accepted_line_fires():
    """The positional guard's upper edge, pinned by construction rather than
    read off _OPENING_BLOCK_LINES: a marker as the 5th nonblank line (index 4,
    the last one included by nonblank[:5]) must fire. If _OPENING_BLOCK_LINES
    is ever changed, this literal line count no longer matches the guard's
    edge and this test starts failing -- which is the point.
    """
    filler = "\n".join(f"header line {i}" for i in range(4))
    text = _make_text("page one", f"{filler}\nTable 1 Specs (Continued)\nmore")
    assert continuation_at_boundary(text, 1) == ["Table 1 Specs"]


def test_marker_one_line_past_the_last_accepted_line_is_silent():
    """The same marker, pushed one nonblank line lower (the 6th, index 5) --
    just outside nonblank[:5] -- must be silent. Paired with the test above,
    this pins both edges of the positional guard.
    """
    filler = "\n".join(f"header line {i}" for i in range(5))
    text = _make_text("page one", f"{filler}\nTable 1 Specs (Continued)\nmore")
    assert continuation_at_boundary(text, 1) == []


def test_realistic_long_caption_is_detected():
    """Vendors commonly repeat the full parameterised caption on the
    continuation page. A 92-char title must not be silently dropped by the
    regex's upper length bound.
    """
    caption = (
        "Table 12. Electrical characteristics (VDD = 3.3 V, TA = 25 degC, "
        "unless otherwise specified)"
    )
    assert len(caption) == 92
    text = _make_text("page one", f"{caption} (continued)\nmore")
    assert continuation_at_boundary(text, 1) == [caption]


def test_paragraph_length_line_is_not_treated_as_a_title():
    """The length bound still exists to reject prose, not titles: a line far
    longer than any real caption, ending in "(continued)", must not match.
    """
    prose = "This is a long sentence of ordinary prose, not a table caption, " * 4
    assert len(prose) > 200
    text = _make_text("page one", f"{prose} (continued)\nmore")
    assert continuation_at_boundary(text, 1) == []


def test_multiple_markers_deduplicated_in_order():
    second = (
        "Table 1 Specs (Continued)\n"
        "Table 2 Timing (Continued)\n"
        "Table 1 Specs (Continued)\n"
    )
    text = _make_text("Table 1 Specs\nTable 2 Timing", second)
    assert continuation_at_boundary(text, 1) == ["Table 1 Specs", "Table 2 Timing"]
