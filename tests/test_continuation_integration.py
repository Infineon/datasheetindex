"""Real-PDF integration tests for the page-cut truncation signal.

The synthetic tests in `test_continuation_boundary.py` and `test_registry.py`
generate their own short pages, so they cannot catch the one failure mode that
only exists in real documents: the positional guard depends on where PyMuPDF
puts the vendor's running header. `_OPENING_BLOCK_LINES = 5` works because a real
continuation marker lands at nonblank line 3, beneath a 2-3 line running header.
If a PyMuPDF upgrade changed text extraction enough to push the marker past the
opening block, detection would silently stop -- and every synthetic test would
still pass.

These tests pin that against a real datasheet. They skip when the PDF is absent,
following the convention in `conftest.py`.

The TLE9350BSJ is also the *silent* case by design: both of its continued tables
sit entirely inside their ToC section (7.4 spans pages 20-21 and the cut is at
20->21), so a whole-section read must produce no note at all. That makes this one
document exercise both directions -- fires on the cut, silent when the range
covers the table.
"""

from __future__ import annotations

from pathlib import Path

from datasheetindex.tools.bound import DatasheetTools

NOTE_PREFIX = "=== NOTE:"


def _notes(section_text: str) -> list[str]:
    return [line for line in section_text.splitlines() if line.startswith(NOTE_PREFIX)]


def test_real_pdf_tail_cut_is_detected(pdf_tle9350_path: Path, tmp_path):
    """Page 20 alone cuts Table 9, which continues on page 21.

    This is the test that pins the positional guard against real PyMuPDF output:
    it only passes if the "(Continued)" marker on page 21 lands inside the
    opening block, under the vendor's running header.
    """
    with DatasheetTools(str(pdf_tle9350_path)) as tools:
        tools.build_datasheet(output_dir=str(tmp_path))
        notes = _notes(tools.get_section_text(20, 20))

    assert len(notes) == 1, notes
    assert "Electrical characteristics transmitter" in notes[0]
    assert "is continued on page 21" in notes[0]


def test_real_pdf_head_cut_is_detected(pdf_tle9350_path: Path, tmp_path):
    """Page 21 alone opens inside Table 9, which began on page 20."""
    with DatasheetTools(str(pdf_tle9350_path)) as tools:
        tools.build_datasheet(output_dir=str(tmp_path))
        notes = _notes(tools.get_section_text(21, 21))

    assert len(notes) == 1, notes
    assert "opens inside" in notes[0]
    assert "is continued from page 20" in notes[0]


def test_real_pdf_whole_section_read_is_silent(pdf_tle9350_path: Path, tmp_path):
    """Section 7.4 spans pages 20-21 and contains the whole of Table 9.

    The signal is range-relative, so a read that covers the table must say
    nothing. A note here would be a false positive on the most common agent
    access pattern -- reading a whole ToC section.
    """
    with DatasheetTools(str(pdf_tle9350_path)) as tools:
        tools.build_datasheet(output_dir=str(tmp_path))
        section_text = tools.get_section_text(20, 21)

    assert _notes(section_text) == []


def test_real_pdf_ordinary_pages_are_silent(pdf_tle9350_path: Path, tmp_path):
    """No continuation anywhere else in the document.

    Guards the false-positive direction on a real document: only the two known
    cuts (20->21, 22->23) may warn. If a future change to the matcher starts
    firing on running headers, footers, or ordinary prose, this catches it.
    """
    cut_pages = {20, 21, 22, 23}
    with DatasheetTools(str(pdf_tle9350_path)) as tools:
        artifacts = tools.build_datasheet(output_dir=str(tmp_path))
        total = artifacts.json_data["total_pages"]
        warned = {
            page
            for page in range(1, total + 1)
            if _notes(tools.get_section_text(page, page))
        }

    assert warned == cut_pages
