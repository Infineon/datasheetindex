"""Tests for text file generation."""

import re
from pathlib import Path
from typing import cast

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


def test_search_text_multi_pattern_tags_matches():
    text = "--- PAGE 1 ---\nAlpha beta\n--- PAGE 2 ---\nGamma alpha delta\n"

    matches = search_text(text, ["beta", "gamma"])

    assert matches == [
        {"page": 1, "start": 6, "end": 10, "snippet": "Alpha beta", "pattern": "beta"},
        {
            "page": 2,
            "start": 0,
            "end": 5,
            "snippet": "Gamma alpha delta",
            "pattern": "gamma",
        },
    ]


def test_search_text_multi_pattern_dedups_overlapping_spans():
    text = "--- PAGE 1 ---\nAlpha beta\n"

    # Both patterns hit the same span; the first pattern wins and it appears once.
    matches = search_text(text, ["alpha", "alpha"])

    assert len(matches) == 1
    assert matches[0]["pattern"] == "alpha"


def test_search_text_multi_pattern_collects_unique_across_patterns():
    text = "--- PAGE 1 ---\nAlpha beta\n--- PAGE 2 ---\nAlpha gamma\n"

    # Each pattern is searched up to the full cap, so a later pattern's unique
    # hits are returned alongside an earlier pattern's (deduped by span).
    matches = search_text(text, ["alpha", "gamma"], max_results=10)

    tagged = {(m["page"], m["pattern"]) for m in matches}
    assert tagged == {(1, "alpha"), (2, "alpha"), (2, "gamma")}


def test_search_text_multi_pattern_max_results_is_global_cap():
    text = "--- PAGE 1 ---\nAlpha beta\n--- PAGE 2 ---\nGamma alpha delta\n"

    matches = search_text(text, ["alpha", "gamma"], max_results=1)

    assert len(matches) == 1


def test_search_text_multi_pattern_empty_raises():
    text = "--- PAGE 1 ---\nAlpha beta\n"

    with pytest.raises(ValueError, match="query must not be empty"):
        search_text(text, [])
    with pytest.raises(ValueError, match="query must not be empty"):
        search_text(text, ["", "  "])


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
def test_real_pdf_no_false_column_detection(monkeypatch):
    """Column detection should not trigger on single-column TLE9350 datasheet.

    Furniture stripping is disabled here: this test's baseline is raw
    ``get_text(sort=True)``, which includes running headers/footers, so
    comparing it against furniture-stripped output would fail for a reason
    unrelated to column detection -- the real TLE9350 datasheet does carry a
    running header, and scan_pages now legitimately removes it.
    """
    if not TLE9350_PATH.exists():
        pytest.skip("Test PDF not found")
    monkeypatch.setenv("DATASHEETINDEX_FURNITURE", "0")
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


def test_generate_text_delegates_to_scan_pages():
    # The retained wrapper must not drift from the function it delegates to.
    from datasheetindex.core.textfile import generate_text, scan_pages

    doc = pymupdf.open()
    page = doc.new_page()
    writer = pymupdf.TextWriter(page.rect)
    writer.append((72, 72), "Supply voltage")
    writer.write_text(page)

    assert generate_text(doc) == scan_pages(doc).text
    doc.close()


def test_scan_pages_collects_figures_across_pages():
    from datasheetindex.core.textfile import scan_pages

    doc = pymupdf.open()
    first = doc.new_page(width=595, height=842)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20))
    pix.set_rect(pix.irect, (0, 255, 0))
    first.insert_image(pymupdf.Rect(100, 100, 400, 400), pixmap=pix)
    second = doc.new_page(width=595, height=842)
    writer = pymupdf.TextWriter(second.rect)
    writer.append((72, 72), "Figure 4. Timing diagram")
    writer.write_text(second)

    scan = scan_pages(doc)
    doc.close()

    kinds = [(entry["page"], entry["kind"]) for entry in scan.figures]
    assert (1, "raster") in kinds
    assert (2, "caption") in kinds
    assert scan.excluded_below_min_area == 0


def test_scan_pages_orders_figures_by_page():
    from datasheetindex.core.textfile import scan_pages

    doc = pymupdf.open()
    for _ in range(3):
        page = doc.new_page(width=595, height=842)
        writer = pymupdf.TextWriter(page.rect)
        writer.append((72, 72), "Figure 1. Something")
        writer.write_text(page)

    pages = [cast(int, entry["page"]) for entry in scan_pages(doc).figures]
    doc.close()

    assert pages == sorted(pages)


def test_ordered_blocks_joined_equals_extract_page_text(tmp_path):
    """_extract_page_text must stay a plain join over _ordered_blocks.

    Characterization test for the Task 2 refactor: it pins the join
    equivalence and the empty-page path -- both sides call _ordered_blocks
    and index the same field, so this holds for any list _ordered_blocks
    returns and does not pin reading order or column partitioning by
    itself. It still has real value: it fails if stripping or filtering
    is later added to _extract_page_text, which core/preamble.py depends
    on staying unstripped. See
    test_ordered_blocks_pins_column_partitioning_order below for a test
    that pins reading order with a concrete expected value.
    """
    import pymupdf

    from datasheetindex.core.textfile import _extract_page_text, _ordered_blocks

    pdf = tmp_path / "two-column.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    # Two columns plus a full-width heading above them.
    page.insert_text((50, 60), "Wide heading across the page", fontsize=11)
    for row in range(6):
        page.insert_text((50, 120 + row * 14), f"left line {row}", fontsize=9)
        page.insert_text((320, 120 + row * 14), f"right line {row}", fontsize=9)
    doc.new_page()  # deliberately empty: the no-blocks path
    doc.save(str(pdf))
    doc.close()

    doc = pymupdf.open(str(pdf))
    try:
        for page in doc:
            joined = "\n".join(b[4] for b in _ordered_blocks(page))
            assert joined == _extract_page_text(page)
    finally:
        doc.close()


def test_ordered_blocks_pins_column_partitioning_order():
    """A concrete expected order that breaks if column partitioning changes.

    Builds a two-column page with three stacked blocks per column. Each
    right-column block starts 20pt higher on the page than its same-row
    left-column counterpart, so a naive top-to-bottom sort across all six
    blocks would interleave them (R0, L0, R1, L1, R2, L2). The correct
    column-aware order groups the whole left column before the whole right
    column. Unlike test_ordered_blocks_joined_equals_extract_page_text
    above, this asserts a literal expected sequence, so it fails if the
    above/left/right/below partitioning in _ordered_blocks is broken (for
    example if left_col and right_col were swapped).
    """
    from datasheetindex.core.textfile import (
        _BLOCK_TYPE,
        _detect_columns,
        _ordered_blocks,
    )

    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)

    def place(x, y_top, name):
        page.insert_text((x, y_top + 24), name + " alpha bravo charlie", fontsize=20)
        page.insert_text((x, y_top + 52), name + " delta echo foxtrot", fontsize=20)
        page.insert_text((x, y_top + 80), name + " golf hotel india", fontsize=20)

    left_x, right_x = 50, 320
    left_tops = [100, 250, 400]
    right_tops = [80, 230, 380]  # each 20pt higher than its left counterpart

    for i, top in enumerate(left_tops):
        place(left_x, top, f"L{i}")
    for i, top in enumerate(right_tops):
        place(right_x, top, f"R{i}")

    # Column detection must actually fire here, or the rest of this test
    # would degenerate into asserting plain top-to-bottom order and prove
    # nothing about partitioning. Column widths (~197-200pt of a 612pt
    # page), the ~73pt gutter, and the ~83.5pt block height are chosen to
    # satisfy _detect_columns' width, gutter, and high-confidence height
    # thresholds.
    raw_blocks = page.get_text("blocks")
    text_blocks = [b for b in raw_blocks if b[_BLOCK_TYPE] == 0]
    assert _detect_columns(text_blocks, page.rect.width) is not None

    ordered_texts = [b[4] for b in _ordered_blocks(page)]
    doc.close()

    assert ordered_texts == [
        "L0 alpha bravo charlie\nL0 delta echo foxtrot\nL0 golf hotel india\n",
        "L1 alpha bravo charlie\nL1 delta echo foxtrot\nL1 golf hotel india\n",
        "L2 alpha bravo charlie\nL2 delta echo foxtrot\nL2 golf hotel india\n",
        "R0 alpha bravo charlie\nR0 delta echo foxtrot\nR0 golf hotel india\n",
        "R1 alpha bravo charlie\nR1 delta echo foxtrot\nR1 golf hotel india\n",
        "R2 alpha bravo charlie\nR2 delta echo foxtrot\nR2 golf hotel india\n",
    ]


def test_extract_page_blocks_tags_the_top_and_bottom_bands(tmp_path):
    """A block counts as banded only if it lies wholly inside an edge band."""
    import pymupdf

    from datasheetindex.core.textfile import _extract_page_blocks

    pdf = tmp_path / "bands.pdf"
    doc = pymupdf.open()
    page = doc.new_page()  # default letter page: 792pt tall
    height = page.rect.height
    page.insert_text((50, height * 0.05), "running header", fontsize=9)
    page.insert_text((50, height * 0.50), "body text in the middle", fontsize=9)
    page.insert_text((50, height * 0.96), "running footer", fontsize=9)
    doc.save(str(pdf))
    doc.close()

    doc = pymupdf.open(str(pdf))
    try:
        blocks = _extract_page_blocks(doc[0])
    finally:
        doc.close()

    banded = {text.strip(): flag for text, flag in blocks}
    assert banded["running header"] is True
    assert banded["running footer"] is True
    assert banded["body text in the middle"] is False


def test_extract_page_blocks_preserves_reading_order(tmp_path):
    """The pairs must arrive in the same order _extract_page_text joins them."""
    import pymupdf

    from datasheetindex.core.textfile import _extract_page_blocks, _extract_page_text

    pdf = tmp_path / "order.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    for row in range(5):
        page.insert_text((50, 100 + row * 20), f"line {row}", fontsize=9)
    doc.save(str(pdf))
    doc.close()

    doc = pymupdf.open(str(pdf))
    try:
        page = doc[0]
        joined = "\n".join(text for text, _ in _extract_page_blocks(page))
        assert joined == _extract_page_text(page)
    finally:
        doc.close()


def test_furniture_enabled_by_env_accepts_the_spellings_people_reach_for(
    monkeypatch,
):
    """Mirrors _parallel_enabled_by_env: matching only "0" would silently
    ignore =false and leave the escape hatch looking broken."""
    from datasheetindex.core.textfile import _furniture_enabled_by_env

    monkeypatch.delenv("DATASHEETINDEX_FURNITURE", raising=False)
    assert _furniture_enabled_by_env() is True

    for off in ("0", "false", "FALSE", "no", "off", "  Off  "):
        monkeypatch.setenv("DATASHEETINDEX_FURNITURE", off)
        assert _furniture_enabled_by_env() is False, off

    for on in ("1", "true", "yes", "anything else"):
        monkeypatch.setenv("DATASHEETINDEX_FURNITURE", on)
        assert _furniture_enabled_by_env() is True, on


def _furniture_pdf(path, pages, *, header=True, footer=True, caption=False):
    """A document with an optional running header, footer and top caption."""
    import pymupdf

    doc = pymupdf.open()
    for p in range(pages):
        page = doc.new_page()
        height = page.rect.height
        if header:
            page.insert_text(
                (50, height * 0.05), "ACME AWC-3200 Controller", fontsize=9
            )
        if caption:
            # A caption high on the page: banded, recurring, must survive.
            page.insert_text(
                (50, height * 0.12), f"Table {p + 1} Pin assignments", fontsize=9
            )
        page.insert_text(
            (50, height * 0.45), f"Body sentence unique to page {p + 1}.", fontsize=9
        )
        if footer:
            page.insert_text((50, height * 0.94), "Datasheet", fontsize=8)
            page.insert_text((50, height * 0.96), f"{p + 1}", fontsize=8)
            page.insert_text((50, height * 0.98), "AWC-3200 Rev. B", fontsize=8)
    doc.save(str(path))
    doc.close()


def test_scan_pages_drops_the_running_header_and_footer(tmp_path):
    import pymupdf

    from datasheetindex.core.textfile import scan_pages

    pdf = tmp_path / "furniture.pdf"
    _furniture_pdf(pdf, pages=8)
    doc = pymupdf.open(str(pdf))
    try:
        text = scan_pages(doc).text
    finally:
        doc.close()

    assert "ACME AWC-3200 Controller" not in text
    assert "AWC-3200 Rev. B" not in text
    # Body survives, and every page marker is still emitted.
    assert "Body sentence unique to page 4." in text
    for p in range(1, 9):
        assert f"--- PAGE {p} ---" in text


def test_scan_pages_keeps_a_recurring_table_caption(tmp_path):
    """The caption guard.

    A 'Table N' caption placed high on the page recurs on every page and
    normalizes to the same key as its neighbours. It must survive: these are
    the captions TocNode.continued_tables is built from, and the line-level
    approach this design replaced would have deleted them.
    """
    import pymupdf

    from datasheetindex.core.textfile import scan_pages

    pdf = tmp_path / "captions.pdf"
    _furniture_pdf(pdf, pages=8, caption=True)
    doc = pymupdf.open(str(pdf))
    try:
        text = scan_pages(doc).text
    finally:
        doc.close()

    assert "ACME AWC-3200 Controller" not in text  # furniture still goes
    for p in range(1, 9):
        assert f"Table {p} Pin assignments" in text  # captions all stay


def test_scan_pages_strips_nothing_from_a_two_page_document(tmp_path):
    """Below the MIN_PAGES floor there is no recurrence evidence."""
    import pymupdf

    from datasheetindex.core.textfile import scan_pages

    pdf = tmp_path / "short.pdf"
    _furniture_pdf(pdf, pages=2)
    doc = pymupdf.open(str(pdf))
    try:
        text = scan_pages(doc).text
    finally:
        doc.close()

    assert text.count("ACME AWC-3200 Controller") == 2


def test_scan_pages_leaves_a_furniture_free_document_unchanged(tmp_path):
    """No running furniture means byte-identical output."""
    import pymupdf

    from datasheetindex.core.textfile import _extract_page_text, scan_pages

    pdf = tmp_path / "plain.pdf"
    _furniture_pdf(pdf, pages=6, header=False, footer=False)
    doc = pymupdf.open(str(pdf))
    try:
        expected = "\n".join(
            part
            for i in range(len(doc))
            for part in (f"--- PAGE {i + 1} ---", _extract_page_text(doc[i]))
        )
        assert scan_pages(doc).text == expected
    finally:
        doc.close()


def test_scan_pages_escape_hatch_restores_the_unstripped_text(tmp_path, monkeypatch):
    import pymupdf

    from datasheetindex.core.textfile import _extract_page_text, scan_pages

    pdf = tmp_path / "hatch.pdf"
    _furniture_pdf(pdf, pages=8)
    monkeypatch.setenv("DATASHEETINDEX_FURNITURE", "0")
    doc = pymupdf.open(str(pdf))
    try:
        expected = "\n".join(
            part
            for i in range(len(doc))
            for part in (f"--- PAGE {i + 1} ---", _extract_page_text(doc[i]))
        )
        assert scan_pages(doc).text == expected
    finally:
        doc.close()


def test_preamble_still_sees_the_running_furniture(tmp_path):
    """The preamble's "raw text, zero heuristics" contract.

    preamble.py reads _extract_page_text, not scan_pages, so stripping must
    not reach it. This is the test that fails if someone "simplifies" the
    design by moving the strip into _extract_page_text -- which would look
    like a tidy-up and would silently break a documented guarantee.
    """
    import pymupdf

    from datasheetindex.core.preamble import generate_preamble
    from datasheetindex.core.textfile import scan_pages

    pdf = tmp_path / "preamble.pdf"
    _furniture_pdf(pdf, pages=8)
    doc = pymupdf.open(str(pdf))
    try:
        preamble = generate_preamble(doc)
        stripped = scan_pages(doc).text
    finally:
        doc.close()

    assert "ACME AWC-3200 Controller" in preamble
    assert "ACME AWC-3200 Controller" not in stripped


def test_scan_pages_keeps_the_figure_index_interleaved_by_page(tmp_path):
    """Splitting into two passes must not reorder the figure index.

    Figures are appended per page as rasters-then-captions. A naive two-pass
    split emits all rasters before all captions, which silently reorders the
    index that build_datasheet publishes.
    """
    import pymupdf

    from datasheetindex.core.textfile import scan_pages

    pdf = tmp_path / "figs.pdf"
    doc = pymupdf.open()
    for p in range(4):
        page = doc.new_page()
        page.insert_text(
            (50, 300), f"Figure {p + 1}. Diagram for page {p + 1}", fontsize=9
        )
    doc.save(str(pdf))
    doc.close()

    doc = pymupdf.open(str(pdf))
    try:
        figures = scan_pages(doc).figures
    finally:
        doc.close()

    pages = [cast(int, f["page"]) for f in figures]
    assert pages == sorted(pages), f"figure index is not page-ordered: {pages}"


def test_is_banded_requires_the_whole_block_inside_the_band(tmp_path):
    """A block straddling the band boundary is not banded, only a block that
    lies wholly inside the top or bottom edge band is.

    test_extract_page_blocks_tags_the_top_and_bottom_bands places its blocks
    far from the 20%/80% boundaries, so it would still pass even if the two
    comparisons in _is_banded were swapped (testing the wrong edge of each
    block). That matters here because a too-permissive band makes furniture
    detection delete real body text. This test builds blocks that actually
    straddle each boundary and asserts on the real block geometry PyMuPDF
    returns, rather than assuming exact coordinates.
    """
    import pymupdf

    from datasheetindex.core.textfile import _is_banded, _ordered_blocks

    pdf = tmp_path / "straddle.pdf"
    doc = pymupdf.open()
    page = doc.new_page()  # default letter page: 792pt tall
    height = page.rect.height
    top_limit = height * 0.20
    bottom_limit = height * 0.80

    # Wholly inside the top band.
    page.insert_text((50, height * 0.03), "top inside", fontsize=8)

    # Straddles the top boundary: starts inside the top band, several lines
    # push the block's bottom edge (y1) well past top_limit.
    y = top_limit - 12
    for i in range(6):
        page.insert_text((300, y + i * 10), f"top straddle line {i}", fontsize=8)

    # Straddles the bottom boundary: starts above bottom_limit, several lines
    # push the block's bottom edge into the bottom band.
    y = bottom_limit - 30
    for i in range(6):
        page.insert_text((300, y + i * 10), f"bottom straddle line {i}", fontsize=8)

    # Wholly inside the bottom band.
    page.insert_text((50, height * 0.97), "bottom inside", fontsize=8)

    doc.save(str(pdf))
    doc.close()

    doc = pymupdf.open(str(pdf))
    try:
        page = doc[0]
        blocks = {
            b[textfile_module._BLOCK_TEXT].strip(): b for b in _ordered_blocks(page)
        }
        page_height = page.rect.height
    finally:
        doc.close()

    top_inside = blocks["top inside"]
    bottom_inside = blocks["bottom inside"]
    top_straddle = next(
        b for text, b in blocks.items() if text.startswith("top straddle")
    )
    bottom_straddle = next(
        b for text, b in blocks.items() if text.startswith("bottom straddle")
    )

    # Sanity: confirm the straddling blocks really do straddle the boundary
    # they are meant to test, rather than assuming exact PyMuPDF geometry.
    assert top_straddle[textfile_module._BLOCK_Y0] < top_limit
    assert top_straddle[textfile_module._BLOCK_Y1] > top_limit
    assert bottom_straddle[textfile_module._BLOCK_Y0] < bottom_limit
    assert bottom_straddle[textfile_module._BLOCK_Y1] > bottom_limit

    assert _is_banded(top_inside, page_height) is True
    assert _is_banded(bottom_inside, page_height) is True
    assert _is_banded(top_straddle, page_height) is False
    assert _is_banded(bottom_straddle, page_height) is False
