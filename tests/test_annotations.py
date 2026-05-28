"""Tests for footnote and cross-reference detection."""

from pathlib import Path

import pymupdf
import pytest

from datasheetindex.core.annotations import (
    enrich_with_cross_references,
    enrich_with_footnote_markers,
)
from datasheetindex.core.structure import build_tree, extract_toc
from datasheetindex.core.textfile import generate_text
from datasheetindex.models import TocNode

DATA2PAGE_DIR = Path(__file__).resolve().parent.parent.parent / "data2page"
TLE9350_PATH = DATA2PAGE_DIR / "Infineon-TLE9350BSJ-DataSheet-v01_00-EN.pdf"


def _make_text(*page_texts: str) -> str:
    parts: list[str] = []
    for i, text in enumerate(page_texts, start=1):
        parts.append(f"--- PAGE {i} ---")
        parts.append(text)
    return "\n".join(parts)


def _make_node(title="Section", start=1, end=1):
    return TocNode(title=title, level=1, start_page=start, end_page=end)


# ===================== Footnote tests =====================


def test_footnote_no_markers():
    text = _make_text("Just plain text with no footnotes.")
    node = _make_node()
    enrich_with_footnote_markers([node], text)
    assert node.footnote_markers == []


def test_footnote_bare_numeric():
    text = _make_text("Some value 1) measured at 25C. Another 2) at 85C.")
    node = _make_node()
    enrich_with_footnote_markers([node], text)
    assert "1)" in node.footnote_markers
    assert "2)" in node.footnote_markers


def test_footnote_note_style():
    text = _make_text("See Note 1 for details. Note 2 applies to all conditions.")
    node = _make_node()
    enrich_with_footnote_markers([node], text)
    assert "Note 1" in node.footnote_markers
    assert "Note 2" in node.footnote_markers


def test_footnote_parenthesized():
    text = _make_text("Rating (1) and (2) apply under certain conditions.")
    node = _make_node()
    enrich_with_footnote_markers([node], text)
    assert "(1)" in node.footnote_markers
    assert "(2)" in node.footnote_markers


def test_footnote_large_numbers_excluded():
    """Parenthesized numbers > 20 should be excluded (avoids year false positives)."""
    text = _make_text("Published in (2024) and reference (100) pages.")
    node = _make_node()
    enrich_with_footnote_markers([node], text)
    # Neither (2024) nor (100) should match
    assert "(2024)" not in node.footnote_markers
    assert "(100)" not in node.footnote_markers


def test_footnote_page_marker_not_matched():
    """PAGE markers should not produce false positives."""
    text = _make_text("Some text here.")
    node = _make_node()
    enrich_with_footnote_markers([node], text)
    # "--- PAGE 1 ---" contains "1" but should not be detected as footnote
    assert node.footnote_markers == []


def test_footnote_deduplication():
    text = _make_text("Value 1) here and again 1) there.")
    node = _make_node()
    enrich_with_footnote_markers([node], text)
    assert node.footnote_markers.count("1)") == 1


def test_footnote_nested_nodes():
    text = _make_text("Note 1 here.", "Value 2) there.")
    child = TocNode(title="Sub", level=2, start_page=2, end_page=2)
    parent = TocNode(title="Parent", level=1, start_page=1, end_page=2, nodes=[child])
    enrich_with_footnote_markers([parent], text)
    assert "Note 1" in parent.footnote_markers
    assert "2)" in parent.footnote_markers
    # Child only sees page 2
    assert "Note 1" not in child.footnote_markers
    assert "2)" in child.footnote_markers


# ===================== Cross-reference tests =====================


def test_crossref_figure():
    text = _make_text("See Figure 3 for the block diagram.")
    node = _make_node()
    enrich_with_cross_references([node], text)
    assert len(node.cross_references) == 1
    ref = node.cross_references[0]
    assert ref["type"] == "figure"
    assert ref["target"] == "Figure 3"


def test_crossref_figure_abbreviated():
    text = _make_text("see Fig. 2-1 for pin layout.")
    node = _make_node()
    enrich_with_cross_references([node], text)
    assert len(node.cross_references) == 1
    assert node.cross_references[0]["target"] == "Fig. 2-1"


def test_crossref_table():
    text = _make_text("See Table 5 for electrical specifications.")
    node = _make_node()
    enrich_with_cross_references([node], text)
    assert len(node.cross_references) == 1
    ref = node.cross_references[0]
    assert ref["type"] == "table"
    assert ref["target"] == "Table 5"


def test_crossref_section_without_node_match():
    text = _make_text("See Section 3.2 for more details.")
    node = _make_node()
    enrich_with_cross_references([node], text)
    assert len(node.cross_references) == 1
    ref = node.cross_references[0]
    assert ref["type"] == "section"
    assert ref["target"] == "3.2"
    assert "target_node_id" not in ref


def test_crossref_section_with_node_match():
    text = _make_text("See Section 3.2 for more details.")
    target_node = TocNode(
        title="3.2 Electrical Specs",
        level=2,
        start_page=5,
        end_page=8,
        node_id="0005",
    )
    node = _make_node()
    enrich_with_cross_references([node, target_node], text)
    ref = node.cross_references[0]
    assert ref["target_node_id"] == "0005"


def test_crossref_refer_to_variant():
    text = _make_text("Refer to Table 10 for maximum ratings.")
    node = _make_node()
    enrich_with_cross_references([node], text)
    assert len(node.cross_references) == 1
    assert node.cross_references[0]["type"] == "table"
    assert node.cross_references[0]["target"] == "Table 10"


def test_crossref_deduplication():
    text = _make_text("See Table 1 here. Also see Table 1 there.")
    node = _make_node()
    enrich_with_cross_references([node], text)
    assert len(node.cross_references) == 1


def test_crossref_no_refs():
    text = _make_text("This section has no references to other parts.")
    node = _make_node()
    enrich_with_cross_references([node], text)
    assert node.cross_references == []


def test_crossref_multiple_types():
    text = _make_text("See Figure 1 and refer to Table 2. Also see Section 4.1.")
    target = TocNode(
        title="4.1 Pin Config", level=2, start_page=5, end_page=6, node_id="0010"
    )
    node = _make_node()
    enrich_with_cross_references([node, target], text)
    types = {r["type"] for r in node.cross_references}
    assert types == {"figure", "table", "section"}


# ===================== Integration tests =====================


@pytest.mark.real_pdf
def test_real_pdf_footnotes():
    if not TLE9350_PATH.exists():
        pytest.skip("Test PDF not found")
    doc = pymupdf.open(str(TLE9350_PATH))
    text_content = generate_text(doc)
    raw_toc = extract_toc(doc)
    total_pages = len(doc)
    doc.close()

    nodes = build_tree(raw_toc, total_pages)
    enrich_with_footnote_markers(nodes, text_content)
    # Just verify no crash; real PDF will have variable results
    assert isinstance(nodes, list)


@pytest.mark.real_pdf
def test_real_pdf_crossrefs():
    if not TLE9350_PATH.exists():
        pytest.skip("Test PDF not found")
    doc = pymupdf.open(str(TLE9350_PATH))
    text_content = generate_text(doc)
    raw_toc = extract_toc(doc)
    total_pages = len(doc)
    doc.close()

    nodes = build_tree(raw_toc, total_pages)
    enrich_with_cross_references(nodes, text_content)
    assert isinstance(nodes, list)
