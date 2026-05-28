"""Tests for ToC tree building and enrichment."""

from pathlib import Path

import pymupdf
import pytest

from datasheetindex.core.structure import (
    assign_breadcrumbs,
    build_tree,
    enrich_with_table_counts,
    extract_toc,
)
from datasheetindex.models import TocNode

DATA2PAGE_DIR = Path(__file__).resolve().parent.parent.parent / "data2page"
TLE9350_PATH = DATA2PAGE_DIR / "Infineon-TLE9350BSJ-DataSheet-v01_00-EN.pdf"


# --- Unit tests with synthetic ToC data ---


def test_empty_toc():
    nodes = build_tree([], total_pages=10)
    assert nodes == []


def test_single_entry():
    raw = [[1, "Overview", 1]]
    nodes = build_tree(raw, total_pages=5)
    assert len(nodes) == 1
    assert nodes[0].title == "Overview"
    assert nodes[0].start_page == 1
    assert nodes[0].end_page == 5
    assert nodes[0].node_id == "0001"


def test_invalid_toc_entry_shape_raises():
    raw = [[1, "Overview"]]
    with pytest.raises(ValueError, match="must include"):
        build_tree(raw, total_pages=5)


def test_invalid_toc_level_raises():
    raw = [[0, "Overview", 1]]
    with pytest.raises(ValueError, match="Invalid ToC level"):
        build_tree(raw, total_pages=5)


def test_flat_entries():
    raw = [
        [1, "A", 1],
        [1, "B", 3],
        [1, "C", 6],
    ]
    nodes = build_tree(raw, total_pages=10)
    assert len(nodes) == 3
    assert nodes[0].end_page == 2  # next sibling starts at 3
    assert nodes[1].end_page == 5  # next sibling starts at 6
    assert nodes[2].end_page == 10  # last node gets parent_end


def test_nested_two_levels():
    raw = [
        [1, "Section 1", 1],
        [2, "Sub 1.1", 1],
        [2, "Sub 1.2", 3],
        [1, "Section 2", 5],
    ]
    nodes = build_tree(raw, total_pages=10)
    assert len(nodes) == 2
    assert nodes[0].title == "Section 1"
    assert len(nodes[0].nodes) == 2
    assert nodes[0].nodes[0].title == "Sub 1.1"
    assert nodes[0].nodes[0].end_page == 2
    assert nodes[0].nodes[1].title == "Sub 1.2"
    assert nodes[0].nodes[1].end_page == 4  # parent end is 4

    assert nodes[1].title == "Section 2"
    assert nodes[1].end_page == 10


def test_deep_three_levels():
    raw = [
        [1, "Ch1", 1],
        [2, "Sec1.1", 1],
        [3, "Sub1.1.1", 1],
        [3, "Sub1.1.2", 3],
        [2, "Sec1.2", 5],
        [1, "Ch2", 8],
    ]
    nodes = build_tree(raw, total_pages=12)
    assert len(nodes) == 2
    ch1 = nodes[0]
    assert ch1.end_page == 7
    assert len(ch1.nodes) == 2
    sec11 = ch1.nodes[0]
    assert sec11.end_page == 4
    assert len(sec11.nodes) == 2
    assert sec11.nodes[0].end_page == 2
    assert sec11.nodes[1].end_page == 4


def test_breadcrumb_root_node():
    raw = [[1, "Overview", 1]]
    nodes = build_tree(raw, total_pages=5)
    assert nodes[0].breadcrumb == "Overview"


def test_breadcrumb_nested():
    raw = [
        [1, "5 Electrical Characteristics", 10],
        [2, "5.1 Absolute Maximum Ratings", 10],
        [3, "5.1.1 Junction Temperature", 11],
        [1, "6 Pin Configuration", 15],
    ]
    nodes = build_tree(raw, total_pages=20)
    assert nodes[0].breadcrumb == "5 Electrical Characteristics"
    assert (
        nodes[0].nodes[0].breadcrumb
        == "5 Electrical Characteristics > 5.1 Absolute Maximum Ratings"
    )
    assert nodes[0].nodes[0].nodes[0].breadcrumb == (
        "5 Electrical Characteristics > 5.1 Absolute Maximum Ratings"
        " > 5.1.1 Junction Temperature"
    )
    assert nodes[1].breadcrumb == "6 Pin Configuration"


def test_build_tree_populates_boilerplate_category():
    """Catches regressions where `flag_boilerplate` gets removed from
    `build_tree`. Without it, the field would be empty for a clear match."""
    raw = [
        [1, "Electrical Characteristics", 1],
        [1, "Revision History", 10],
    ]
    nodes = build_tree(raw, total_pages=15)
    assert nodes[0].boilerplate_category == ""
    assert nodes[1].boilerplate_category == "revision"


def test_assign_breadcrumbs_strips_title_whitespace():
    nodes = [
        TocNode(
            title="  Outer  ",
            level=1,
            start_page=1,
            nodes=[TocNode(title="\tInner\n", level=2, start_page=1)],
        )
    ]
    assign_breadcrumbs(nodes)
    assert nodes[0].breadcrumb == "Outer"
    assert nodes[0].nodes[0].breadcrumb == "Outer > Inner"


def test_node_ids_depth_first():
    raw = [
        [1, "A", 1],
        [2, "A.1", 1],
        [2, "A.2", 3],
        [1, "B", 5],
    ]
    nodes = build_tree(raw, total_pages=10)
    # Depth-first: A=0001, A.1=0002, A.2=0003, B=0004
    assert nodes[0].node_id == "0001"
    assert nodes[0].nodes[0].node_id == "0002"
    assert nodes[0].nodes[1].node_id == "0003"
    assert nodes[1].node_id == "0004"


def test_architecture_doc_example():
    """Verify end_page computation matches the architecture doc example.

    Architecture doc specifies:
    - Block diagram start=5, Pin Config start=6 -> Block diagram end=5
    - Last section gets total_pages as end
    """
    raw = [
        [1, "Overview", 1],
        [1, "Block Diagram", 5],
        [1, "Pin Configuration", 6],
        [1, "Electrical Characteristics", 10],
    ]
    nodes = build_tree(raw, total_pages=20)
    assert nodes[0].end_page == 4  # Overview: 1 to 4
    assert nodes[1].end_page == 5  # Block Diagram: 5 to 5
    assert nodes[2].end_page == 9  # Pin Configuration: 6 to 9
    assert nodes[3].end_page == 20  # Last section: 10 to 20


def test_malformed_child_start_after_parent_end_is_clamped():
    """Last child should never get end_page lower than its start_page."""
    raw = [
        [1, "16 Communication", 50],
        [2, "16.1 Functional description", 51],
        [3, "16.1.1 Register write modes", 53],
        [3, "16.1.2 Communication frames", 53],
        [3, "16.1.3 Register read modes", 55],
        [2, "16.2 Electrical characteristics communication", 55],
    ]
    nodes = build_tree(raw, total_pages=73)

    section_161 = nodes[0].nodes[0]
    last_child = section_161.nodes[2]
    assert last_child.start_page == 55
    assert last_child.end_page == 55


# --- Integration tests with real PDF ---


@pytest.mark.real_pdf
def test_real_pdf_extract_toc():
    if not TLE9350_PATH.exists():
        pytest.skip("Test PDF not found")
    doc = pymupdf.open(str(TLE9350_PATH))
    raw_toc = extract_toc(doc)
    doc.close()
    assert len(raw_toc) > 0
    # Each entry should be [level, title, page]
    for entry in raw_toc:
        assert len(entry) >= 3
        assert isinstance(entry[0], int)
        assert isinstance(entry[1], str)
        assert isinstance(entry[2], int)


@pytest.mark.real_pdf
def test_real_pdf_build_tree():
    if not TLE9350_PATH.exists():
        pytest.skip("Test PDF not found")
    doc = pymupdf.open(str(TLE9350_PATH))
    raw_toc = extract_toc(doc)
    total_pages = len(doc)
    doc.close()

    nodes = build_tree(raw_toc, total_pages)
    assert len(nodes) > 0

    # All node_ids should be unique
    all_ids: list[str] = []
    _collect_ids(nodes, all_ids)
    assert len(all_ids) == len(set(all_ids))

    # All end_pages should be >= start_pages
    _assert_valid_ranges(nodes)


@pytest.mark.real_pdf
def test_real_pdf_table_enrichment():
    if not TLE9350_PATH.exists():
        pytest.skip("Test PDF not found")
    doc = pymupdf.open(str(TLE9350_PATH))
    raw_toc = extract_toc(doc)
    total_pages = len(doc)
    nodes = build_tree(raw_toc, total_pages)
    enrich_with_table_counts(nodes, doc)
    doc.close()

    # At least some sections should have tables (it's a datasheet)
    all_nodes: list = []
    _collect_all(nodes, all_nodes)
    has_any_tables = any(n.has_tables for n in all_nodes)
    assert has_any_tables


# --- Helpers ---


def _collect_ids(nodes, ids):
    for node in nodes:
        ids.append(node.node_id)
        _collect_ids(node.nodes, ids)


def _assert_valid_ranges(nodes):
    for node in nodes:
        assert node.end_page >= node.start_page, (
            f"{node.title}: end_page {node.end_page} < start_page {node.start_page}"
        )
        _assert_valid_ranges(node.nodes)


def _collect_all(nodes, result):
    for node in nodes:
        result.append(node)
        _collect_all(node.nodes, result)
