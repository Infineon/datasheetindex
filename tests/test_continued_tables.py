"""Tests for multi-page (continued) table detection."""

from pathlib import Path

import pymupdf
import pytest

from datasheetindex.core.structure import (
    build_tree,
    enrich_with_continued_tables,
    extract_toc,
)
from datasheetindex.core.textfile import generate_text

DATA2PAGE_DIR = Path(__file__).resolve().parent.parent.parent / "data2page"
TLE9350_PATH = DATA2PAGE_DIR / "Infineon-TLE9350BSJ-DataSheet-v01_00-EN.pdf"


def _make_text(*page_texts: str) -> str:
    """Build a text_content string from per-page text snippets."""
    parts: list[str] = []
    for i, text in enumerate(page_texts, start=1):
        parts.append(f"--- PAGE {i} ---")
        parts.append(text)
    return "\n".join(parts)


def _make_node(title="Section", start=1, end=2):
    from datasheetindex.models import TocNode

    return TocNode(title=title, level=1, start_page=start, end_page=end)


# --- Unit tests ---


def test_no_markers():
    text = _make_text("Just regular text.", "More text here.")
    node = _make_node(start=1, end=2)
    enrich_with_continued_tables([node], text)
    assert node.continued_tables == []


def test_single_continued_marker():
    text = _make_text(
        "Table 1 Electrical Specs\nSome data",
        "Table 1 Electrical Specs (Continued)\nMore data",
    )
    node = _make_node(start=1, end=2)
    enrich_with_continued_tables([node], text)
    assert node.continued_tables == ["Table 1 Electrical Specs"]


def test_multiple_continued_markers():
    text = _make_text(
        "Table 1 Specs\ndata",
        "Table 1 Specs (Continued)\nTable 2 Timing (Continued)\ndata",
    )
    node = _make_node(start=1, end=2)
    enrich_with_continued_tables([node], text)
    assert node.continued_tables == ["Table 1 Specs", "Table 2 Timing"]


def test_case_insensitive_continued():
    text = _make_text("Table 3 Params (continued)\ndata")
    node = _make_node(start=1, end=1)
    enrich_with_continued_tables([node], text)
    assert node.continued_tables == ["Table 3 Params"]


def test_case_cont_abbreviation():
    text = _make_text("Table 5.1 Output Characteristics (Cont.)\ndata")
    node = _make_node(start=1, end=1)
    enrich_with_continued_tables([node], text)
    assert node.continued_tables == ["Table 5.1 Output Characteristics"]


def test_deduplication():
    text = _make_text(
        "Table 1 Specs (Continued)\ndata",
        "Table 1 Specs (Continued)\nmore data",
    )
    node = _make_node(start=1, end=2)
    enrich_with_continued_tables([node], text)
    assert node.continued_tables == ["Table 1 Specs"]


def test_scoped_to_node_page_range():
    """Markers outside the node's page range should not be included."""
    text = _make_text(
        "Table 1 Specs (Continued)\ndata",
        "Normal text",
        "Table 2 Timing (Continued)\ndata",
    )
    node = _make_node(start=2, end=2)
    enrich_with_continued_tables([node], text)
    assert node.continued_tables == []


def test_nested_nodes():
    """Should process child nodes recursively."""
    from datasheetindex.models import TocNode

    text = _make_text(
        "data", "Table 1 Specs (Continued)\ndata", "Table 2 Timing (Continued)\ndata"
    )
    child = TocNode(title="Sub", level=2, start_page=2, end_page=2)
    parent = TocNode(title="Parent", level=1, start_page=1, end_page=3, nodes=[child])
    enrich_with_continued_tables([parent], text)
    assert "Table 1 Specs" in parent.continued_tables
    assert "Table 2 Timing" in parent.continued_tables
    assert child.continued_tables == ["Table 1 Specs"]


# --- Integration test ---


@pytest.mark.real_pdf
def test_real_pdf():
    if not TLE9350_PATH.exists():
        pytest.skip("Test PDF not found")
    doc = pymupdf.open(str(TLE9350_PATH))
    text_content = generate_text(doc)
    raw_toc = extract_toc(doc)
    total_pages = len(doc)
    doc.close()

    nodes = build_tree(raw_toc, total_pages)
    enrich_with_continued_tables(nodes, text_content)
    # Just verify it runs without error; real PDF may or may not have continued tables
    assert isinstance(nodes, list)
