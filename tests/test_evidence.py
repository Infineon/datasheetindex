from __future__ import annotations

import pymupdf

from datasheetindex.core.evidence import (
    EvidenceElement,
    annotate_figure_entries,
    ground_page_span,
    ground_range,
    link_to_toc,
    section_elements,
    table_element,
    text_element,
)
from datasheetindex.core.textfile import scan_pages
from datasheetindex.models import TocNode


def test_scan_pages_preserves_ordered_text_geometry_and_ranges():
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=400)
    page.insert_text((40, 60), "left block")
    page.insert_text((40, 100), "second block")

    scan = scan_pages(doc)
    assert len(scan.evidence) == 2
    first, second = scan.evidence
    assert first["element_id"] == "p0001-text-000"
    assert first["element_type"] == "text_block"
    assert first["source_kind"] == "literal"
    assert first["bbox"][0] == 40.0
    assert len(first["bbox"]) == 4
    assert scan.text[first["text_range"]["start"] : first["text_range"]["end"]].strip()
    assert (
        scan.text[second["text_range"]["start"] : second["text_range"]["end"]].strip()
        == "second block"
    )
    assert ground_range(
        scan.evidence, second["text_range"]["start"], second["text_range"]["end"]
    ) == [second]
    assert ground_page_span(scan.evidence, 1, 0, 4) == [first]
    doc.close()


def test_evidence_helpers_cover_sections_figures_and_tables():
    nodes = [
        TocNode(
            title="Electrical characteristics",
            level=1,
            start_page=2,
            end_page=4,
            node_id="0001",
            breadcrumb="Electrical characteristics",
        )
    ]
    sections = section_elements(nodes)
    assert sections[0]["element_id"] == "section-0001"
    assert sections[0]["page_range"] == {"start": 2, "end": 4}

    figures: list[dict[str, object]] = [
        {
            "page": 2,
            "kind": "raster",
            "bbox": [1, 2, 3, 4],
            "region": {"left": 0.1, "right": 0.3, "top": 0.2, "bottom": 0.4},
            "caption": "Generated caption",
            "caption_source": "generated",
        }
    ]
    figure_elements = annotate_figure_entries(figures)
    assert figures[0]["element_id"] == "p0002-raster-000"
    assert figure_elements[0]["source_kind"] == "generated"

    llm_figures: list[dict[str, object]] = [
        {"page": 2, "kind": "raster", "caption_source": "llm"}
    ]
    assert annotate_figure_entries(llm_figures)[0]["source_kind"] == "generated"

    table = table_element(
        element_id="p0002-table-000",
        page=2,
        bbox=[10, 20, 110, 220],
        page_width=200,
        page_height=400,
    )
    assert table["region"] == {
        "left": 0.05,
        "top": 0.05,
        "right": 0.55,
        "bottom": 0.55,
    }


def test_section_linking_preserves_nested_section_identity():
    child = TocNode(
        title="Child",
        level=2,
        start_page=1,
        end_page=2,
        node_id="0002",
        breadcrumb="Parent > Child",
    )
    parent = TocNode(
        title="Parent",
        level=1,
        start_page=1,
        end_page=5,
        node_id="0001",
        breadcrumb="Parent",
        nodes=[child],
    )

    elements = section_elements([parent])
    link_to_toc(elements, [parent])

    assert elements[0]["node_id"] == "0001"
    assert elements[0]["breadcrumb"] == "Parent"
    assert elements[1]["node_id"] == "0002"
    assert elements[1]["breadcrumb"] == "Parent > Child"


def test_evidence_regions_are_clipped_to_page_bounds():
    element = text_element(
        element_id="p0001-text-000",
        page=1,
        canonical_start=0,
        canonical_end=4,
        page_start=0,
        page_end=4,
        bbox=[390, 390, 402, 402],
        page_width=400,
        page_height=400,
    )

    assert element["bbox"] == [390.0, 390.0, 402.0, 402.0]
    assert element["region"] == {
        "left": 0.975,
        "right": 1.0,
        "top": 0.975,
        "bottom": 1.0,
    }


def test_evidence_and_search_breadcrumb_agree_on_overlapping_siblings():
    """Siblings covering the same page must resolve to one section everywhere.

    3.1 covers pages 4-5 and 3.2 starts on page 5: a search hit on page 5 and
    an evidence record on page 5 must name the same (first, in document order)
    section, or one hit would cite two sections.
    """
    from datasheetindex.core.structure import find_breadcrumb_for_page

    first = TocNode(
        title="3.1",
        level=2,
        start_page=4,
        end_page=5,
        node_id="0002",
        breadcrumb="3 > 3.1",
    )
    second = TocNode(
        title="3.2",
        level=2,
        start_page=5,
        end_page=6,
        node_id="0003",
        breadcrumb="3 > 3.2",
    )
    parent = TocNode(
        title="3",
        level=1,
        start_page=4,
        end_page=6,
        node_id="0001",
        breadcrumb="3",
        nodes=[first, second],
    )
    elements: list[EvidenceElement] = [
        {"element_id": "p0005-text-000", "element_type": "text", "page": 5}
    ]
    link_to_toc(elements, [parent])

    assert elements[0]["breadcrumb"] == find_breadcrumb_for_page([parent], 5)
    assert elements[0]["node_id"] == "0002"
