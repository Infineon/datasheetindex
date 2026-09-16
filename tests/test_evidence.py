from __future__ import annotations

import base64
import json

import pymupdf
import pytest

from datasheetindex.core import textfile
from datasheetindex.core.evidence import (
    EvidenceElement,
    figure_elements,
    ground_page_span,
    ground_range,
    link_to_toc,
    section_elements,
    table_element,
    text_element,
)
from datasheetindex.core.figures import raster_regions
from datasheetindex.core.textfile import scan_pages
from datasheetindex.models import TocNode
from datasheetindex.tools.bound import DatasheetTools
from datasheetindex.tools.vision import inspect_page


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
    # Inclusive page numbers, named like the ToC -- not the half-open
    # {start, end} shape the character ranges use.
    assert (sections[0]["start_page"], sections[0]["end_page"]) == (2, 4)

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
    before = json.dumps(figures)
    records = figure_elements(figures)
    assert records[0]["element_id"] == "p0002-raster-000"
    assert records[0]["source_kind"] == "generated"
    assert json.dumps(figures) == before, "the ToC JSON figure index was mutated"

    llm_figures: list[dict[str, object]] = [
        {"page": 2, "kind": "raster", "caption_source": "llm"}
    ]
    assert figure_elements(llm_figures)[0]["source_kind"] == "generated"
    caption_only: list[dict[str, object]] = [
        {"page": 3, "kind": "caption", "bbox": None, "region": None}
    ]
    assert "region" not in figure_elements(caption_only)[0]

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
        text="edge",
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


def _ink(doc: pymupdf.Document, region: dict[str, float]) -> int:
    """Dark pixels inspect_page renders for ``region`` on page 1."""
    data = base64.b64decode(inspect_page(doc, 1, region=region, dpi=72)[0]["data"])
    pix = pymupdf.Pixmap(data)
    samples = pix.samples
    return sum(1 for i in range(0, len(samples), pix.n) if samples[i] < 128)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_text_regions_crop_their_own_text_on_rotated_pages(rotation):
    """get_text reports the unrotated page; inspect_page crops the displayed one.

    Unconverted, a 90-degree page produced a region over blank paper, or no
    region at all once the raw y exceeded the rotated height.
    """
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((50, 100), "NEAR ORIGIN", fontsize=24)
    page.insert_text((400, 700), "FAR CORNER", fontsize=24)
    page.set_rotation(rotation)

    evidence = scan_pages(doc).evidence
    assert len(evidence) == 2
    for element in evidence:
        assert "region" in element, element
        assert _ink(doc, element["region"]) > 0, (rotation, element)
    doc.close()


@pytest.mark.parametrize("rotation", [0, 90])
def test_raster_regions_are_in_displayed_page_space(rotation):
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    black = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20), False)
    black.clear_with(0)
    page.insert_image(pymupdf.Rect(50, 88, 300, 300), pixmap=black)
    page.set_rotation(rotation)

    entries, _ = raster_regions(page)
    region = entries[0]["region"]
    assert isinstance(region, dict)
    # Fully inked: the crop is the image, not the paper beside it.
    width = (region["right"] - region["left"]) * page.rect.width
    height = (region["bottom"] - region["top"]) * page.rect.height
    assert _ink(doc, region) >= 0.9 * width * height
    doc.close()


def test_whitespace_only_blocks_are_not_evidence(monkeypatch):
    doc = pymupdf.open()
    doc.new_page(width=400, height=400)
    # Offsets are computed on this text; the blank block must not shift them.
    monkeypatch.setattr(
        textfile,
        "_extract_page_blocks",
        lambda page: [("alpha", False), ("   ", False), ("bravo", False)],
    )
    monkeypatch.setattr(
        textfile,
        "_ordered_blocks",
        lambda page: [
            (10, 10, 50, 20, "alpha", 0, 0),
            (10, 30, 50, 40, "   ", 1, 0),
            (10, 50, 50, 60, "bravo", 2, 0),
        ],
    )
    scan = scan_pages(doc)
    assert [e["element_id"] for e in scan.evidence] == [
        "p0001-text-000",
        "p0001-text-002",
    ]
    bravo = scan.evidence[1]
    assert scan.text[bravo["text_range"]["start"] : bravo["text_range"]["end"]] == (
        "bravo"
    )
    doc.close()


def test_a_block_without_geometry_does_not_shift_later_offsets(monkeypatch):
    """Injected text with no matching block still occupies characters."""
    doc = pymupdf.open()
    doc.new_page(width=400, height=400)
    monkeypatch.setattr(
        textfile,
        "_extract_page_blocks",
        lambda page: [("injected", False), ("second", False)],
    )
    monkeypatch.setattr(
        textfile,
        "_extract_page_block_records",
        lambda page: [
            (None, "injected", False),
            ((10, 10, 50, 20, "second", 0, 0), "second", False),
        ],
    )
    scan = scan_pages(doc)
    (second,) = scan.evidence
    assert scan.text[second["text_range"]["start"] : second["text_range"]["end"]] == (
        "second"
    )
    doc.close()


def _two_page_pdf(tmp_path):
    path = tmp_path / "doc.pdf"
    doc = pymupdf.open()
    for number in (1, 2):
        page = doc.new_page(width=400, height=400)
        page.insert_text((40, 60), f"Supply voltage page {number}")
    doc.set_toc([[1, "Intro", 1], [1, "Specs", 2]])
    doc.save(path)
    doc.close()
    return path


def test_evidence_lives_beside_the_toc_json_not_inside_it(tmp_path):
    pdf = _two_page_pdf(tmp_path)
    out = tmp_path / "out"
    with DatasheetTools(str(pdf)) as tools:
        artifacts = tools.build_datasheet(output_dir=str(out), caption_figures=False)
        assert artifacts.evidence_path == out / "doc.evidence.jsonl"
        assert artifacts.evidence_path is not None
        assert artifacts.json_data["evidence"] == {
            "schema_version": 1,
            "path": "doc.evidence.jsonl",
        }
        lines = artifacts.evidence_path.read_text(encoding="utf-8").splitlines()
        header = json.loads(lines[0])
        records = [json.loads(line) for line in lines[1:]]
        assert header["element_count"] == len(records)
        # Page order, section first: what an agent reading top-down expects.
        assert [(r["page"], r["element_type"]) for r in records] == [
            (1, "section"),
            (1, "text_block"),
            (2, "section"),
            (2, "text_block"),
        ]
        # One grep for the phrase returns the whole record, geometry included.
        (line,) = [line for line in lines if "page 2" in line]
        found = json.loads(line)
        assert found["text"] == "Supply voltage page 2"
        assert found["breadcrumb"] == "Specs"
        assert "region" in found
        assert artifacts.text_path is not None
        text_content = artifacts.text_path.read_text(encoding="utf-8", newline="")
        span = found["text_range"]
        assert text_content[span["start"] : span["end"]] == found["text"]
        assert "evidence" not in tools.get_artifact_manifest()

        hits = tools.search_text("voltage", include_evidence=True)
        assert [h["page"] for h in hits] == [1, 2]
        for hit in hits:
            assert hit["evidence"]
            assert all(e["page"] == hit["page"] for e in hit["evidence"])
        assert "evidence" not in tools.search_text("voltage")[0]
        assert tools.ground_span(page=2, start=0, end=6)


def test_a_modified_evidence_file_is_refused_not_served(tmp_path):
    pdf = _two_page_pdf(tmp_path)
    with DatasheetTools(str(pdf)) as tools:
        artifacts = tools.build_datasheet(
            output_dir=str(tmp_path / "out"), caption_figures=False
        )
        assert artifacts.evidence_path is not None
        artifacts.evidence_path.write_text(
            '{"schema_version":1,"elements":[]}', encoding="utf-8"
        )
        with pytest.raises(RuntimeError, match="changed after it was built"):
            tools.search_text("voltage", include_evidence=True)
