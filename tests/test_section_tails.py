"""A read keeps its section's tail when the next heading sits mid-page.

Page ranges tile the document: a section ends on the page before its next
sibling starts. When that sibling's heading is partway down its page, the text
above it belongs to the earlier section, and before this change a read of the
earlier section's range silently dropped it -- BME280 "1.3 Pressure sensor
specification" lost half its spec table that way. These tests pin the read-time
fix: the head of the next page is appended up to the heading, every mid-page
heading inside a read is marked, and search breadcrumbs follow position.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from datasheetindex.core.headings import locate_heading
from datasheetindex.core.structure import build_tree
from datasheetindex.core.textfile import extract_page_text
from datasheetindex.models import DatasheetArtifacts
from datasheetindex.tools.bound import DatasheetTools

_PAGES = {
    1: "1 Overview\nIntro text.",
    2: "2 Pressure specification\nAccuracy row A",
    3: (
        "Long term stability row B\n"
        "See 3 Temperature specification for limits.\n"
        "3 Temperature specification\n"
        "Temp row C"
    ),
    4: "4 Ordering\nOrder row D",
}
_TOC = [
    [1, "1 Overview", 1],
    [1, "2 Pressure specification", 2],
    [1, "3 Temperature specification", 3],
    [1, "4 Ordering", 4],
]


def _text(pages: dict[int, str]) -> str:
    return "\n".join(f"--- PAGE {n} ---\n{body}" for n, body in sorted(pages.items()))


def _framing(out: str) -> list[str]:
    return [line for line in out.splitlines() if line.startswith("=== Continued")]


def _tools(pages: dict[int, str] = _PAGES, toc: list[list] = _TOC) -> DatasheetTools:
    total = max(pages)
    tools = DatasheetTools.__new__(DatasheetTools)
    tools._artifacts = DatasheetArtifacts(
        json_data={"total_pages": total, "toc": [], "figures": []},
        text_content=_text(pages),
        nodes=build_tree(toc, total),
    )
    return tools


class TestLocateHeading:
    def test_heading_at_the_top_is_offset_zero(self):
        assert locate_heading("4 Ordering\nOrder row D", "4 Ordering") == 0

    def test_mid_page_heading_is_its_line_start(self):
        text = _PAGES[3]
        assert locate_heading(text, "3 Temperature specification") == text.index(
            "3 Temperature specification\nTemp"
        )

    def test_a_mention_inside_a_line_is_not_the_heading(self):
        assert locate_heading("See 2 Foo for details.\nbody", "2 Foo") is None

    def test_a_heading_wrapped_across_lines_is_found(self):
        text = "tail row\n3 Temperature\nspecification\nrow"
        assert locate_heading(text, "3 Temperature specification") == 9

    def test_case_differences_are_tolerated(self):
        text = "tail row\nELECTRICAL CHARACTERISTICS\nrow"
        assert locate_heading(text, "Electrical Characteristics") == 9

    def test_printed_numbering_absent_from_the_bookmark_is_tolerated(self):
        text = "tail row\n5.2 Recommended Operating Conditions\nrow"
        assert locate_heading(text, "Recommended Operating Conditions") == 9

    def test_absent_title_is_none(self):
        assert locate_heading("nothing here", "4 Ordering") is None


class TestNextPageHead:
    def test_read_includes_the_tail_above_a_mid_page_heading(self):
        out = DatasheetTools.get_section_text(_tools(), 2, 2)
        assert "Accuracy row A" in out
        assert "Long term stability row B" in out
        assert "Temp row C" not in out

    def test_the_appended_head_is_framed_and_names_where_it_stops(self):
        out = DatasheetTools.get_section_text(_tools(), 2, 2)
        framing = [
            line for line in out.splitlines() if line.startswith("=== Continued")
        ]
        assert len(framing) == 1, out
        assert "page 3" in framing[0]
        assert "3 Temperature specification" in framing[0]
        # The framing line comes before the appended text, never a page marker.
        assert out.index(framing[0]) < out.index("Long term stability row B")
        assert "--- PAGE 3 ---" not in out

    def test_nothing_is_appended_when_the_next_heading_opens_its_page(self):
        tools = _tools()
        out = DatasheetTools.get_section_text(tools, 3, 3)
        assert "Order row D" not in out
        assert "=== Continued" not in out

    def test_nothing_is_appended_when_no_section_starts_on_the_next_page(self):
        toc = [[1, "1 Overview", 1], [1, "4 Ordering", 4]]
        out = DatasheetTools.get_section_text(_tools(toc=toc), 2, 2)
        assert "Long term stability row B" not in out

    def test_nothing_is_appended_after_the_last_page(self):
        out = DatasheetTools.get_section_text(_tools(), 4, 4)
        assert out.rstrip().endswith("Order row D")

    def test_unlocated_heading_appends_the_whole_page_and_says_so(self):
        toc = [*_TOC[:2], [1, "3 Thermal limits", 3], _TOC[3]]
        out = DatasheetTools.get_section_text(_tools(toc=toc), 2, 2)
        assert "Long term stability row B" in out
        assert "Temp row C" in out
        framing = [
            line for line in out.splitlines() if line.startswith("=== Continued")
        ]
        assert len(framing) == 1, out
        assert "page 3" in framing[0]
        assert "could not be located" in framing[0]

    def test_the_head_stops_at_the_earliest_of_two_headings(self):
        pages = {
            **_PAGES,
            3: "tail row B\n3 Temperature\nrow C\n3.1 Limits\nrow E",
        }
        toc = [
            *_TOC[:2],
            [1, "3 Temperature", 3],
            [2, "3.1 Limits", 3],
            _TOC[3],
        ]
        out = DatasheetTools.get_section_text(_tools(pages, toc), 2, 2)
        assert "tail row B" in out
        assert "row C" not in out


class TestSectionStartMarkers:
    def test_a_mid_page_heading_inside_the_read_is_marked(self):
        out = DatasheetTools.get_section_text(_tools(), 3, 3)
        lines = out.splitlines()
        marker = next(line for line in lines if line.startswith("=== SECTION STARTS"))
        assert "3 Temperature specification" in marker
        assert lines[lines.index(marker) + 1] == "3 Temperature specification"
        assert out.index("Long term stability row B") < out.index(marker)

    def test_only_the_first_page_of_a_read_is_marked(self):
        """Inside a read, later headings are the read's own structure.

        Marking every one cost a median 5-10% of read characters on the corpus
        (p90 13-20%) while repeating headings the agent can already see. The
        marker earns its place only where the text above it is another
        section's: the first page of the read.
        """
        pages = {**_PAGES, 4: "tail row X\n4 Ordering\nOrder row D"}
        out = DatasheetTools.get_section_text(_tools(pages), 3, 4)
        assert "=== SECTION STARTS: 3 Temperature specification" in out
        assert "=== SECTION STARTS: 4 Ordering" not in out

    def test_the_marker_says_the_text_above_is_another_section(self):
        out = DatasheetTools.get_section_text(_tools(), 3, 3)
        marker = next(line for line in out.splitlines() if "SECTION STARTS" in line)
        assert "above" in marker

    def test_no_marker_when_a_heading_opens_the_first_page(self):
        """Text above a later heading then belongs to the opening section."""
        pages = {**_PAGES, 3: "3 Temperature\nintro\n3.1 Limits\nrow E"}
        toc = [*_TOC[:2], [1, "3 Temperature", 3], [2, "3.1 Limits", 3], _TOC[3]]
        out = DatasheetTools.get_section_text(_tools(pages, toc), 3, 3)
        assert "SECTION STARTS" not in out

    def test_a_heading_opening_its_page_is_not_marked(self):
        out = DatasheetTools.get_section_text(_tools(), 4, 4)
        assert "SECTION STARTS" not in out

    def test_marker_names_the_section_title(self):
        pages = {**_PAGES, 3: "tail row B\n3.1 Limits\nrow E"}
        toc = [*_TOC[:2], [2, "3.1 Limits", 3], _TOC[3]]
        out = DatasheetTools.get_section_text(_tools(pages, toc), 3, 3)
        assert "=== SECTION STARTS: 3.1 Limits (" in out

    def test_leading_spaces_do_not_shift_the_marker_into_the_previous_line(self):
        pages = {**_PAGES, 3: "tail row B\n  3 Temperature specification\nrow C"}
        out = DatasheetTools.get_section_text(_tools(pages), 3, 3)
        assert "tail row B\n=== SECTION STARTS" in out


class TestReadsDoNotMutate:
    def test_repeated_reads_are_identical_and_leave_the_text_untouched(self):
        tools = _tools()
        artifacts = tools._artifacts
        assert artifacts is not None
        before = artifacts.text_content
        first = DatasheetTools.get_section_text(tools, 2, 3)
        second = DatasheetTools.get_section_text(tools, 2, 3)
        assert first == second
        assert artifacts.text_content == before


class TestSearchBreadcrumbs:
    def test_hits_either_side_of_a_mid_page_heading_get_their_own_section(self):
        tools = _tools()
        hits = DatasheetTools.search_text(tools, ["stability row B", "Temp row C"])
        by_pattern = {hit.get("pattern"): hit.get("breadcrumb") for hit in hits}
        assert by_pattern == {
            "stability row B": "2 Pressure specification",
            "Temp row C": "3 Temperature specification",
        }

    def test_a_tail_hit_resolves_to_the_last_descendant_of_the_previous_section(
        self,
    ):
        pages = {
            1: "1 A\nintro",
            2: "1.1 A one\nrow",
            3: "tail of A one\n2 B\nrow of B",
            4: "2.1 B one\nrow",
        }
        toc = [[1, "1 A", 1], [2, "1.1 A one", 2], [1, "2 B", 3], [2, "2.1 B one", 4]]
        hits = DatasheetTools.search_text(_tools(pages, toc), "tail of A one")
        assert hits[0].get("breadcrumb") == "1 A > 1.1 A one"

    def test_text_above_a_mid_page_child_belongs_to_the_parent_opening_the_page(
        self,
    ):
        pages = {
            1: "1 Overview\nintro",
            2: "2 Specifications\nspec intro\n2.1 Limits\nlimit row",
        }
        toc = [[1, "1 Overview", 1], [1, "2 Specifications", 2], [2, "2.1 Limits", 2]]
        tools = _tools(pages, toc)
        hits = DatasheetTools.search_text(tools, ["spec intro", "limit row"])
        by_pattern = {hit.get("pattern"): hit.get("breadcrumb") for hit in hits}
        assert by_pattern == {
            "spec intro": "2 Specifications",
            "limit row": "2 Specifications > 2.1 Limits",
        }


_PSOC = Path(__file__).resolve().parent.parent / (
    "infineon-psoc-6-mcu-cy8c62x8-cy8c62xa-datasheet-datasheet-en.pdf"
)


@pytest.mark.real_pdf
def test_real_pdf_section_read_carries_its_tail(tmp_path):
    """PSoC 6 "3.1.1 CPUs" is pages 12-12, but its text runs onto page 13.

    The CPU description continues at the top of page 13, above "3.1.2
    Interrupts". Before the fix, reading the section's own range ended at page
    12 and the closing paragraph was simply absent.
    """
    if not _PSOC.exists():
        pytest.skip("bundled PSoC datasheet not present")
    with DatasheetTools(str(_PSOC)) as tools:
        tools.build_datasheet(output_dir=str(tmp_path), caption_figures=False)
        artifacts = tools._artifacts
        assert artifacts is not None
        page_12 = extract_page_text(artifacts.text_content, 12)
        out = tools.get_section_text(12, 12)

    tail = "Embedded Cross Trigger"
    assert tail not in page_12
    assert tail in out
    assert '"3.1.2 Interrupts"' in out
    assert out.index("=== Continued from the top of page 13") < out.index(tail)
