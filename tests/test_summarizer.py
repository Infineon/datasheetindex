"""Tests for LLM-powered section summarizer."""

from __future__ import annotations

import pytest

from datasheetindex.core.textfile import extract_section_text
from datasheetindex.llm.summarizer import add_summaries
from datasheetindex.models import TocNode

SAMPLE_TEXT = (
    "--- PAGE 1 ---\n"
    "Overview of the TLE9350BSJ high-speed CAN transceiver. "
    "This device provides a robust interface between a CAN protocol "
    "controller and the physical CAN bus. " + "x" * 100 + "\n"
    "--- PAGE 2 ---\n"
    "Features include low power standby mode, wake-up via CAN, "
    "and bus fault detection with thermal shutdown protection. " + "y" * 100 + "\n"
    "--- PAGE 3 ---\n"
    "Electrical specifications: supply voltage range 4.5V to 5.5V.\n"
    "--- PAGE 4 ---\n"
    "Short page.\n"
)


def _make_nodes() -> list[TocNode]:
    """Build a small tree for testing."""
    child = TocNode(
        title="Features",
        level=2,
        start_page=2,
        end_page=2,
        node_id="0002",
    )
    root = TocNode(
        title="Overview",
        level=1,
        start_page=1,
        end_page=3,
        node_id="0001",
        nodes=[child],
    )
    short_node = TocNode(
        title="Appendix",
        level=1,
        start_page=4,
        end_page=4,
        node_id="0003",
    )
    return [root, short_node]


# --- Unit tests ---


def test_extract_section_text():
    """Should extract text between page markers."""
    text = extract_section_text(SAMPLE_TEXT, 1, 2)
    assert "Overview" in text
    assert "Features" in text
    assert "Electrical" not in text


def test_extract_section_text_last_page():
    """Last section should include everything to end of text."""
    text = extract_section_text(SAMPLE_TEXT, 4, 4)
    assert "Short page" in text


def test_extract_section_text_not_found():
    """Missing page marker should return empty string."""
    assert extract_section_text(SAMPLE_TEXT, 99, 99) == ""


def test_add_summaries_mock():
    """Mock LLM should populate summary fields."""

    def mock_llm(system: str, user: str) -> str:
        return "This section covers key specifications."

    nodes = _make_nodes()
    add_summaries(nodes, SAMPLE_TEXT, mock_llm)

    # Root node (pages 1-3) has enough text -> summary populated
    assert nodes[0].summary == "This section covers key specifications."
    # Child node (page 2) has enough text -> summary populated
    assert nodes[0].nodes[0].summary == "This section covers key specifications."


def test_add_summaries_skips_short():
    """Nodes with short text should not get summaries."""
    call_count = [0]

    def mock_llm(system: str, user: str) -> str:
        call_count[0] += 1
        return "Summary"

    nodes = _make_nodes()
    add_summaries(nodes, SAMPLE_TEXT, mock_llm)

    # Appendix (page 4) has very short text -> should be skipped
    assert nodes[1].summary == ""


# --- Integration test ---


@pytest.mark.usefixtures("_has_env")
@pytest.mark.integration
def test_add_summaries_integration():
    """Integration: verify real LLM produces non-empty summaries."""
    from datasheetindex.llm.client import create_llm_client

    llm = create_llm_client()
    nodes = _make_nodes()
    add_summaries(nodes, SAMPLE_TEXT, llm)
    # Root should have a summary (pages 1-3 have enough text)
    assert len(nodes[0].summary) > 0
