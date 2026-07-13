"""Tests for LLM-based ToC fallback generation."""

from __future__ import annotations

import json

import pytest

from datasheetindex.core.quality import assess_toc_quality
from datasheetindex.llm.client import StructuredLlmResult
from datasheetindex.llm.toc_fallback import (
    MAX_PREVIOUS_CONTEXT_CHARS,
    _format_previous_entries,
    _parse_json_response,
    _split_into_chunks,
    generate_toc_from_text,
)

# --- Unit tests ---


def test_split_into_chunks_single():
    """Short text should produce a single chunk."""
    text = "--- PAGE 1 ---\nHello world\n--- PAGE 2 ---\nGoodbye\n"
    chunks = _split_into_chunks(text, max_chars=10000)
    assert len(chunks) == 1
    assert "PAGE 1" in chunks[0]
    assert "PAGE 2" in chunks[0]


def test_split_into_chunks_multiple():
    """Text exceeding max_chars should split on page boundaries."""
    pages = []
    for i in range(1, 11):
        pages.append(f"--- PAGE {i} ---\n{'x' * 200}\n")
    text = "".join(pages)

    chunks = _split_into_chunks(text, max_chars=500)
    assert len(chunks) > 1
    # Every chunk should contain at least one PAGE marker
    for chunk in chunks:
        assert "PAGE" in chunk


def test_split_empty_text():
    """Empty text should produce no chunks."""
    assert _split_into_chunks("", max_chars=1000) == []


def test_parse_json_response_clean():
    """Clean JSON array should parse correctly."""
    raw = '[{"level": 1, "title": "Intro", "start_page": 1}]'
    result = _parse_json_response(raw)
    assert len(result) == 1
    assert result[0]["title"] == "Intro"
    assert result[0]["level"] == 1
    assert result[0]["start_page"] == 1


def test_parse_json_response_code_fence():
    """JSON wrapped in markdown code fences should parse."""
    raw = '```json\n[{"level": 1, "title": "Intro", "start_page": 1}]\n```'
    result = _parse_json_response(raw)
    assert len(result) == 1
    assert result[0]["title"] == "Intro"


def test_parse_json_response_invalid():
    """Non-JSON response should return empty list."""
    assert _parse_json_response("This is not JSON") == []


def test_parse_json_response_missing_fields():
    """Entries missing required fields should be filtered out."""
    raw = '[{"level": 1, "title": "Good", "start_page": 1}, {"level": 2}]'
    result = _parse_json_response(raw)
    assert len(result) == 1
    assert result[0]["title"] == "Good"


def test_format_previous_entries_bounds_context():
    entries = [
        {"level": 1, "title": f"Section {i} " + ("x" * 80), "start_page": i}
        for i in range(1, 80)
    ]

    text = _format_previous_entries(entries)

    assert len(text) <= MAX_PREVIOUS_CONTEXT_CHARS
    assert text.startswith("[")
    assert text.endswith("]")


def test_generate_toc_from_text_mock():
    """With a mock LLM, should produce a valid tree."""
    canned = json.dumps(
        [
            {"level": 1, "title": "Overview", "start_page": 1},
            {"level": 2, "title": "Features", "start_page": 2},
            {"level": 1, "title": "Specifications", "start_page": 5},
        ]
    )

    def mock_llm(system: str, user: str) -> str:
        return canned

    text = "--- PAGE 1 ---\nOverview\n--- PAGE 2 ---\nFeatures\n"
    text += "--- PAGE 5 ---\nSpecs\n"

    nodes = generate_toc_from_text(text, total_pages=10, llm_callable=mock_llm)
    assert len(nodes) == 2  # Two level-1 nodes
    assert nodes[0].title == "Overview"
    assert nodes[0].nodes[0].title == "Features"
    assert nodes[1].title == "Specifications"
    # End pages computed
    assert nodes[0].end_page == 4
    assert nodes[1].end_page == 10
    # Node IDs assigned
    assert nodes[0].node_id == "0001"


def test_generate_toc_multi_chunk_mock():
    """Multi-chunk text should accumulate entries from both calls."""
    call_count = [0]

    def mock_llm(system: str, user: str) -> str:
        call_count[0] += 1
        if call_count[0] == 1:
            return json.dumps([{"level": 1, "title": "Part A", "start_page": 1}])
        return json.dumps([{"level": 1, "title": "Part B", "start_page": 50}])

    # Build text large enough to split into 2 chunks
    pages = []
    for i in range(1, 101):
        pages.append(f"--- PAGE {i} ---\n{'content ' * 30}\n")
    text = "".join(pages)

    nodes = generate_toc_from_text(text, total_pages=100, llm_callable=mock_llm)
    assert len(nodes) == 2
    assert nodes[0].title == "Part A"
    assert nodes[1].title == "Part B"
    assert call_count[0] >= 2


def test_generate_toc_from_text_structured_mock():
    """A structured-capable client should use schema-constrained chunk calls."""

    class _StructuredMockLlm:
        def __init__(self) -> None:
            self.call_count = 0

        def __call__(self, system: str, user: str) -> str:
            raise AssertionError("legacy text path should not be used")

        def structured_json(
            self, _system: str, _user: str, **_kwargs
        ) -> StructuredLlmResult:
            self.call_count += 1
            if self.call_count == 1:
                payload = {
                    "entries": [{"level": 1, "title": "Part A", "start_page": 1}]
                }
            else:
                payload = {
                    "entries": [
                        {"level": 1, "title": "Part A", "start_page": 1},
                        {"level": 1, "title": "Part B", "start_page": 50},
                    ]
                }
            return StructuredLlmResult(
                output_text=json.dumps(payload),
                status="completed",
            )

    pages = []
    for i in range(1, 101):
        pages.append(f"--- PAGE {i} ---\n{'content ' * 30}\n")
    text = "".join(pages)

    llm = _StructuredMockLlm()
    nodes = generate_toc_from_text(text, total_pages=100, llm_callable=llm)

    assert len(nodes) == 2
    assert nodes[0].title == "Part A"
    assert nodes[1].title == "Part B"
    assert llm.call_count >= 2


def _multi_chunk_text(pages: int = 100) -> str:
    """Text long enough that ``_split_into_chunks`` produces several chunks."""
    return "".join(
        f"--- PAGE {i} ---\n{'content ' * 30}\n" for i in range(1, pages + 1)
    )


def test_generate_toc_from_text_structured_truncated_chunk_keeps_other_chunks():
    """A truncated chunk is skipped; entries from the other chunks survive."""

    class _StructuredMockLlm:
        def __init__(self) -> None:
            self.call_count = 0

        def __call__(self, system: str, user: str) -> str:
            raise AssertionError("free-text path should not be used")

        def structured_json(
            self, _system: str, _user: str, **_kwargs
        ) -> StructuredLlmResult:
            self.call_count += 1
            if self.call_count == 2:
                return StructuredLlmResult(
                    output_text="",
                    status="incomplete",
                    incomplete_details={"reason": "max_output_tokens"},
                )
            payload = {
                "entries": [
                    {
                        "level": 1,
                        "title": f"Part {self.call_count}",
                        "start_page": self.call_count * 10,
                    }
                ]
            }
            return StructuredLlmResult(
                output_text=json.dumps(payload), status="completed"
            )

    llm = _StructuredMockLlm()
    nodes = generate_toc_from_text(
        _multi_chunk_text(pages=200), total_pages=200, llm_callable=llm
    )

    assert llm.call_count >= 3
    titles = [node.title for node in nodes]
    assert "Part 1" in titles
    assert "Part 3" in titles
    assert "Part 2" not in titles


def test_generate_toc_from_text_structured_missing_status_is_accepted():
    """A gateway that reports no status at all still yields a usable ToC."""

    class _StructuredMockLlm:
        def __call__(self, system: str, user: str) -> str:
            raise AssertionError("free-text path should not be used")

        def structured_json(
            self, _system: str, _user: str, **_kwargs
        ) -> StructuredLlmResult:
            payload = {"entries": [{"level": 1, "title": "Overview", "start_page": 1}]}
            return StructuredLlmResult(output_text=json.dumps(payload))

    nodes = generate_toc_from_text(
        "--- PAGE 1 ---\nOverview\n",
        total_pages=1,
        llm_callable=_StructuredMockLlm(),
    )

    assert [node.title for node in nodes] == ["Overview"]


def test_generate_toc_from_text_falls_back_to_free_text_when_structured_unusable():
    """A client whose structured path is rejected still produces a ToC."""

    class _StructuredMockLlm:
        def __init__(self) -> None:
            self.text_calls = 0

        def __call__(self, system: str, user: str) -> str:
            self.text_calls += 1
            return json.dumps([{"level": 1, "title": "Overview", "start_page": 1}])

        def structured_json(
            self, _system: str, _user: str, **_kwargs
        ) -> StructuredLlmResult:
            raise RuntimeError("json_schema is not supported by this model")

    llm = _StructuredMockLlm()
    nodes = generate_toc_from_text(
        "--- PAGE 1 ---\nOverview\n", total_pages=1, llm_callable=llm
    )

    assert llm.text_calls == 1
    assert [node.title for node in nodes] == ["Overview"]


def test_generate_toc_skips_entries_with_unusable_page():
    """A null start_page drops that entry, not the whole extraction."""

    def mock_llm(system: str, user: str) -> str:
        return json.dumps(
            [
                {"level": 1, "title": "Features", "start_page": None},
                {"level": 1, "title": "Overview", "start_page": 1},
            ]
        )

    nodes = generate_toc_from_text(
        "--- PAGE 1 ---\nOverview\n", total_pages=1, llm_callable=mock_llm
    )

    assert [node.title for node in nodes] == ["Overview"]


def test_generate_toc_invalid_level_raises():
    """Invalid levels should raise with clear validation errors."""

    def mock_llm(system: str, user: str) -> str:
        return json.dumps([{"level": 0, "title": "Bad", "start_page": 1}])

    text = "--- PAGE 1 ---\nBad\n"
    with pytest.raises(ValueError, match="Invalid ToC level"):
        generate_toc_from_text(text, total_pages=1, llm_callable=mock_llm)


# --- Integration test ---


@pytest.mark.usefixtures("_has_env")
@pytest.mark.integration
def test_generate_toc_integration():
    """Integration: generate ToC from sample datasheet text."""
    from datasheetindex.llm.client import create_llm_client

    sample_text = (
        "--- PAGE 1 ---\n"
        "TLE9350BSJ Data Sheet\nHigh-speed CAN transceiver\n\n"
        "--- PAGE 2 ---\n"
        "Table of Contents\n"
        "1 Overview\n2 Features\n3 Electrical Specifications\n\n"
        "--- PAGE 3 ---\n"
        "1 Overview\nThe TLE9350BSJ is a high-speed CAN transceiver.\n\n"
        "--- PAGE 4 ---\n"
        "2 Features\n- Low power mode\n- Wake-up pattern detection\n\n"
        "--- PAGE 5 ---\n"
        "3 Electrical Specifications\n3.1 Absolute Maximum Ratings\n\n"
    )

    llm = create_llm_client()
    nodes = generate_toc_from_text(sample_text, total_pages=5, llm_callable=llm)
    assert len(nodes) > 0
    # At least one node should reference page 1-5
    all_pages = set()
    for n in nodes:
        all_pages.add(n.start_page)
    assert len(all_pages) > 1

    quality = assess_toc_quality(nodes, total_pages=5)
    assert quality.score >= 0.5, quality.details
