"""Tests for the framework-neutral datasheet tool definitions.

These exercise the tool handlers directly, without ``claude-agent-sdk`` -- the
whole point of ``create_datasheet_tool_defs`` is that a non-SDK host can realize
the tools without importing the SDK.
"""

import asyncio
import dataclasses
import inspect
import json
import sys

import pymupdf
import pytest

from datasheetindex.tools.defs import DatasheetToolDef, create_datasheet_tool_defs

EXPECTED_TOOL_NAMES = {
    "build_datasheet",
    "get_section_text",
    "search_text",
    "inspect_page",
    "locate_text",
    "extract_table_markdown",
}


def _make_pdf(path, text="Supply voltage 4.5V to 5.5V"):
    """Write a one-page PDF with a line of text at (72, 72)."""
    doc = pymupdf.open()
    page = doc.new_page()
    writer = pymupdf.TextWriter(page.rect)
    writer.append((72, 72), text)
    writer.write_text(page)
    doc.save(str(path))
    doc.close()


def _defs_by_name():
    return {d.name: d for d in create_datasheet_tool_defs()}


def _run(handler, args):
    """Drive a neutral tool handler synchronously.

    The handler contract is ``Callable[[dict], Awaitable[dict]]``; awaiting it
    inside a coroutine keeps the call type-clean for asyncio.run.
    """

    async def _invoke():
        return await handler(args)

    return asyncio.run(_invoke())


def test_create_tool_defs_returns_expected_tools():
    defs = create_datasheet_tool_defs()

    assert {d.name for d in defs} == EXPECTED_TOOL_NAMES
    for d in defs:
        assert isinstance(d, DatasheetToolDef)
        assert isinstance(d.description, str) and d.description
        assert d.input_schema.get("type") == "object"
        assert inspect.iscoroutinefunction(d.handler)


def test_tool_def_is_frozen():
    d = create_datasheet_tool_defs()[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.name = "renamed"  # ty: ignore[invalid-assignment]


def test_create_tool_defs_does_not_import_sdk():
    """Realizing the neutral defs must not pull in claude-agent-sdk."""
    sys.modules.pop("claude_agent_sdk", None)
    create_datasheet_tool_defs()
    assert "claude_agent_sdk" not in sys.modules


def test_query_handlers_require_build_first():
    """Query tools return an error envelope until build_datasheet has run."""
    defs = _defs_by_name()

    result = _run(defs["get_section_text"].handler, {"start_page": 1, "end_page": 1})
    assert result["is_error"] is True
    assert "No datasheet loaded" in result["content"][0]["text"]


def test_build_then_query_end_to_end(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    _make_pdf(pdf_path)
    defs = _defs_by_name()

    build_result = _run(
        defs["build_datasheet"].handler,
        {"pdf_source": str(pdf_path), "output_dir": str(tmp_path / "out")},
    )
    assert build_result["is_error"] is False

    section_result = _run(
        defs["get_section_text"].handler, {"start_page": 1, "end_page": 1}
    )
    assert section_result["is_error"] is False
    section_payload = json.loads(section_result["content"][0]["text"])
    assert "Supply voltage" in section_payload["text"]

    search_result = _run(defs["search_text"].handler, {"query": "5.5v"})
    assert search_result["is_error"] is False
    search_payload = json.loads(search_result["content"][0]["text"])
    assert search_payload["results"][0]["page"] == 1

    locate_result = _run(defs["locate_text"].handler, {"query": "Supply", "page": 1})
    assert locate_result["is_error"] is False
    locate_payload = json.loads(locate_result["content"][0]["text"])
    assert locate_payload["results"][0]["match_method"] == "search_for"

    inspect_result = _run(defs["inspect_page"].handler, {"page": 1})
    assert inspect_result["is_error"] is False
    assert inspect_result["content"][0]["type"] == "image"


def test_build_datasheet_rebinds_on_new_source(tmp_path):
    """A second build with a different source switches the active document."""
    pdf_a = tmp_path / "a.pdf"
    pdf_b = tmp_path / "b.pdf"
    _make_pdf(pdf_a, text="Alpha marker one")
    _make_pdf(pdf_b, text="Bravo marker two")
    defs = _defs_by_name()

    _run(
        defs["build_datasheet"].handler,
        {"pdf_source": str(pdf_a), "output_dir": str(tmp_path / "out_a")},
    )
    _run(
        defs["build_datasheet"].handler,
        {"pdf_source": str(pdf_b), "output_dir": str(tmp_path / "out_b")},
    )

    alpha = _run(defs["search_text"].handler, {"query": "Alpha"})
    bravo = _run(defs["search_text"].handler, {"query": "Bravo"})

    assert json.loads(alpha["content"][0]["text"])["results"] == []
    assert json.loads(bravo["content"][0]["text"])["results"]


def test_failed_switch_preserves_working_document(tmp_path):
    """A failed switch to a bad source must leave the working document intact."""
    pdf_a = tmp_path / "a.pdf"
    _make_pdf(pdf_a, text="Alpha marker one")
    defs = _defs_by_name()

    ok = _run(
        defs["build_datasheet"].handler,
        {"pdf_source": str(pdf_a), "output_dir": str(tmp_path / "out_a")},
    )
    assert ok["is_error"] is False

    # Switch to a source that cannot be opened -> the build must fail.
    bad = _run(
        defs["build_datasheet"].handler,
        {
            "pdf_source": str(tmp_path / "does_not_exist.pdf"),
            "output_dir": str(tmp_path / "out_b"),
        },
    )
    assert bad["is_error"] is True

    # ...but document A must still be bound and queryable, not closed.
    section = _run(defs["get_section_text"].handler, {"start_page": 1, "end_page": 1})
    assert section["is_error"] is False
    assert "Alpha marker" in json.loads(section["content"][0]["text"])["text"]


def test_build_datasheet_requires_pdf_source():
    defs = _defs_by_name()
    result = _run(defs["build_datasheet"].handler, {})
    assert result["is_error"] is True
    assert "pdf_source is required" in result["content"][0]["text"]


def test_each_defs_call_is_an_independent_session(tmp_path):
    """Two factory calls own separate state -- binding one must not affect the other."""
    pdf_path = tmp_path / "test.pdf"
    _make_pdf(pdf_path)

    session_a = {d.name: d for d in create_datasheet_tool_defs()}
    session_b = {d.name: d for d in create_datasheet_tool_defs()}

    _run(
        session_a["build_datasheet"].handler,
        {"pdf_source": str(pdf_path), "output_dir": str(tmp_path / "out")},
    )

    # session_b never built -> still unbound.
    result_b = _run(session_b["search_text"].handler, {"query": "voltage"})
    assert result_b["is_error"] is True
    assert "No datasheet loaded" in result_b["content"][0]["text"]
