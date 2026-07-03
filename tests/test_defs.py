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

from datasheetindex.tools.defs import (
    DatasheetToolDef,
    DatasheetToolSession,
    create_datasheet_tool_defs,
    create_datasheet_tool_session,
)

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

    The handler contract is ``Callable[[dict], Coroutine[..., dict]]``, so
    ``handler(args)`` is a coroutine that ``asyncio.run`` accepts directly.
    """
    return asyncio.run(handler(args))


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

    # Drive extract_table_markdown too, so every tool handler is exercised here.
    # It needs the optional pymupdf4llm (layout extra); without it the handler
    # returns a clean error envelope rather than raising -- either way is_error
    # is a bool, which is what we assert.
    table_result = _run(defs["extract_table_markdown"].handler, {"page": 1})
    assert isinstance(table_result["is_error"], bool)


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


def test_failed_switch_closes_fresh_instance(tmp_path, monkeypatch):
    """A failed switch closes the fresh instance (no leak) and leaves A open."""
    import datasheetindex.tools.defs as defs_mod

    created: list = []
    real_tools = defs_mod.DatasheetTools

    class TrackingTools(real_tools):
        def __init__(self, pdf_path):
            super().__init__(pdf_path)
            self.close_calls = 0
            created.append(self)

        def close(self):
            self.close_calls += 1
            super().close()

    monkeypatch.setattr(defs_mod, "DatasheetTools", TrackingTools)

    pdf_a = tmp_path / "a.pdf"
    _make_pdf(pdf_a, text="Alpha marker one")
    defs = _defs_by_name()

    _run(
        defs["build_datasheet"].handler,
        {"pdf_source": str(pdf_a), "output_dir": str(tmp_path / "out_a")},
    )
    bad_source = str(tmp_path / "does_not_exist.pdf")
    bad = _run(
        defs["build_datasheet"].handler,
        {"pdf_source": bad_source, "output_dir": str(tmp_path / "out_b")},
    )
    assert bad["is_error"] is True

    bad_instances = [t for t in created if t.pdf_path == bad_source]
    a_instances = [t for t in created if t.pdf_path == str(pdf_a)]
    # The fresh instance built for the bad source must have been closed...
    assert bad_instances and all(t.close_calls >= 1 for t in bad_instances)
    # ...and the still-bound document A must NOT have been closed.
    assert a_instances and all(t.close_calls == 0 for t in a_instances)


def test_successful_switch_survives_old_close_failure(tmp_path, monkeypatch):
    """A successful switch must bind the new document even if closing the old raises."""
    import datasheetindex.tools.defs as defs_mod

    real_tools = defs_mod.DatasheetTools
    fail_close_for: dict = {}

    class FlakyClose(real_tools):
        def close(self):
            if fail_close_for.get("path") == self.pdf_path:
                raise OSError("temp file vanished during cleanup")
            super().close()

    monkeypatch.setattr(defs_mod, "DatasheetTools", FlakyClose)

    pdf_a = tmp_path / "a.pdf"
    pdf_b = tmp_path / "b.pdf"
    _make_pdf(pdf_a, text="Alpha marker one")
    _make_pdf(pdf_b, text="Bravo marker two")
    defs = _defs_by_name()

    _run(
        defs["build_datasheet"].handler,
        {"pdf_source": str(pdf_a), "output_dir": str(tmp_path / "out_a")},
    )
    # Now make closing A blow up, then switch to B.
    fail_close_for["path"] = str(pdf_a)
    res = _run(
        defs["build_datasheet"].handler,
        {"pdf_source": str(pdf_b), "output_dir": str(tmp_path / "out_b")},
    )
    # The switch succeeds and B is bound, despite A's close() raising.
    assert res["is_error"] is False
    bravo = _run(defs["search_text"].handler, {"query": "Bravo"})
    assert json.loads(bravo["content"][0]["text"])["results"]
    alpha = _run(defs["search_text"].handler, {"query": "Alpha"})
    assert json.loads(alpha["content"][0]["text"])["results"] == []


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


def test_create_tool_session_exposes_defs_and_close():
    session = create_datasheet_tool_session()
    assert isinstance(session, DatasheetToolSession)
    assert {d.name for d in session.defs} == EXPECTED_TOOL_NAMES
    assert all(isinstance(d, DatasheetToolDef) for d in session.defs)
    assert callable(session.close)


def test_create_datasheet_tool_defs_matches_session_defs():
    """The list factory stays backward-compatible: same six defs a session exposes."""
    defs = create_datasheet_tool_defs()
    assert {d.name for d in defs} == EXPECTED_TOOL_NAMES


def test_session_close_closes_bound_document(tmp_path, monkeypatch):
    """session.close() must close the currently bound DatasheetTools (temp cleanup)."""
    import datasheetindex.tools.defs as defs_mod

    created: list = []
    real_tools = defs_mod.DatasheetTools

    class TrackingTools(real_tools):
        def __init__(self, pdf_path):
            super().__init__(pdf_path)
            self.close_calls = 0
            created.append(self)

        def close(self):
            self.close_calls += 1
            super().close()

    monkeypatch.setattr(defs_mod, "DatasheetTools", TrackingTools)

    pdf_a = tmp_path / "a.pdf"
    _make_pdf(pdf_a)
    session = create_datasheet_tool_session()
    handlers = {d.name: d for d in session.defs}

    _run(
        handlers["build_datasheet"].handler,
        {"pdf_source": str(pdf_a), "output_dir": str(tmp_path / "out")},
    )
    bound = created[-1]
    assert bound.close_calls == 0
    assert bound._doc is not None  # building opened the document

    session.close()
    assert bound.close_calls >= 1
    assert bound._doc is None  # the underlying document handle was released


def test_session_close_is_safe_when_unbound_and_idempotent():
    session = create_datasheet_tool_session()
    # No document ever bound -> must not raise; and calling twice is safe.
    session.close()
    session.close()
