"""Tests for tool registry."""

import sys
import types
from pathlib import Path

import pymupdf
import pytest

from datasheetindex.tools.registry import (
    DatasheetTools,
    create_datasheet_tools_server,
)

DATA2PAGE_DIR = Path(__file__).resolve().parent.parent.parent / "data2page"
TLE9350_PATH = DATA2PAGE_DIR / "Infineon-TLE9350BSJ-DataSheet-v01_00-EN.pdf"


def test_datasheet_tools_inspect_page(tmp_path):
    """DatasheetTools.inspect_page should work with a valid PDF."""
    # Create a minimal test PDF
    pdf_path = tmp_path / "test.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    writer = pymupdf.TextWriter(page.rect)
    writer.append((72, 72), "Registry test")
    writer.write_text(page)
    doc.save(str(pdf_path))
    doc.close()

    tools = DatasheetTools(str(pdf_path))
    result = tools.inspect_page(page=1)
    tools.close()

    assert len(result) == 1
    assert result[0]["type"] == "image"


def test_datasheet_tools_build_and_query_artifacts(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    writer = pymupdf.TextWriter(page.rect)
    writer.append((72, 72), "Supply voltage 4.5V to 5.5V")
    writer.write_text(page)
    doc.save(str(pdf_path))
    doc.close()

    output_dir = tmp_path / "out"
    tools = DatasheetTools(str(pdf_path))
    artifacts = tools.build_datasheet(output_dir=str(output_dir))

    section_text = tools.get_section_text(1, 1)
    matches = tools.search_text("5.5v")
    tools.close()

    assert artifacts.json_path is not None
    assert artifacts.text_path is not None
    assert "--- PAGE 1 ---" in section_text
    assert "Supply voltage" in section_text
    assert matches == [
        {
            "page": 1,
            "start": 23,
            "end": 27,
            "snippet": "Supply voltage 4.5V to 5.5V",
        }
    ]


def test_build_datasheet_omitted_output_dir_uses_resolver(monkeypatch, tmp_path):
    """DatasheetTools.build_datasheet(output_dir=None) writes to resolver default."""
    pdf_path = tmp_path / "test.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    writer = pymupdf.TextWriter(page.rect)
    writer.append((72, 72), "Hello")
    writer.write_text(page)
    doc.save(str(pdf_path))
    doc.close()

    pinned = tmp_path / "env-pinned"
    monkeypatch.setenv("DATASHEETINDEX_OUTPUT_DIR", str(pinned))

    tools = DatasheetTools(str(pdf_path))
    try:
        artifacts = tools.build_datasheet()
    finally:
        tools.close()

    assert artifacts.json_path is not None
    assert artifacts.json_path.parent == pinned


def test_build_datasheet_cache_invalidated_when_resolver_changes(monkeypatch, tmp_path):
    """Cache must miss if env var (and thus resolver default) changed between calls."""
    pdf_path = tmp_path / "test.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    writer = pymupdf.TextWriter(page.rect)
    writer.append((72, 72), "Hello")
    writer.write_text(page)
    doc.save(str(pdf_path))
    doc.close()

    first = tmp_path / "first"
    second = tmp_path / "second"

    tools = DatasheetTools(str(pdf_path))
    try:
        monkeypatch.setenv("DATASHEETINDEX_OUTPUT_DIR", str(first))
        a1 = tools.build_datasheet()
        monkeypatch.setenv("DATASHEETINDEX_OUTPUT_DIR", str(second))
        a2 = tools.build_datasheet()
    finally:
        tools.close()

    assert a1.json_path is not None and a1.json_path.parent == first
    assert a2.json_path is not None and a2.json_path.parent == second


def test_datasheet_tools_artifact_queries_require_build(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()

    tools = DatasheetTools(str(pdf_path))
    with pytest.raises(RuntimeError, match="build_datasheet"):
        tools.get_section_text(1, 1)
    with pytest.raises(RuntimeError, match="build_datasheet"):
        tools.search_text("foo")
    tools.close()


def test_datasheet_tools_lazy_doc(tmp_path):
    """Document should be lazy-opened."""
    pdf_path = tmp_path / "test.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()

    tools = DatasheetTools(str(pdf_path))
    assert tools._doc is None
    _ = tools.doc
    assert tools._doc is not None
    tools.close()
    assert tools._doc is None


def test_create_server_raises_without_sdk():
    """create_datasheet_tools_server should raise ImportError without SDK."""
    with pytest.raises(ImportError, match="claude-agent-sdk"):
        create_datasheet_tools_server()


def test_create_server_registers_tools(monkeypatch, tmp_path):
    """Server factory should register 5 agent-ready tools via SDK pattern."""
    import asyncio

    pdf_path = tmp_path / "test.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    writer = pymupdf.TextWriter(page.rect)
    writer.append((72, 72), "Registry MCP test")
    writer.write_text(page)
    doc.save(str(pdf_path))
    doc.close()

    def fake_tool(name, description, params):
        def decorator(func):
            func._tool_name = name
            return func

        return decorator

    def fake_create_sdk_mcp_server(name, version, tools):
        return types.SimpleNamespace(
            name=name,
            version=version,
            tools={t._tool_name: t for t in tools},
        )

    monkeypatch.setitem(
        sys.modules,
        "claude_agent_sdk",
        types.SimpleNamespace(
            tool=fake_tool,
            create_sdk_mcp_server=fake_create_sdk_mcp_server,
        ),
    )

    server = create_datasheet_tools_server()

    assert set(server.tools) == {
        "build_datasheet",
        "get_section_text",
        "search_text",
        "inspect_page",
        "extract_table_markdown",
    }

    build_result = asyncio.run(
        server.tools["build_datasheet"](
            {"pdf_source": str(pdf_path), "output_dir": str(tmp_path / "out")}
        )
    )
    assert build_result["is_error"] is False

    section_result = asyncio.run(
        server.tools["get_section_text"]({"start_page": 1, "end_page": 1})
    )
    assert section_result["is_error"] is False

    search_result = asyncio.run(server.tools["search_text"]({"query": "registry"}))
    assert search_result["is_error"] is False

    inspect_result = asyncio.run(server.tools["inspect_page"]({"page": 1}))
    assert inspect_result["is_error"] is False

    # extract_table_markdown requires pymupdf4llm; verify graceful error
    table_md_result = asyncio.run(server.tools["extract_table_markdown"]({"page": 1}))
    # Will be is_error=True if pymupdf4llm not installed, False if it is
    assert isinstance(table_md_result["is_error"], bool)


def test_mcp_build_datasheet_omits_output_dir(monkeypatch, tmp_path):
    """When the MCP caller omits output_dir, the library default is used."""
    import asyncio

    pdf_path = tmp_path / "test.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    writer = pymupdf.TextWriter(page.rect)
    writer.append((72, 72), "Default output_dir test")
    writer.write_text(page)
    doc.save(str(pdf_path))
    doc.close()

    def fake_tool(name, description, params):
        def decorator(func):
            func._tool_name = name
            return func

        return decorator

    def fake_create_sdk_mcp_server(name, version, tools):
        return types.SimpleNamespace(
            name=name, version=version, tools={t._tool_name: t for t in tools}
        )

    monkeypatch.setitem(
        sys.modules,
        "claude_agent_sdk",
        types.SimpleNamespace(
            tool=fake_tool, create_sdk_mcp_server=fake_create_sdk_mcp_server
        ),
    )
    # Pin the resolver so the test stays hermetic
    pinned = tmp_path / "resolved-out"
    monkeypatch.setenv("DATASHEETINDEX_OUTPUT_DIR", str(pinned))

    server = create_datasheet_tools_server()
    result = asyncio.run(server.tools["build_datasheet"]({"pdf_source": str(pdf_path)}))
    assert result["is_error"] is False
    assert pinned.exists() and any(pinned.iterdir())


@pytest.mark.real_pdf
def test_real_pdf_tools():
    """DatasheetTools should work with the real test PDF."""
    if not TLE9350_PATH.exists():
        pytest.skip("Test PDF not found")

    tools = DatasheetTools(str(TLE9350_PATH))
    result = tools.inspect_page(page=1)
    tools.close()

    assert result[0]["type"] == "image"
    assert len(result[0]["data"]) > 0


def test_datasheet_tools_supports_url_source(monkeypatch):
    from tests.conftest import DummyDoc, FakeResponse

    opened_paths: list[str] = []

    def fake_urlopen(url: str, timeout: int):
        assert url == "https://example.com/test.pdf"
        assert timeout > 0
        return FakeResponse(b"%PDF-1.7\nmock")

    def fake_open(path: str):
        opened_paths.append(path)
        return DummyDoc()

    monkeypatch.setattr("datasheetindex.index.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("datasheetindex.index.pymupdf.open", fake_open)

    tools = DatasheetTools("https://example.com/test.pdf")
    _ = tools.doc
    assert len(opened_paths) == 1
    temp_path = Path(opened_paths[0])
    assert temp_path.exists()

    tools.close()
    assert tools._doc is None
    assert not temp_path.exists()
