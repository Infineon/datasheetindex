"""Tests for the local MCP server entry point."""

from __future__ import annotations

import importlib
import sys
import types

import pytest


def _install_fake_mcp(monkeypatch):
    class _FakeContext:
        @classmethod
        def __class_getitem__(cls, _item):
            return cls

    class _FakeImageContent:
        def __init__(self, *, type: str, data: str, mimeType: str) -> None:
            self.type = type
            self.data = data
            self.mimeType = mimeType

    class _FakeCallToolResult:
        def __init__(self, *, content: list[object]) -> None:
            self.content = content

    class _FakeFastMCP:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs
            self.registered_tools: dict[str, dict[str, object]] = {}
            self.run_calls: list[tuple[str, str | None]] = []

        def tool(self, name=None, description=None, **_kwargs):
            def decorator(func):
                tool_name = name or func.__name__
                self.registered_tools[tool_name] = {
                    "func": func,
                    "description": description,
                }
                return func

            return decorator

        def run(self, transport="stdio", mount_path=None) -> None:
            self.run_calls.append((transport, mount_path))

    monkeypatch.setitem(
        sys.modules,
        "mcp.server.fastmcp",
        types.SimpleNamespace(FastMCP=_FakeFastMCP, Context=_FakeContext),
    )
    monkeypatch.setitem(
        sys.modules,
        "mcp.server.session",
        types.SimpleNamespace(ServerSession=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "mcp.types",
        types.SimpleNamespace(
            CallToolResult=_FakeCallToolResult,
            ImageContent=_FakeImageContent,
        ),
    )
    return _FakeCallToolResult, _FakeImageContent


def test_create_local_mcp_server_registers_inspect_page(monkeypatch):
    _FakeCallToolResult, _FakeImageContent = _install_fake_mcp(monkeypatch)

    from datasheetindex.mcp_server import create_local_mcp_server

    server = create_local_mcp_server(
        host="0.0.0.0",
        port=9001,
        streamable_http_path="/custom-mcp",
    )

    assert server.kwargs["host"] == "0.0.0.0"
    assert server.kwargs["port"] == 9001
    assert server.kwargs["streamable_http_path"] == "/custom-mcp"
    assert set(server.registered_tools) == {
        "build_datasheet",
        "get_section_text",
        "inspect_page",
        "search_text",
        "extract_table_markdown",
        "locate_text",
    }

    calls: list[tuple[int, dict[str, float] | None, int | None, str]] = []
    search_calls: list[tuple[str, int | None, bool, int]] = []

    fake_tools = types.SimpleNamespace(
        pdf_path="sample.pdf",
        get_section_text=lambda start_page, end_page: (
            f"--- PAGE {start_page} ---\npage {start_page} text"
        ),
        search_text=lambda query, page=None, case_sensitive=False, max_results=20: (
            search_calls.append((query, page, case_sensitive, max_results))
            or [{"page": 2, "start": 0, "end": 3, "snippet": "foo"}]
        ),
        inspect_page=lambda page, region=None, dpi=None, detail="medium": (
            calls.append((page, region, dpi, detail))
            or [{"type": "image", "data": "Zm9v", "mime_type": "image/png"}]
        ),
        extract_table_markdown=lambda page: f"| col1 | col2 |\n| p{page} | val |",
        locate_text=lambda query, page=None, max_results=20: [
            {
                "page": 1,
                "match_method": "search_for",
                "page_width": 612.0,
                "page_height": 792.0,
                "region": {
                    "pct": {"top": 0.0, "bottom": 0.1, "left": 0.0, "right": 0.1},
                    "points": {"x0": 0.0, "y0": 0.0, "x1": 61.2, "y1": 79.2},
                },
                "boxes": [
                    {
                        "pct": {"top": 0.0, "bottom": 0.1, "left": 0.0, "right": 0.1},
                        "points": {"x0": 0.0, "y0": 0.0, "x1": 61.2, "y1": 79.2},
                    }
                ],
            }
        ],
    )
    # Pre-load tools in server context (simulates a prior build_datasheet call)
    server_ctx = types.SimpleNamespace(tools=fake_tools)
    ctx = types.SimpleNamespace(
        request_context=types.SimpleNamespace(lifespan_context=server_ctx)
    )

    tool = server.registered_tools["inspect_page"]["func"]
    result = tool(page=2, region={"top": 0.1}, dpi=200, ctx=ctx)
    section_text_result = server.registered_tools["get_section_text"]["func"](
        start_page=2, end_page=2, ctx=ctx
    )
    search_result = server.registered_tools["search_text"]["func"](
        query="foo",
        page=2,
        case_sensitive=True,
        max_results=1,
        ctx=ctx,
    )

    assert calls == [(2, {"top": 0.1}, 200, "medium")]
    assert section_text_result == {
        "start_page": 2,
        "end_page": 2,
        "text": "--- PAGE 2 ---\npage 2 text",
    }
    assert search_calls == [("foo", 2, True, 1)]
    assert search_result["results"][0]["snippet"] == "foo"

    # A list-valued query is forwarded unchanged to the tools layer.
    server.registered_tools["search_text"]["func"](query=["foo", "bar"], ctx=ctx)
    assert search_calls[-1] == (["foo", "bar"], None, False, 20)

    import asyncio

    table_md_result = asyncio.run(
        server.registered_tools["extract_table_markdown"]["func"](page=2, ctx=ctx)
    )
    assert table_md_result == {
        "page": 2,
        "markdown": "| col1 | col2 |\n| p2 | val |",
    }
    locate_result = server.registered_tools["locate_text"]["func"](
        query="Hello", page=1, ctx=ctx
    )
    assert locate_result["query"] == "Hello"
    assert locate_result["results"][0]["match_method"] == "search_for"

    assert isinstance(result, _FakeCallToolResult)
    assert len(result.content) == 1
    image = result.content[0]
    assert isinstance(image, _FakeImageContent)
    assert image.type == "image"
    assert image.data == "Zm9v"
    assert image.mimeType == "image/png"


def test_create_local_mcp_server_raises_without_mcp(monkeypatch):
    from datasheetindex import mcp_server

    original_import_module = importlib.import_module

    def _fake_import_module(name: str, package=None):
        if name.startswith("mcp"):
            raise ImportError("missing mcp")
        return original_import_module(name, package)

    monkeypatch.setattr(mcp_server.importlib, "import_module", _fake_import_module)

    with pytest.raises(ImportError, match="uv sync --extra mcp"):
        mcp_server.create_local_mcp_server()


def test_run_mcp_server_invokes_fastmcp_run(monkeypatch):
    from datasheetindex import mcp_server

    class _FakeServer:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str | None]] = []

        def run(self, transport="stdio", mount_path=None) -> None:
            self.calls.append((transport, mount_path))

    fake_server = _FakeServer()

    def _fake_create(*args, **kwargs):
        _ = args, kwargs
        return fake_server

    monkeypatch.setattr(mcp_server, "create_local_mcp_server", _fake_create)

    mcp_server.run_mcp_server(transport="streamable-http")

    assert fake_server.calls == [("streamable-http", None)]


def test_main_runs_server(monkeypatch):
    from datasheetindex import mcp_server

    calls: list[tuple[str, str, int, str]] = []

    def _fake_run(transport, host, port, streamable_http_path):
        calls.append((transport, host, port, streamable_http_path))

    monkeypatch.setattr(mcp_server, "run_mcp_server", _fake_run)

    exit_code = mcp_server.main(
        [
            "--transport",
            "streamable-http",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--streamable-http-path",
            "/inspect",
        ]
    )

    assert exit_code == 0
    assert calls == [("streamable-http", "0.0.0.0", 9000, "/inspect")]


def test_main_reports_error(monkeypatch, capsys):
    from datasheetindex import mcp_server

    def _raise(*args, **kwargs):
        _ = args, kwargs
        raise ImportError("boom")

    monkeypatch.setattr(mcp_server, "run_mcp_server", _raise)

    exit_code = mcp_server.main([])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error: boom" in captured.err


def test_inspect_page_tool_without_datasheet_raises(monkeypatch):
    _install_fake_mcp(monkeypatch)

    from datasheetindex.mcp_server import create_local_mcp_server

    server = create_local_mcp_server()
    server_ctx = types.SimpleNamespace(tools=None)
    ctx = types.SimpleNamespace(
        request_context=types.SimpleNamespace(lifespan_context=server_ctx)
    )
    func = server.registered_tools["inspect_page"]["func"]
    with pytest.raises(RuntimeError, match="No datasheet loaded"):
        func(page=1, ctx=ctx)


def test_locate_text_tool_without_datasheet_raises(monkeypatch):
    _install_fake_mcp(monkeypatch)

    from datasheetindex.mcp_server import create_local_mcp_server

    server = create_local_mcp_server()
    server_ctx = types.SimpleNamespace(tools=None)
    ctx = types.SimpleNamespace(
        request_context=types.SimpleNamespace(lifespan_context=server_ctx)
    )
    func = server.registered_tools["locate_text"]["func"]
    with pytest.raises(RuntimeError, match="No datasheet loaded"):
        func(query="anything", page=1, ctx=ctx)
