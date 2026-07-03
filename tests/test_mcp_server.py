"""Tests for the local MCP server entry point.

The server is a thin adapter over the framework-neutral tool session, so the
behavioural tests drive the real low-level MCP ``Server`` handlers (guarded by
``importorskip`` since ``mcp`` is an optional extra). The ImportError and CLI
tests don't need ``mcp`` installed.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import types

import pymupdf
import pytest


def _make_pdf(path, text="Supply voltage 4.5V to 5.5V"):
    doc = pymupdf.open()
    page = doc.new_page()
    writer = pymupdf.TextWriter(page.rect)
    writer.append((72, 72), text)
    writer.write_text(page)
    doc.save(str(path))
    doc.close()


def _drive(server, request):
    """Invoke a low-level Server request handler and return the result payload."""
    handler = server.request_handlers[type(request)]
    return asyncio.run(handler(request)).root


def _list_tools(server, types):
    return _drive(server, types.ListToolsRequest(method="tools/list")).tools


def _call(server, types, name, arguments):
    request = types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name=name, arguments=arguments),
    )
    return _drive(server, request)


def test_create_local_mcp_server_stores_config():
    pytest.importorskip("mcp")
    from datasheetindex.mcp_server import create_local_mcp_server

    server = create_local_mcp_server(
        host="0.0.0.0", port=9001, streamable_http_path="/custom-mcp"
    )
    assert server.host == "0.0.0.0"
    assert server.port == 9001
    assert server.streamable_http_path == "/custom-mcp"


def test_serves_neutral_tool_defs_verbatim():
    """The local server exposes exactly the neutral defs' names/descriptions/schemas."""
    types = pytest.importorskip("mcp.types")
    from datasheetindex.mcp_server import create_local_mcp_server
    from datasheetindex.tools.defs import create_datasheet_tool_defs

    server = create_local_mcp_server()
    served = {t.name: t for t in _list_tools(server.mcp_server, types)}
    neutral = {d.name: d for d in create_datasheet_tool_defs()}

    assert set(served) == set(neutral)
    for name, d in neutral.items():
        assert served[name].description == d.description
        assert served[name].inputSchema == d.input_schema


def test_call_tool_end_to_end(tmp_path):
    types = pytest.importorskip("mcp.types")
    from datasheetindex.mcp_server import create_local_mcp_server

    pdf = tmp_path / "t.pdf"
    _make_pdf(pdf)
    server = create_local_mcp_server().mcp_server

    build = _call(
        server,
        types,
        "build_datasheet",
        {"pdf_source": str(pdf), "output_dir": str(tmp_path / "out")},
    )
    assert build.isError is False
    assert build.content[0].type == "text"

    section = _call(server, types, "get_section_text", {"start_page": 1, "end_page": 1})
    assert section.isError is False
    assert "Supply voltage" in json.loads(section.content[0].text)["text"]

    search = _call(server, types, "search_text", {"query": "5.5v"})
    assert search.isError is False
    assert json.loads(search.content[0].text)["results"][0]["page"] == 1

    # list-valued query is forwarded unchanged
    multi = _call(server, types, "search_text", {"query": ["5.5v", "voltage"]})
    assert multi.isError is False

    locate = _call(server, types, "locate_text", {"query": "Supply", "page": 1})
    assert locate.isError is False
    located = json.loads(locate.content[0].text)["results"][0]
    assert located["match_method"] == "search_for"

    image = _call(server, types, "inspect_page", {"page": 1})
    assert image.isError is False
    assert image.content[0].type == "image"
    assert image.content[0].mimeType == "image/png"


def test_call_tool_before_build_is_tool_error():
    types = pytest.importorskip("mcp.types")
    from datasheetindex.mcp_server import create_local_mcp_server

    server = create_local_mcp_server().mcp_server
    result = _call(server, types, "search_text", {"query": "anything"})
    assert result.isError is True
    assert "No datasheet loaded" in result.content[0].text


def test_run_dispatches_transport_and_closes_session(monkeypatch):
    pytest.importorskip("mcp")
    from datasheetindex.mcp_server import create_local_mcp_server

    server = create_local_mcp_server()
    calls: list[str] = []
    # stdio is served via `anyio.run(self._serve_stdio)`; stub anyio.run so no
    # real event loop starts. http/sse call their runners directly.
    monkeypatch.setattr("datasheetindex.mcp_server._preload_layout_model", lambda: None)
    monkeypatch.setattr("anyio.run", lambda fn, *a, **k: calls.append("stdio"))
    monkeypatch.setattr(server, "_serve_streamable_http", lambda: calls.append("http"))
    monkeypatch.setattr(server, "_serve_sse", lambda: calls.append("sse"))
    closed: list[bool] = []
    server.session.close()  # the real session must close cleanly when unbound
    monkeypatch.setattr(
        server, "session", types.SimpleNamespace(close=lambda: closed.append(True))
    )

    server.run(transport="streamable-http")
    server.run(transport="sse")
    server.run(transport="stdio")

    assert calls == ["http", "sse", "stdio"]
    assert closed == [True, True, True]  # session closed after each transport run

    with pytest.raises(ValueError, match="unsupported transport"):
        server.run(transport="bogus")


def test_run_mcp_server_builds_and_runs(monkeypatch):
    from datasheetindex import mcp_server

    class _FakeServer:
        def __init__(self):
            self.calls: list[str] = []

        def run(self, transport="stdio"):
            self.calls.append(transport)

    fake = _FakeServer()
    monkeypatch.setattr(mcp_server, "create_local_mcp_server", lambda **kwargs: fake)

    mcp_server.run_mcp_server(transport="streamable-http")
    assert fake.calls == ["streamable-http"]


def test_create_local_mcp_server_raises_without_mcp(monkeypatch):
    from datasheetindex import mcp_server

    original_import_module = importlib.import_module

    def _fake_import_module(name, package=None):
        if name.startswith("mcp"):
            raise ImportError("missing mcp")
        return original_import_module(name, package)

    monkeypatch.setattr(mcp_server.importlib, "import_module", _fake_import_module)

    with pytest.raises(ImportError, match="uv sync --extra mcp"):
        mcp_server.create_local_mcp_server()


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
