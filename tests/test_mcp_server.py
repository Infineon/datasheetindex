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


# --- Cross-major drivers ------------------------------------------------------
# mcp 1.x registers handlers in a `request_handlers` dict keyed by request type
# and wraps results in a ServerResult root; mcp 2.x exposes them via
# `get_request_handler(method)` and returns the result directly. The helpers
# below hide that so every behavioural test below is written once and asserts
# the same thing on both majors -- which is the whole point of the dual-support
# branch in `_build_mcp_server`.
#
# The 2.x handler is called with a ``None`` request context. That is not a stub
# standing in for behaviour: our handlers close over the tool session and never
# read the context, so passing one would add setup (a live ServerSession) that
# proves nothing. If a handler ever starts using it, this raises rather than
# silently passing.


def _is_v2(server):
    return hasattr(server, "get_request_handler")


def _list_tools(server, types):
    if _is_v2(server):
        entry = server.get_request_handler("tools/list")
        return asyncio.run(entry.handler(None, None)).tools
    request = types.ListToolsRequest(method="tools/list")
    handler = server.request_handlers[type(request)]
    return asyncio.run(handler(request)).root.tools


def _call(server, types, name, arguments):
    if _is_v2(server):
        entry = server.get_request_handler("tools/call")
        params = entry.params_type.model_validate(
            {"name": name, "arguments": arguments}
        )
        return asyncio.run(entry.handler(None, params))
    request = types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name=name, arguments=arguments),
    )
    handler = server.request_handlers[type(request)]
    return asyncio.run(handler(request)).root


def _is_error(result):
    """``CallToolResult.isError`` (mcp 1.x) / ``.is_error`` (mcp 2.x)."""
    return result.is_error if hasattr(result, "is_error") else result.isError


def _mime(block):
    """``ImageContent.mimeType`` (mcp 1.x) / ``.mime_type`` (mcp 2.x)."""
    return block.mime_type if hasattr(block, "mime_type") else block.mimeType


def _input_schema(tool):
    """``Tool.inputSchema`` (mcp 1.x) / ``.input_schema`` (mcp 2.x).

    Only the *attribute* was renamed. ``mcp_server`` still constructs with
    ``inputSchema=``, which 2.x accepts as an alias -- so this asymmetry is
    real and the production code needs no branch for it.
    """
    return tool.input_schema if hasattr(tool, "input_schema") else tool.inputSchema


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
        assert _input_schema(served[name]) == d.input_schema


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
    assert _is_error(build) is False
    assert build.content[0].type == "text"
    manifest = json.loads(build.content[0].text)
    assert manifest["total_pages"] == 1
    assert "toc" in manifest

    section = _call(server, types, "get_section_text", {"start_page": 1, "end_page": 1})
    assert _is_error(section) is False
    assert "Supply voltage" in json.loads(section.content[0].text)["text"]

    search = _call(server, types, "search_text", {"query": "5.5v"})
    assert _is_error(search) is False
    assert json.loads(search.content[0].text)["results"][0]["page"] == 1

    # list-valued query is forwarded unchanged
    multi = _call(server, types, "search_text", {"query": ["5.5v", "voltage"]})
    assert _is_error(multi) is False

    image = _call(server, types, "inspect_page", {"page": 1})
    assert _is_error(image) is False
    assert image.content[0].type == "image"
    assert _mime(image.content[0]) == "image/png"

    # extract_table_markdown goes through the adapter too. It needs the optional
    # pymupdf4llm; with it, isError is False and we get markdown text; without it,
    # the handler returns a clean tool error. Either way the adapter path runs.
    table = _call(server, types, "extract_table_markdown", {"page": 1})
    assert isinstance(_is_error(table), bool)
    assert table.content[0].type == "text"


def test_call_tool_before_build_is_tool_error():
    types = pytest.importorskip("mcp.types")
    from datasheetindex.mcp_server import create_local_mcp_server

    server = create_local_mcp_server().mcp_server
    result = _call(server, types, "search_text", {"query": "anything"})
    assert _is_error(result) is True
    assert "No datasheet loaded" in result.content[0].text


def test_invalid_arguments_get_a_schema_diagnostic():
    """Bad arguments must be named by the schema, not by a Python traceback.

    mcp 1.x validates against ``inputSchema`` inside its ``@server.call_tool()``
    wrapper before our handler runs. The 2.x ``on_call_tool`` callable has no
    such wrapper, so without an explicit check the request reaches the handler
    and the agent is told ``KeyError: 'query'`` instead of what to fix. These
    tools exist to steer an LLM through a datasheet, so the diagnostic *is* the
    product -- the two majors must not disagree about it.
    """
    types = pytest.importorskip("mcp.types")
    from datasheetindex.mcp_server import create_local_mcp_server

    server = create_local_mcp_server().mcp_server

    missing = _call(server, types, "search_text", {})
    assert _is_error(missing) is True
    assert "Input validation error" in missing.content[0].text
    assert "'query' is a required property" in missing.content[0].text

    wrong_type = _call(
        server, types, "get_section_text", {"start_page": "abc", "end_page": 1}
    )
    assert _is_error(wrong_type) is True
    assert "Input validation error" in wrong_type.content[0].text
    assert "'abc' is not of type 'integer'" in wrong_type.content[0].text


def test_unknown_tool_is_a_tool_error_not_a_protocol_error():
    """An unknown name is a recoverable tool error on both majors.

    On 1.x the framework converts our raised ``ValueError`` into
    ``isError=True``. On 2.x nothing catches it, so it escapes as a JSON-RPC
    *protocol* error -- which a 1.x client surfaces as a raised ``McpError``
    rather than a result the agent can read and recover from, and which upstream
    emits with the invalid error code 0. So the 2.x branch must return the error
    rather than raise it.
    """
    types = pytest.importorskip("mcp.types")
    from datasheetindex.mcp_server import create_local_mcp_server

    server = create_local_mcp_server().mcp_server

    result = _call(server, types, "bogus_tool", {})
    assert _is_error(result) is True
    assert "unknown tool: bogus_tool" in result.content[0].text


def _server_over(handler, *, name="boom", schema=None):
    """A real low-level server whose only tool runs ``handler``.

    Builds genuine ``DatasheetToolDef`` / ``DatasheetToolSession`` values rather
    than stand-ins -- both are plain frozen dataclasses, so there is nothing to
    fake -- which keeps the call correctly typed and drives the same
    ``_build_mcp_server`` path the real session takes.
    """
    from datasheetindex.mcp_server import _build_mcp_server, _load_mcp_modules
    from datasheetindex.tools.defs import DatasheetToolDef, DatasheetToolSession

    definition = DatasheetToolDef(
        name=name,
        description="a tool that exists to misbehave",
        input_schema=schema or {"type": "object"},
        handler=handler,
    )
    session = DatasheetToolSession(defs=[definition], close=lambda: None)
    server_cls, types_module = _load_mcp_modules()
    return _build_mcp_server(session, server_cls, types_module)


def test_handler_exception_becomes_a_tool_error():
    """A raising handler must produce a result, not escape as a protocol error.

    This is mcp 1.x's *third* pre-dispatch guard -- ``except Exception as e:
    return self._make_error_result(str(e))`` -- and it is the one most likely to
    matter later: no shipped handler raises today (all five catch internally),
    but a sixth tool, or one line moved outside an existing ``try``, changes
    shape on 2.x only. 1.x's text is exactly ``str(e)``, so matching it keeps the
    two majors byte-identical here as everywhere else.
    """
    types = pytest.importorskip("mcp.types")

    async def _raises(_args):
        raise RuntimeError("handler blew up")

    result = _call(_server_over(_raises), types, "boom", {})
    assert _is_error(result) is True
    assert "handler blew up" in result.content[0].text


def test_url_elicitation_still_propagates():
    """The catch-all must not swallow the protocol's own control-flow signal.

    ``UrlElicitationRequiredError`` is not a failure: the runner converts it into
    a ``-32042`` response that drives URL elicitation. mcp 1.x re-raises it
    immediately *before* its catch-all for exactly this reason, so a blanket
    ``except Exception`` here would silently turn a protocol feature into a tool
    error. Nothing we ship raises it today -- this guards the catch-all added for
    the test above, and it passes both before and after that change by design.
    """
    types = pytest.importorskip("mcp.types")
    from mcp.shared.exceptions import UrlElicitationRequiredError

    async def _elicits(_args):
        raise UrlElicitationRequiredError([], "needs a URL")

    with pytest.raises(UrlElicitationRequiredError):
        _call(_server_over(_elicits), types, "boom", {})


def test_error_envelope_without_content_still_carries_a_message():
    """An error result with no content block says less than "it failed".

    Not reachable through any shipped handler -- every one emits a text block --
    but 1.x's ``_error_message`` fallback predates this branch, so dropping it on
    the 2.x path would recreate the asymmetry this module exists to remove.
    """
    types = pytest.importorskip("mcp.types")

    async def _empty(_args):
        return {"content": [], "is_error": True}

    result = _call(_server_over(_empty), types, "boom", {})
    assert _is_error(result) is True
    assert result.content[0].text == "boom failed"


def test_unrecognised_mcp_api_fails_with_a_message_naming_the_problem():
    """A future major that matches neither API must say so, not TypeError.

    The ``[mcp]`` extra is deliberately unbounded (see ``pyproject.toml``), so
    an ``mcp`` 3.0 dropping ``on_list_tools`` would arrive here unannounced --
    exactly how 2.0.0 arrived. Falling through to the 2.x constructor would
    raise an unexpected-keyword ``TypeError`` naming neither the version nor the
    fix, which is the failure mode that made the original break take so long to
    read.
    """
    from datasheetindex.mcp_server import _build_mcp_server
    from datasheetindex.tools.defs import create_datasheet_tool_session

    class _FutureServer:
        def __init__(self, name, *, version="", instructions=None):
            self.name = name

    # The real session, not a stand-in: it needs no mcp, and typing the
    # parameter honestly is what makes `ty` cover this call site.
    session = create_datasheet_tool_session()
    try:
        with pytest.raises(RuntimeError, match="unsupported mcp version"):
            _build_mcp_server(
                session, _FutureServer, types_module=types.SimpleNamespace()
            )
    finally:
        session.close()


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


def test_sse_app_builds_with_expected_routes():
    """The hand-wired SSE transport constructs with its /sse + /messages/ routes."""
    pytest.importorskip("mcp")
    from datasheetindex.mcp_server import create_local_mcp_server

    app = create_local_mcp_server()._build_sse_app()
    paths = {getattr(r, "path", getattr(r, "path_format", None)) for r in app.routes}
    assert "/sse" in paths
    assert any(p and p.startswith("/messages") for p in paths)


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


def test_envelope_to_content_translation():
    """Pure envelope -> MCP content translation; needs no mcp extra."""
    from datasheetindex.mcp_server import _envelope_to_content

    class _Block:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    fake_types = types.SimpleNamespace(TextContent=_Block, ImageContent=_Block)

    text_out = _envelope_to_content(
        {"content": [{"type": "text", "text": "hi"}], "is_error": False}, fake_types
    )
    assert len(text_out) == 1
    assert text_out[0].type == "text" and text_out[0].text == "hi"

    image_out = _envelope_to_content(
        {
            "content": [{"type": "image", "data": "Zm9v", "mime_type": "image/png"}],
            "is_error": False,
        },
        fake_types,
    )
    assert len(image_out) == 1
    assert image_out[0].type == "image"
    assert image_out[0].data == "Zm9v"
    assert image_out[0].mimeType == "image/png"


# --- Integration: real MCP client <-> server over a live transport ------------
# Marked `integration` (excluded from the fast pre-commit subset) because they
# spawn the server as a subprocess. They back the "verified end-to-end with a
# real MCP client" claim and exercise the actual _serve_* transport wiring.

_SERVER_ARGS = ["-c", "from datasheetindex.mcp_server import main_cli; main_cli()"]
_EXPECTED = {
    "build_datasheet",
    "get_section_text",
    "search_text",
    "inspect_page",
    "extract_table_markdown",
}


@pytest.mark.integration
def test_stdio_roundtrip_with_real_client(tmp_path):
    pytest.importorskip("mcp")
    import sys

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.types import TextContent

    pdf = tmp_path / "t.pdf"
    _make_pdf(pdf)

    async def go():
        params = StdioServerParameters(command=sys.executable, args=_SERVER_ARGS)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert {t.name for t in tools.tools} == _EXPECTED
                build = await session.call_tool(
                    "build_datasheet",
                    {"pdf_source": str(pdf), "output_dir": str(tmp_path / "out")},
                )
                assert _is_error(build) is False
                search = await session.call_tool("search_text", {"query": "5.5v"})
                block = search.content[0]
                assert isinstance(block, TextContent)
                assert json.loads(block.text)["results"][0]["page"] == 1

    asyncio.run(asyncio.wait_for(go(), timeout=60))


def _free_port() -> int:
    """Pick an available localhost port (small TOCTOU window, fine for tests)."""
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.integration
def test_streamable_http_roundtrip_with_real_client(tmp_path):
    pytest.importorskip("mcp")
    import signal
    import subprocess
    import sys

    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    pdf = tmp_path / "t.pdf"
    _make_pdf(pdf)
    port = _free_port()

    proc = subprocess.Popen(
        [
            sys.executable,
            *_SERVER_ARGS,
            "--transport",
            "streamable-http",
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    async def go():
        last: Exception | None = None
        for _ in range(40):
            await asyncio.sleep(0.5)
            try:
                async with streamable_http_client(
                    f"http://127.0.0.1:{port}/mcp"
                ) as streams:
                    # mcp 1.x yields (read, write, get_session_id); 2.x yields
                    # (read, write). Take the two we use either way.
                    read, write = streams[0], streams[1]
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        tools = await session.list_tools()
                        assert {t.name for t in tools.tools} == _EXPECTED
                        build = await session.call_tool(
                            "build_datasheet",
                            {
                                "pdf_source": str(pdf),
                                "output_dir": str(tmp_path / "out"),
                            },
                        )
                        assert _is_error(build) is False
                        return
            except AssertionError:
                # A failed assertion means we reached the server and it answered
                # wrongly. Retrying would bury a real cross-major failure under
                # 20s of polling and then report it as "could not reach server".
                raise
            except Exception as exc:  # server not up yet / transient
                last = exc
        raise AssertionError(f"could not reach streamable-http server: {last!r}")

    try:
        asyncio.run(asyncio.wait_for(go(), timeout=60))
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def test_local_server_reports_installed_version():
    """initialize must advertise the installed version, never a literal.

    A registry entry pinning 0.20.0 while the runtime reports 1.0.0 is a
    mismatch no version-sync guard over server.json can detect.
    """
    pytest.importorskip("mcp")

    from datasheetindex import mcp_server
    from datasheetindex._version import package_version

    server = mcp_server.create_local_mcp_server()

    assert server.mcp_server.version == package_version()
    assert server.mcp_server.version != "1.0.0"
