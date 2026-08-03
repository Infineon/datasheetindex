"""Local MCP server entry point for datasheetindex.

This server is a thin adapter over the framework-neutral tool session
(:func:`datasheetindex.tools.defs.create_datasheet_tool_session`): it serves the
same five tools -- with the same names, descriptions, and JSON schemas -- that the
Claude Agent SDK surface (``create_datasheet_tools_server``) and non-SDK hosts
get. There is a single source of truth for the tool definitions; this module only
wires them onto MCP transports (stdio / streamable-http / sse).
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import inspect
import sys
from typing import Any

from datasheetindex._version import package_version
from datasheetindex.core.engine import layout_engine
from datasheetindex.tools.defs import (
    DatasheetToolSession,
    create_datasheet_tool_session,
)


def _load_mcp_modules() -> tuple[Any, Any]:
    """Import the low-level MCP server + types, or raise a helpful ImportError."""
    try:
        lowlevel = importlib.import_module("mcp.server.lowlevel")
        types_module = importlib.import_module("mcp.types")
    except ImportError:
        raise ImportError(
            "mcp is required for local MCP server support. "
            "Install it with: uv sync --extra mcp"
        ) from None
    return lowlevel.Server, types_module


def _preload_layout_model() -> None:
    """Import pymupdf4llm to trigger ONNX model loading before serving.

    The layout model takes ~2s to load. Doing this eagerly at server start
    avoids a long GIL-holding pause on the first extract_table_markdown call,
    which can cause MCP client timeouts.

    Routed through layout_engine() so that engine.py stays the only place in
    the package that imports pymupdf4llm, and the hook is only ever installed
    under the lock. This runs before serving begins, so it cannot race today.
    """
    with contextlib.suppress(ImportError), layout_engine():
        pass  # optional dependency; extract_table_markdown reports the error


def _envelope_to_content(envelope: dict[str, Any], types_module: Any) -> list[Any]:
    """Translate a neutral ``{"content": [...]}`` envelope into MCP content blocks."""
    blocks: list[Any] = []
    for block in envelope.get("content", []):
        if block.get("type") == "text":
            blocks.append(types_module.TextContent(type="text", text=block["text"]))
        elif block.get("type") == "image":
            blocks.append(
                types_module.ImageContent(
                    type="image",
                    data=block["data"],
                    mimeType=block["mime_type"],
                )
            )
    return blocks


#: Served verbatim on both mcp majors, so it lives here rather than inline in one
#: branch -- an agent's first impression of this server must not depend on which
#: version of the SDK the resolver happened to pick.
SERVER_INSTRUCTIONS = (
    "Extract technical parameters from PDF datasheets. Call "
    "build_datasheet FIRST with a pdf_source (local path or URL) to load "
    "a document -- it returns the full enriched ToC for navigation "
    "planning. Then use get_section_text to read page ranges, search_text "
    "to locate keywords, inspect_page for visual content, and "
    "extract_table_markdown for a clean Markdown table when "
    "get_section_text shows a garbled one."
)


def _text_error(message: str) -> dict[str, Any]:
    """A neutral error envelope carrying one text block."""
    return {"content": [{"type": "text", "text": message}], "is_error": True}


def _schema_violation(schema: dict[str, Any], arguments: dict[str, Any]) -> str | None:
    """The first schema violation in ``arguments``, or ``None`` if it validates.

    Mirrors what mcp 1.x's ``@server.call_tool()`` wrapper does before dispatch
    (``jsonschema.validate`` against the tool's ``inputSchema``, reported as
    ``Input validation error: {e.message}``), so the 2.x branch tells an agent
    the same thing about the same bad call. ``jsonschema`` is not a declared
    dependency of this package but ships with ``mcp`` on both majors; the import
    is local because this module is importable without the ``[mcp]`` extra.
    """
    import jsonschema

    try:
        jsonschema.validate(instance=arguments, schema=schema)
    except jsonschema.ValidationError as exc:
        return exc.message
    return None


def _build_mcp_server(
    session: DatasheetToolSession, server_cls: Any, types_module: Any
):
    """Register the neutral tool session's defs onto a low-level MCP ``Server``.

    Supports **both mcp majors** from one code path, because our two consumers
    disagree about which one to install: ``claude-agent-sdk`` still requires
    ``mcp<2``, while an unpinned ``uvx --from datasheetindex[mcp]`` -- how the
    registry entry installs -- resolves ``mcp>=2``. Only one ``mcp`` can be
    present, so the branch is on the *installed* API rather than on a constraint
    we could declare. Pinning either way would break the other consumer; see the
    ``[mcp]`` extra's comment in ``pyproject.toml``.

    The split is confined to handler registration. mcp 1.x registers via
    ``@server.list_tools()`` / ``@server.call_tool()`` decorators and takes
    unwrapped returns; mcp 2.x takes ``on_list_tools`` / ``on_call_tool``
    constructor callables that return ``ListToolsResult`` / ``CallToolResult``.
    ``run()``, ``create_initialization_options()`` and ``_envelope_to_content``
    are identical across both, and ``mimeType=`` is still accepted as an alias in
    2.x (where the field is ``mime_type``).

    **The 2.x callable is not wrapped by the framework, so guards 1.x applied for
    us have to be applied here.** ``@server.call_tool()`` validated ``arguments``
    against ``inputSchema`` before dispatch and converted a raised exception into
    ``isError``; the 2.x callable gets the raw request and its exceptions escape
    as JSON-RPC *protocol* errors. Both are reproduced explicitly in
    ``_on_call_tool`` -- schema validation via ``_schema_violation``, and an
    unknown name returned rather than raised -- so the observable behaviour of a
    bad call is identical on both majors. Do not "simplify" either away.

    Drop the 1.x branch once ``claude-agent-sdk`` lifts its ``mcp<2`` pin -- at
    which point ``pyproject.toml`` can require ``mcp[cli]>=2``.
    """
    by_name = {d.name: d for d in session.defs}

    def _tool_models() -> list[Any]:
        return [
            types_module.Tool(
                name=d.name,
                description=d.description,
                inputSchema=d.input_schema,
            )
            for d in session.defs
        ]

    async def _invoke(name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
        definition = by_name.get(name)
        if definition is None:
            raise ValueError(f"unknown tool: {name}")
        return await definition.handler(arguments or {})

    def _error_message(name: str, envelope: dict[str, Any]) -> str:
        content = envelope.get("content") or []
        # Error envelopes are text-only today; fall back defensively if a
        # future handler surfaces a non-text first block.
        return (content[0].get("text") if content else None) or f"{name} failed"

    # mcp 1.x is identified by the decorator method whose absence is exactly what
    # breaks this module on 2.x -- so the probe names the thing it is guarding.
    if hasattr(server_cls, "list_tools"):
        server = server_cls(
            name="datasheetindex",
            version=package_version(),
            instructions=SERVER_INSTRUCTIONS,
        )

        @server.list_tools()
        async def _list_tools() -> list[Any]:
            return _tool_models()

        @server.call_tool()
        async def _call_tool(name: str, arguments: dict[str, Any] | None) -> list[Any]:
            envelope = await _invoke(name, arguments)
            if envelope.get("is_error"):
                # 1.x has no is_error on the handler's return, so a tool-level
                # failure is signalled by raising; the framework sets isError.
                raise RuntimeError(_error_message(name, envelope))
            return _envelope_to_content(envelope, types_module)

        return server

    # Positive probe for 2.x rather than a bare `else`. The extra is
    # deliberately unbounded, so a future major dropping this API arrives
    # unannounced -- exactly how 2.0.0 did -- and falling through would raise an
    # unexpected-keyword TypeError naming neither the version nor the fix.
    if "on_list_tools" not in inspect.signature(server_cls.__init__).parameters:
        raise RuntimeError(
            "unsupported mcp version: mcp.server.lowlevel.Server exposes neither "
            "the 1.x decorator API (list_tools) nor the 2.x constructor handlers "
            "(on_list_tools). Pin mcp to a supported release, or teach "
            "_build_mcp_server the new API."
        )

    async def _on_list_tools(_ctx: Any, _params: Any) -> Any:
        return types_module.ListToolsResult(tools=_tool_models())

    async def _on_call_tool(_ctx: Any, params: Any) -> Any:
        # 2.x hands us the raw request: there is no framework wrapper between
        # the wire and this callable, so the two guards mcp 1.x applies before
        # dispatch have to be applied here or they are simply lost. Both must
        # produce a *result*, never an exception -- an escaping error becomes a
        # JSON-RPC protocol error the agent cannot read and recover from, and
        # upstream emits it with the invalid error code 0.
        definition = by_name.get(params.name)
        arguments = params.arguments or {}
        if definition is None:
            envelope = _text_error(f"unknown tool: {params.name}")
        else:
            violation = _schema_violation(definition.input_schema, arguments)
            if violation is not None:
                envelope = _text_error(f"Input validation error: {violation}")
            else:
                envelope = await definition.handler(arguments)

        content = _envelope_to_content(envelope, types_module)
        is_error = bool(envelope.get("is_error"))
        if is_error and not content:
            # Same defence as the 1.x branch's ``_error_message`` fallback: an
            # error result carrying no text at all says less than "it failed".
            content = [
                types_module.TextContent(
                    type="text", text=_error_message(params.name, envelope)
                )
            ]
        # 2.x carries the flag on the result, so the handler's own error text is
        # returned verbatim instead of being reconstructed from an exception.
        return types_module.CallToolResult(content=content, is_error=is_error)

    return server_cls(
        name="datasheetindex",
        version=package_version(),
        instructions=SERVER_INSTRUCTIONS,
        on_list_tools=_on_list_tools,
        on_call_tool=_on_call_tool,
    )


class LocalMcpServer:
    """A running-configurable local MCP server over the neutral datasheet tools.

    Wraps the low-level MCP ``Server`` plus the tool session's lifecycle, and
    exposes ``run(transport=...)`` for stdio / streamable-http / sse. The bound
    document is closed when serving stops.
    """

    def __init__(
        self,
        mcp_server: Any,
        session: DatasheetToolSession,
        host: str,
        port: int,
        streamable_http_path: str,
    ) -> None:
        self.mcp_server = mcp_server
        self.session = session
        self.host = host
        self.port = port
        self.streamable_http_path = streamable_http_path

    def run(self, transport: str = "stdio") -> None:
        """Serve over the given transport, closing the session on shutdown."""
        import anyio

        _preload_layout_model()
        try:
            if transport == "stdio":
                anyio.run(self._serve_stdio)
            elif transport == "streamable-http":
                self._serve_streamable_http()
            elif transport == "sse":
                self._serve_sse()
            else:
                raise ValueError(f"unsupported transport: {transport!r}")
        finally:
            self.session.close()

    def _init_options(self) -> Any:
        return self.mcp_server.create_initialization_options()

    async def _serve_stdio(self) -> None:
        from mcp.server.stdio import stdio_server

        async with stdio_server() as (read_stream, write_stream):
            await self.mcp_server.run(read_stream, write_stream, self._init_options())

    def _serve_streamable_http(self) -> None:
        import uvicorn
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        from starlette.applications import Starlette
        from starlette.routing import Mount

        manager = StreamableHTTPSessionManager(app=self.mcp_server)

        @contextlib.asynccontextmanager
        async def lifespan(_app: Any):
            async with manager.run():
                yield

        # manager.handle_request is itself a valid ASGI callable.
        app = Starlette(
            routes=[Mount(self.streamable_http_path, app=manager.handle_request)],
            lifespan=lifespan,
        )
        uvicorn.run(app, host=self.host, port=self.port)

    def _build_sse_app(self) -> Any:
        """Build the Starlette app for the SSE transport (extracted for testing)."""
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.responses import Response
        from starlette.routing import Mount, Route

        sse = SseServerTransport("/messages/")

        async def handle_sse(request: Any) -> Any:
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as (read_stream, write_stream):
                await self.mcp_server.run(
                    read_stream, write_stream, self._init_options()
                )
            return Response()

        return Starlette(
            routes=[
                Route("/sse", endpoint=handle_sse),
                Mount("/messages/", app=sse.handle_post_message),
            ]
        )

    def _serve_sse(self) -> None:
        import uvicorn

        uvicorn.run(self._build_sse_app(), host=self.host, port=self.port)


def create_local_mcp_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    streamable_http_path: str = "/mcp",
) -> LocalMcpServer:
    """Create a local MCP server serving the neutral datasheet tools.

    The server starts without a bound PDF. Call ``build_datasheet`` with a
    ``pdf_source`` (local path or URL) to load a datasheet. Calling it again with
    a different source replaces the current document.
    """
    server_cls, types_module = _load_mcp_modules()
    session = create_datasheet_tool_session()
    mcp_server = _build_mcp_server(session, server_cls, types_module)
    return LocalMcpServer(mcp_server, session, host, port, streamable_http_path)


def run_mcp_server(
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
    streamable_http_path: str = "/mcp",
) -> None:
    """Run the local MCP server."""
    server = create_local_mcp_server(
        host=host,
        port=port,
        streamable_http_path=streamable_http_path,
    )
    server.run(transport=transport)


def _add_mcp_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the MCP transport options shared by both entry points.

    `datasheetindex mcp` and the `datasheetindex-mcp-server` console script are
    two doors to the same server, so their options must not drift apart. The
    registry entry depends on the defaults here.
    """
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="MCP transport to expose (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind for HTTP-based transports",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind for HTTP-based transports",
    )
    parser.add_argument(
        "--streamable-http-path",
        default="/mcp",
        help="Path to expose when using streamable-http transport",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="datasheetindex-mcp-server",
        description=(
            "Run datasheetindex as a local MCP server. "
            "Use build_datasheet to load a PDF source."
        ),
    )
    _add_mcp_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the local MCP server and return an exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        run_mcp_server(
            transport=args.transport,
            host=args.host,
            port=args.port,
            streamable_http_path=args.streamable_http_path,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def main_cli() -> None:
    """Console-script entry point."""
    raise SystemExit(main())


if __name__ == "__main__":
    main_cli()
