"""Local MCP server entry point for datasheetindex.

This server is a thin adapter over the framework-neutral tool session
(:func:`datasheetindex.tools.defs.create_datasheet_tool_session`): it serves the
same six tools -- with the same names, descriptions, and JSON schemas -- that the
Claude Agent SDK surface (``create_datasheet_tools_server``) and non-SDK hosts
get. There is a single source of truth for the tool definitions; this module only
wires them onto MCP transports (stdio / streamable-http / sse).
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
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


def _build_mcp_server(
    session: DatasheetToolSession, server_cls: Any, types_module: Any
):
    """Register the neutral tool session's defs onto a low-level MCP ``Server``."""
    by_name = {d.name: d for d in session.defs}
    server = server_cls(
        name="datasheetindex",
        version=package_version(),
        instructions=(
            "Extract technical parameters from PDF datasheets. Call "
            "build_datasheet FIRST with a pdf_source (local path or URL) to load "
            "a document -- it returns the full enriched ToC for navigation "
            "planning. Then use get_section_text to read page ranges, search_text "
            "to locate keywords, locate_text for a string's bounding box, "
            "inspect_page for visual content, and extract_table_markdown for a "
            "clean Markdown table when get_section_text shows a garbled one."
        ),
    )

    @server.list_tools()
    async def _list_tools() -> list[Any]:
        return [
            types_module.Tool(
                name=d.name,
                description=d.description,
                inputSchema=d.input_schema,
            )
            for d in session.defs
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any] | None) -> list[Any]:
        definition = by_name.get(name)
        if definition is None:
            raise ValueError(f"unknown tool: {name}")
        envelope = await definition.handler(arguments or {})
        if envelope.get("is_error"):
            content = envelope.get("content") or []
            # Error envelopes are text-only today; fall back defensively if a
            # future handler surfaces a non-text first block.
            message = (content[0].get("text") if content else None) or f"{name} failed"
            # Surface a tool-level failure as an MCP tool error (isError=True).
            raise RuntimeError(message)
        return _envelope_to_content(envelope, types_module)

    return server


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
