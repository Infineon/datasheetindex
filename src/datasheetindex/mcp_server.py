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


def _silence_pymupdf_stdout_notice() -> None:
    """Stop PyMuPDF printing its layout advertisement onto the JSON-RPC wire.

    ``find_tables()`` calls ``pymupdf._warn_layout_once()``, which ``print()``s
    "Consider using the pymupdf_layout package..." to **stdout** -- the channel
    the stdio transport carries JSON-RPC on -- once per process, unless
    ``pymupdf.layout`` is importable (it is with the full ``pymupdf`` 2.x wheel,
    not with ``pymupdf-lite``). Our table scan calls ``find_tables()`` on
    every page, and ``engine.classic_tables()`` sets ``pymupdf._get_layout =
    None``, which *satisfies* the notice's trigger rather than avoiding it.

    The parallel scan is already immune: ``structure._subprocess_init`` puts
    worker stdin/stdout on devnull. The sequential path
    (``DATASHEETINDEX_PARALLEL=0``, and the fallback after a pool failure or
    scan timeout) runs in-process, so the line lands on the wire and the client's
    JSON parse fails. Under mcp 2.x it surfaces *after* the last response: that
    transport dup2s a private fd for the wire, so the text sits in
    ``sys.stdout``'s block buffer until the interpreter's final flush writes it
    to the restored fd 1.

    ``no_recommend_layout()`` is PyMuPDF's own supported opt-out (the env var
    ``PYMUPDF_SUGGEST_LAYOUT_ANALYZER=0`` is the other). Preferred over wrapping
    the server in ``redirect_stdout``, which would swallow unrelated output and
    leave the cause in place. This touches ``pymupdf._recommend_layout``, not the
    ``_get_layout`` table-engine hook, so it is not ``core.engine``'s to own.

    Applied for stdio only. On the HTTP transports stdout is not the wire, and
    silencing another library's advice there would be gratuitous.
    """
    import pymupdf

    silence = getattr(pymupdf, "no_recommend_layout", None)
    if silence is not None:
        silence()


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
                    mime_type=block["mime_type"],
                )
            )
    return blocks


#: The server's ``instructions``: an agent's first impression of this server.
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


def _compile_validators(defs: list[Any]) -> dict[str, Any]:
    """One pre-built jsonschema validator per tool, keyed by name.

    Built once at server construction rather than per call.
    ``jsonschema.validate`` -- which is what mcp 1.x's wrapper called, and what
    this module called first -- re-checks the *schema against its metaschema* on
    every invocation: measured on
    ``build_datasheet``'s schema that is 4545us per call, of which 5219us is
    ``check_schema`` alone, against 13us for a pre-built validator. Roughly 340x,
    and for the cheap tools the throwaway work dwarfed the tool itself (one real
    ``search_text`` call is ~803us). Reporting is unchanged because the message
    still comes from ``ValidationError.message``.

    ``check_schema`` still runs, once, here -- so a malformed tool schema fails
    loudly at construction instead of turning every request into a ``SchemaError``.

    ``jsonschema`` is a core (non-extra) requirement of ``mcp`` and is declared
    by the ``[mcp]`` extra; the import is local because this module is
    importable without that extra.
    """
    from jsonschema.validators import validator_for

    validators: dict[str, Any] = {}
    for definition in defs:
        validator_cls = validator_for(definition.input_schema)
        validator_cls.check_schema(definition.input_schema)
        validators[definition.name] = validator_cls(definition.input_schema)
    return validators


def _schema_violation(validator: Any, arguments: dict[str, Any]) -> str | None:
    """The first schema violation in ``arguments``, or ``None`` if it validates.

    Mirrors what mcp 1.x's ``@server.call_tool()`` wrapper did before dispatch,
    reported with its exact ``Input validation error: {message}`` wording, so an
    agent is told the same thing about the same bad call as it always was.
    """
    import jsonschema

    try:
        validator.validate(arguments)
    except jsonschema.ValidationError as exc:
        return exc.message
    return None


def _passthrough_exceptions() -> tuple[type[BaseException], ...]:
    """Exceptions the runner interprets, which a catch-all must not swallow.

    ``UrlElicitationRequiredError`` is not a failure -- the runner turns it into a
    ``-32042`` response that drives URL elicitation -- which is why mcp 1.x's
    wrapper re-raised it immediately *before* its own ``except Exception``. Reproducing
    the catch-all without reproducing this passthrough would silently convert a
    protocol feature into a tool error.
    """
    try:
        from mcp.shared.exceptions import UrlElicitationRequiredError
    except ImportError:  # pragma: no cover - present on every supported mcp
        return ()
    return (UrlElicitationRequiredError,)


def _build_mcp_server(
    session: DatasheetToolSession, server_cls: Any, types_module: Any
):
    """Register the neutral tool session's defs onto a low-level MCP ``Server``.

    mcp 2.x takes ``on_list_tools`` / ``on_call_tool`` constructor callables that
    return ``ListToolsResult`` / ``CallToolResult``. (1.x's ``@server.list_tools()``
    / ``@server.call_tool()`` decorators were served alongside until 0.41.0.)

    **The callable is not wrapped by the framework, so three guards are ours to
    apply.** 1.x's ``@server.call_tool()`` (a) jsonschema-validated ``arguments``
    against ``inputSchema`` before dispatch, (b) turned an unknown name into a
    tool error, and (c) ended in ``except Exception as e: return
    self._make_error_result(str(e))``. The 2.x callable gets the raw request, and
    anything escaping it becomes a JSON-RPC *protocol* error rather than a result
    the agent can read. All three are reproduced in ``_on_call_tool``, whose
    ``try``/``except`` enforces "always return a result" structurally rather
    than leaving it to be maintained by counting guards -- which is how (c) was
    missed the first time. Do not "simplify" any of it away.
    """
    by_name = {d.name: d for d in session.defs}
    validators = _compile_validators(session.defs)
    passthrough = _passthrough_exceptions()

    # Positive probe rather than letting the constructor fail. The extra has no
    # upper bound, so a future major dropping this API arrives unannounced --
    # exactly how 2.0.0 did -- and the constructor would raise an
    # unexpected-keyword TypeError naming neither the version nor the fix.
    # An unreadable signature (a decorator without functools.wraps, a C-level
    # __init__) is not evidence of an unsupported version, so it proceeds rather
    # than claiming one -- a wrong "unsupported mcp version" would be worse than
    # the TypeError it replaced, which was cryptic but never lied.
    try:
        accepted = inspect.signature(server_cls.__init__).parameters
    except (ValueError, TypeError):
        accepted = None
    if accepted is not None and "on_list_tools" not in accepted:
        raise RuntimeError(
            "unsupported mcp version: mcp.server.lowlevel.Server does not take the "
            "2.x constructor handlers (on_list_tools). datasheetindex[mcp] requires "
            "mcp 2.x; pin it, or teach _build_mcp_server the new API."
        )

    def _error_message(name: str, envelope: dict[str, Any]) -> str:
        content = envelope.get("content") or []
        # Error envelopes are text-only today; fall back defensively if a
        # future handler surfaces a non-text first block.
        return (content[0].get("text") if content else None) or f"{name} failed"

    async def _on_list_tools(_ctx: Any, _params: Any) -> Any:
        return types_module.ListToolsResult(
            tools=[
                types_module.Tool(
                    name=d.name,
                    description=d.description,
                    input_schema=d.input_schema,
                )
                for d in session.defs
            ]
        )

    async def _on_call_tool(_ctx: Any, params: Any) -> Any:
        # The raw request: there is nothing between the wire and this callable,
        # so all *three* guards have to be applied here or they are simply lost
        # -- schema validation, the unknown-tool result, and the catch-all. The
        # invariant is that this returns a result and never raises (except for
        # `passthrough`, which the runner interprets): an escaping error becomes
        # a JSON-RPC protocol error the agent cannot read and recover from, and
        # upstream emits it with the invalid error code 0. The try/except
        # enforces that structurally rather than leaving it to be maintained by
        # counting guards.
        arguments = params.arguments or {}
        try:
            definition = by_name.get(params.name)
            if definition is None:
                envelope = _text_error(f"unknown tool: {params.name}")
            elif (
                violation := _schema_violation(validators[params.name], arguments)
            ) is not None:
                envelope = _text_error(f"Input validation error: {violation}")
            else:
                envelope = await definition.handler(arguments)
        except passthrough:
            raise
        except Exception as exc:
            # 1.x's wrapper ended in `return self._make_error_result(str(e))`;
            # `str(exc)` keeps the text an agent sees unchanged since then.
            envelope = _text_error(str(exc))

        content = _envelope_to_content(envelope, types_module)
        is_error = bool(envelope.get("is_error"))
        if is_error and not content:
            # An error result carrying no text at all says less than "it failed".
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

        try:
            if transport == "stdio":
                # stdout IS the wire here; nothing else may write to it.
                _silence_pymupdf_stdout_notice()
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
