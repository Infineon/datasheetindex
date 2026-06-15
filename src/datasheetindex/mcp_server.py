"""Local MCP server entry point for datasheetindex."""

import argparse
import asyncio
import importlib
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, TypedDict, cast

from datasheetindex.tools.registry import DatasheetTools
from datasheetindex.tools.vision import Detail


class Region(TypedDict, total=False):
    top: float
    bottom: float
    left: float
    right: float


@dataclass
class _ServerContext:
    tools: DatasheetTools | None = None


def _load_mcp_modules() -> tuple[Any, Any, Any]:
    try:
        return (
            importlib.import_module("mcp.server.fastmcp"),
            importlib.import_module("mcp.server.session"),
            importlib.import_module("mcp.types"),
        )
    except ImportError:
        raise ImportError(
            "mcp is required for local MCP server support. "
            "Install it with: uv sync --extra mcp"
        ) from None


def _preload_layout_model() -> None:
    """Import pymupdf4llm to trigger ONNX model loading at startup.

    The layout model takes ~2s to load. Doing this eagerly at server
    start avoids a long GIL-holding pause on the first tool call, which
    can cause MCP client timeouts.
    """
    try:
        import pymupdf4llm  # noqa: F401
    except ImportError:
        pass  # optional dependency; extract_table_markdown will report the error


def create_local_mcp_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    streamable_http_path: str = "/mcp",
) -> Any:
    """Create a standard MCP server for local stdio/HTTP testing.

    The server starts without a bound PDF. Call ``build_datasheet`` with a
    ``pdf_source`` (local path or URL) to load a datasheet. Calling it again
    with a different source replaces the current document.
    """
    fastmcp_module, session_module, types_module = _load_mcp_modules()
    FastMCP = fastmcp_module.FastMCP
    Context = fastmcp_module.Context
    ServerSession = session_module.ServerSession
    CallToolResult = types_module.CallToolResult
    ImageContent = types_module.ImageContent

    @asynccontextmanager
    async def _lifespan(_server: Any) -> AsyncIterator[_ServerContext]:
        ctx = _ServerContext()
        # Pre-load pymupdf4llm ONNX models so the first extract_table_markdown
        # call doesn't block for ~2s during model initialization.
        await asyncio.to_thread(_preload_layout_model)
        try:
            yield ctx
        finally:
            if ctx.tools is not None:
                ctx.tools.close()

    server = FastMCP(
        name="datasheetindex",
        instructions=(
            "Extract technical parameters from PDF datasheets. Call "
            "build_datasheet FIRST with a pdf_source (local path or URL) "
            "to load a document -- it returns the full enriched ToC for "
            "navigation planning. Then use get_section_text to read page "
            "ranges, search_text to locate keywords, and inspect_page for "
            "visual content. You can switch documents by calling "
            "build_datasheet with a new source. When a table in "
            "get_section_text looks garbled, use extract_table_markdown "
            "for a clean Markdown table (cheaper than inspect_page). "
            "Use locate_text to get the bounding-box coordinates of a string "
            "on a page (for highlighting or to crop inspect_page precisely)."
        ),
        host=host,
        port=port,
        streamable_http_path=streamable_http_path,
        lifespan=_lifespan,
    )

    def inspect_page_tool(
        page: int,
        region: Region | None = None,
        dpi: int | None = None,
        detail: Detail = "medium",
        ctx: Context[ServerSession, _ServerContext] | None = None,
    ) -> Any:
        """Render a PDF page as an MCP image result.

        `detail` defaults to "medium" (100 dpi, ~1150 vision tokens) -- the
        right tier for most agent calls. Bump to "high" (150 dpi, ~2580
        tokens) for footnotes/subscripts/dense schematics, or drop to
        "low" (75 dpi, ~650 tokens) for layout overview. `dpi` is a
        power-user override that wins over `detail` when set.
        """
        blocks = _require_tools(ctx).inspect_page(
            page,
            region=cast("dict[str, float] | None", region),
            dpi=dpi,
            detail=detail,
        )
        if len(blocks) != 1:
            raise RuntimeError("inspect_page returned an unexpected content shape")

        block = blocks[0]
        return CallToolResult(
            content=[
                ImageContent(
                    type="image",
                    data=block["data"],
                    mimeType=block["mime_type"],
                )
            ]
        )

    async def build_datasheet_tool(
        pdf_source: str,
        output_dir: str | None = None,
        output_stem: str | None = None,
        include_summaries: bool = False,
        model: str | None = None,
        force_rebuild: bool = False,
        ctx: Context[ServerSession, _ServerContext] | None = None,
    ) -> dict[str, object]:
        """Build and save datasheet artifacts for a PDF source.

        ``output_dir`` is optional; the resolution rules live with
        :meth:`DatasheetIndex.build` (single source of truth).
        """
        if ctx is None:
            raise RuntimeError("MCP context was not provided")
        server_ctx = ctx.request_context.lifespan_context
        # Re-bind to a new PDF if the source changed
        if server_ctx.tools is None or server_ctx.tools.pdf_path != pdf_source:
            if server_ctx.tools is not None:
                server_ctx.tools.close()
            server_ctx.tools = DatasheetTools(pdf_source)
        await asyncio.to_thread(
            server_ctx.tools.build_datasheet,
            output_dir=output_dir,
            output_stem=output_stem,
            include_summaries=include_summaries,
            model=model,
            force_rebuild=force_rebuild,
        )
        return server_ctx.tools.get_artifact_manifest()

    def get_section_text_tool(
        start_page: int,
        end_page: int,
        ctx: Context[ServerSession, _ServerContext] | None = None,
    ) -> dict[str, object]:
        """Return extracted text for a page range (inclusive, 1-indexed)."""
        tools = _require_tools(ctx)
        return {
            "start_page": start_page,
            "end_page": end_page,
            "text": tools.get_section_text(start_page, end_page),
        }

    def search_text_tool(
        query: str | list[str],
        page: int | None = None,
        case_sensitive: bool = False,
        max_results: int = 20,
        ctx: Context[ServerSession, _ServerContext] | None = None,
    ) -> dict[str, object]:
        """Search the latest built text artifact and return page-aware matches."""
        tools = _require_tools(ctx)
        return {
            "query": query,
            "page": page,
            "case_sensitive": case_sensitive,
            "results": tools.search_text(
                query,
                page=page,
                case_sensitive=case_sensitive,
                max_results=max_results,
            ),
        }

    def locate_text_tool(
        query: str | list[str],
        page: int | None = None,
        max_results: int = 20,
        ctx: Context[ServerSession, _ServerContext] | None = None,
    ) -> dict[str, object]:
        """Map a string to bounding-box coordinates on a page."""
        tools = _require_tools(ctx)
        return {
            "query": query,
            "results": tools.locate_text(query, page=page, max_results=max_results),
        }

    server.tool(
        name="build_datasheet",
        description=(
            "Build the enriched ToC JSON and page-matched text file for a "
            "datasheet. CALL THIS FIRST with a pdf_source (local path or URL) "
            "before using any other tool. Calling again with a different "
            "source switches documents (cached if same source). Returns an "
            "artifact manifest with source info, total pages, ToC quality "
            "score, and the full enriched Table of Contents with section "
            "hierarchy, page ranges, table counts, footnote markers, and "
            "cross-references.\n\n"
            "IMPORTANT - include_summaries: Leave as False (default) unless "
            "the user explicitly requests summaries. Generating summaries "
            "makes one LLM call per ToC section, which is slow and "
            "expensive. The ToC, text file, and other tools already provide "
            "enough context for most tasks.\n\n"
            "IMPORTANT - model: Do NOT set this unless summaries are "
            "requested or ToC quality is very poor. When needed, use one of "
            "the models available on the LiteLLM gateway: gpt-4.1 "
            "(recommended default), gpt-5-mini, gpt-5-nano, gpt-4.1-nano, "
            "gpt-4o-mini, gpt-5, gpt-5.1, gpt-5.2. Do NOT invent or guess "
            "model names."
        ),
    )(build_datasheet_tool)
    server.tool(
        name="get_section_text",
        description=(
            "Read the extracted text for a page range (inclusive, 1-indexed). "
            "Use when you know WHERE to read -- pass start_page/end_page from "
            "ToC nodes to read specific sections. For a single page use the same "
            "value for both. Prefer reading whole sections rather than "
            "page-by-page. The text opens with a '=== Pages X-Y of N ===' header "
            "so you know your position in the document."
        ),
    )(get_section_text_tool)
    server.tool(
        name="search_text",
        description=(
            "Search the full extracted text and return page-aware snippets with "
            "surrounding context. Use when you know WHAT to look for -- a "
            "parameter name, value, or keyword -- to locate it across the "
            "datasheet before reading specific sections. 'query' may be a single "
            "string or a list of strings to search several terms in one call. "
            "Each result includes the ToC 'breadcrumb' of the section containing "
            "the match; list searches also tag each result with the matching "
            "'pattern'. Omit 'page' to search all pages."
        ),
    )(search_text_tool)
    server.tool(
        name="inspect_page",
        description=(
            "Render a PDF page as a PNG image for visual inspection. Use "
            "when extracted text is garbled or insufficient -- tables with "
            "complex layouts, block diagrams, pin-out figures, timing "
            "diagrams. Optionally crop with top/bottom/left/right percentages "
            "(0.0-1.0)."
        ),
    )(inspect_page_tool)
    server.tool(
        name="locate_text",
        description=(
            "Map a piece of text to its bounding-box coordinates on a page, "
            "for highlighting or precise visual inspection. Returns one result "
            "per occurrence, each with 'region' (the union rectangle) and "
            "'boxes' (one per line), in both percentages and PDF points. Feed "
            "region['pct'] into inspect_page(region=...) to crop to the exact "
            "spot; use region['points'] to annotate the PDF. Pass 'page' when "
            "you know it (e.g. from a search_text hit) to stay cheap; omit to "
            "scan all pages."
        ),
    )(locate_text_tool)

    async def extract_table_markdown_tool(
        page: int,
        ctx: Context[ServerSession, _ServerContext] | None = None,
    ) -> dict[str, object]:
        """Re-extract a single page as layout-aware Markdown."""
        tools = _require_tools(ctx)
        markdown = await asyncio.to_thread(tools.extract_table_markdown, page)
        return {
            "page": page,
            "markdown": markdown,
        }

    server.tool(
        name="extract_table_markdown",
        description=(
            "Re-extract a single page as layout-aware Markdown with proper "
            "table formatting (| delimited rows). Use when get_section_text "
            "shows a garbled or misaligned table and you need clean structured "
            "data for parameter extraction. Cheaper than inspect_page (text "
            "tokens vs vision tokens) but slower (~3s per page). Pass the "
            "1-indexed page number from the PAGE marker."
        ),
    )(extract_table_markdown_tool)
    return server


def _require_tools(ctx: Any) -> DatasheetTools:
    if ctx is None:
        raise RuntimeError("MCP context was not provided")
    tools = ctx.request_context.lifespan_context.tools
    if tools is None:
        raise RuntimeError(
            "No datasheet loaded. Call build_datasheet with a pdf_source first."
        )
    return tools


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="datasheetindex-mcp-server",
        description=(
            "Run datasheetindex as a local MCP server. "
            "Use build_datasheet to load a PDF source."
        ),
    )
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
    """Entry point for console_scripts."""
    raise SystemExit(main())


if __name__ == "__main__":
    main_cli()
