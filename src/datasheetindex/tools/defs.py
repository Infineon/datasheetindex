"""Framework-neutral datasheet tool definitions.

This module realizes the datasheet agent tools as plain, self-describing
definitions -- **without importing ``claude-agent-sdk``**. Hosts that are not on
the Claude Agent SDK (pydantic-ai, plain function-calling agents, custom MCP
servers) can wrap each :class:`DatasheetToolDef` directly: the
``{"content": [...], "is_error": bool}`` envelope its ``handler`` returns already
matches what most hosts expect.

:func:`create_datasheet_tool_defs` owns per-session state (the current
``DatasheetTools`` bound by the ``build_datasheet`` handler and read by the
others). One call == one session, so two calls own independent documents.

The SDK adapter :func:`datasheetindex.tools.registry.create_datasheet_tools_server`
is a thin wrapper built on top of these defs, so tool names, descriptions, and
JSON schemas stay identical across both surfaces.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from datasheetindex.tools.registry import DatasheetTools


@dataclass(frozen=True)
class DatasheetToolDef:
    """A single datasheet tool, described independently of any agent framework.

    Attributes:
        name: The tool name exposed to the agent.
        description: The natural-language description the agent sees.
        input_schema: JSON Schema (a plain dict) for the tool arguments.
        handler: ``async (args: dict) -> {"content": [...], "is_error": bool}``.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


def create_datasheet_tool_defs() -> list[DatasheetToolDef]:
    """Build the datasheet tools as framework-neutral definitions.

    Returns the same six tools that
    :func:`datasheetindex.tools.registry.create_datasheet_tools_server` exposes
    -- ``build_datasheet``, ``get_section_text``, ``search_text``,
    ``inspect_page``, ``locate_text``, ``extract_table_markdown`` -- without
    importing ``claude-agent-sdk``.

    Per-session state (the current :class:`DatasheetTools` bound by
    ``build_datasheet`` and read by the other tools via ``_require()``) lives in
    this factory's closure: one call == one session. The tools start unbound;
    call the ``build_datasheet`` handler with a ``pdf_source`` (local path or
    URL) to load a document.
    """
    tools_instance: DatasheetTools | None = None

    def _ok(result: object) -> dict[str, Any]:
        return {
            "content": [{"type": "text", "text": json.dumps(result, default=str)}],
            "is_error": False,
        }

    def _err(msg: str) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": msg}], "is_error": True}

    def _require() -> DatasheetTools:
        if tools_instance is None:
            raise RuntimeError(
                "No datasheet loaded. Call build_datasheet with a pdf_source first."
            )
        return tools_instance

    async def build_datasheet(args: dict[str, Any]) -> dict[str, Any]:
        nonlocal tools_instance
        try:
            pdf_source = args.get("pdf_source", "")
            if not pdf_source:
                return _err("pdf_source is required")

            same_source = (
                tools_instance is not None and tools_instance.pdf_path == pdf_source
            )
            # For a switch, build into a FRESH instance and only commit (close the
            # old document, rebind) once the build succeeds -- a failed switch to a
            # bad/unavailable source must leave the working document intact rather
            # than close it and strand every later query.
            target = tools_instance if same_source else DatasheetTools(pdf_source)
            try:
                await asyncio.to_thread(
                    target.build_datasheet,
                    output_dir=args.get("output_dir"),
                    output_stem=args.get("output_stem"),
                    include_summaries=args.get("include_summaries", False),
                    model=args.get("model"),
                    force_rebuild=args.get("force_rebuild", False),
                )
            except Exception:
                if not same_source:
                    target.close()
                raise

            if not same_source and tools_instance is not None:
                tools_instance.close()
            tools_instance = target
            return _ok(tools_instance.get_artifact_manifest())
        except Exception as exc:
            return _err(str(exc))

    async def get_section_text(args: dict[str, Any]) -> dict[str, Any]:
        try:
            ti = _require()
            text = ti.get_section_text(args["start_page"], args["end_page"])
            result = {
                "start_page": args["start_page"],
                "end_page": args["end_page"],
                "text": text,
            }
            return _ok(result)
        except Exception as exc:
            return _err(str(exc))

    async def search_text(args: dict[str, Any]) -> dict[str, Any]:
        try:
            results = _require().search_text(
                args["query"],
                page=args.get("page"),
                case_sensitive=args.get("case_sensitive", False),
                max_results=args.get("max_results", 20),
            )
            return _ok({"query": args["query"], "results": results})
        except Exception as exc:
            return _err(str(exc))

    async def inspect_page(args: dict[str, Any]) -> dict[str, Any]:
        try:
            blocks = _require().inspect_page(
                args["page"],
                region=args.get("region"),
                dpi=args.get("dpi"),
                detail=args.get("detail", "medium"),
            )
            return {
                "content": [
                    {
                        "type": "image",
                        "data": blocks[0]["data"],
                        "mime_type": blocks[0]["mime_type"],
                    }
                ],
                "is_error": False,
            }
        except Exception as exc:
            return _err(str(exc))

    async def locate_text(args: dict[str, Any]) -> dict[str, Any]:
        try:
            results = _require().locate_text(
                args["query"],
                page=args.get("page"),
                max_results=args.get("max_results", 20),
            )
            return _ok({"query": args["query"], "results": results})
        except Exception as exc:
            return _err(str(exc))

    async def extract_table_markdown(args: dict[str, Any]) -> dict[str, Any]:
        try:
            md = await asyncio.to_thread(
                _require().extract_table_markdown, args["page"]
            )
            return _ok({"page": args["page"], "markdown": md})
        except ImportError as exc:
            return _err(str(exc))
        except Exception as exc:
            return _err(str(exc))

    return [
        DatasheetToolDef(
            name="build_datasheet",
            description=(
                "Build the enriched ToC JSON and page-matched text file for a "
                "datasheet. CALL THIS FIRST with a pdf_source (local path or URL) "
                "before using any other tool. Calling again with a different source "
                "switches documents. Returns an artifact manifest with source info, "
                "total pages, ToC quality score, and the full enriched Table of "
                "Contents with section hierarchy, page ranges, table counts, "
                "footnote markers, and cross-references.\n\n"
                "output_dir is optional -- omit it unless you need artifacts at a "
                "specific path; the library picks a writable default.\n\n"
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
            input_schema={
                "type": "object",
                "properties": {
                    "pdf_source": {"type": "string"},
                    "output_dir": {"type": "string"},
                    "output_stem": {"type": "string"},
                    "include_summaries": {"type": "boolean"},
                    "model": {"type": "string"},
                    "force_rebuild": {"type": "boolean"},
                },
                "required": ["pdf_source"],
            },
            handler=build_datasheet,
        ),
        DatasheetToolDef(
            name="get_section_text",
            description=(
                "Read the extracted text for a page range (inclusive, 1-indexed). "
                "Use when you know WHERE to read -- pass start_page/end_page from "
                "ToC nodes to read specific sections. For a single page use the "
                "same value for both. Prefer reading whole sections rather than "
                "page-by-page. The text opens with a '=== Pages X-Y of N ===' "
                "header so you know your position in the document."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "start_page": {"type": "integer", "minimum": 1},
                    "end_page": {"type": "integer", "minimum": 1},
                },
                "required": ["start_page", "end_page"],
            },
            handler=get_section_text,
        ),
        DatasheetToolDef(
            name="search_text",
            description=(
                "Search the full extracted text and return page-aware snippets "
                "with surrounding context. Use when you know WHAT to look for -- a "
                "parameter name, value, or keyword -- to locate it across the "
                "datasheet before reading specific sections. 'query' may be a "
                "single string or a list of strings to search several terms in one "
                "call. Each result includes the ToC 'breadcrumb' of the section "
                "containing the match; list searches also tag each result with the "
                "matching 'pattern'. Omit 'page' to search all."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}},
                        ],
                        "description": "A single pattern or a list of patterns.",
                    },
                    "page": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "1-indexed page to search. Omit to search all.",
                    },
                    "case_sensitive": {"type": "boolean"},
                    "max_results": {"type": "integer"},
                },
                "required": ["query"],
            },
            handler=search_text,
        ),
        DatasheetToolDef(
            name="inspect_page",
            description=(
                "Render a PDF page as a PNG image for visual inspection. Use when "
                "extracted text is garbled or insufficient -- tables with complex "
                "layouts, block diagrams, pin-out figures, timing diagrams. "
                "Optionally crop with top/bottom/left/right percentages (0.0-1.0). "
                "Pick `detail` to control vision-token cost: 'low' for layout "
                "overview, 'medium' (recommended default) for body text and table "
                "cells, 'high' for footnotes / subscripts / dense schematics."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "page": {"type": "integer", "minimum": 1},
                    "region": {
                        "type": "object",
                        "description": (
                            "Crop region with top/bottom/left/right (0.0-1.0)"
                        ),
                    },
                    "detail": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": (
                            "Vision-token-cost tier. low=75 dpi, "
                            "medium=100 dpi (recommended), high=150 dpi."
                        ),
                    },
                    "dpi": {
                        "type": "integer",
                        "description": "Explicit override; wins over `detail`.",
                    },
                },
                "required": ["page"],
            },
            handler=inspect_page,
        ),
        DatasheetToolDef(
            name="locate_text",
            description=(
                "Map a piece of text to its bounding-box coordinates on a page, "
                "for highlighting or precise visual inspection. Returns a result "
                "per match, each with `region` (a bounding rectangle) and `boxes` "
                "(one or more per-line rectangles; `region` is their union), in "
                "both normalized percentages and PDF points. A string that appears "
                "more than once yields multiple results. Feed region['pct'] into "
                "inspect_page(region=...) to crop to the exact spot; use "
                "region['points'] (PDF points) to annotate the PDF. Pass `page` "
                "when you know it (e.g. from a search_text hit) to stay cheap; omit "
                "it to scan all pages. `query` may be a single string or a list of "
                "strings."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}},
                        ],
                        "description": "A single pattern or a list of patterns.",
                    },
                    "page": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "1-indexed page to locate on. Omit to scan all.",
                    },
                    "max_results": {"type": "integer", "minimum": 1},
                },
                "required": ["query"],
            },
            handler=locate_text,
        ),
        DatasheetToolDef(
            name="extract_table_markdown",
            description=(
                "Re-extract a single page as layout-aware Markdown with proper "
                "table formatting. Use when get_section_text shows a garbled or "
                "misaligned table and you need clean | delimited rows for parameter "
                "extraction. Cheaper than inspect_page (text tokens vs vision "
                "tokens) but slower (~3s per page). Pass the 1-indexed page number "
                "from the PAGE marker."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "page": {"type": "integer", "minimum": 1},
                },
                "required": ["page"],
            },
            handler=extract_table_markdown,
        ),
    ]
