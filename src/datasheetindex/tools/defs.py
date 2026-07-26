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
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

from datasheetindex.llm.figure_captions import DEFAULT_MAX_FIGURE_CAPTIONS
from datasheetindex.tools.bound import DatasheetTools


@dataclass(frozen=True)
class DatasheetToolDef:
    """A single datasheet tool, described independently of any agent framework.

    Attributes:
        name: The tool name exposed to the agent.
        description: The natural-language description the agent sees.
        input_schema: JSON Schema (a plain dict) for the tool arguments. The
            dataclass is frozen (the attribute cannot be rebound), but the dict
            itself is mutable and shared by reference -- treat it as read-only;
            deep-copy before mutating if your host needs to adapt it.
        handler: ``async (args: dict) -> {"content": [...], "is_error": bool}``.
            It is an ``async def``, so ``handler(args)`` returns a coroutine you
            can ``await`` or pass straight to ``asyncio.run``.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Coroutine[Any, Any, dict[str, Any]]]


@dataclass(frozen=True)
class DatasheetToolSession:
    """One datasheet tool session: the tool defs plus their lifecycle handle.

    Attributes:
        defs: The framework-neutral :class:`DatasheetToolDef` list for this
            session (see :func:`create_datasheet_tool_defs`). The dataclass is
            frozen (the attribute cannot be rebound), but the list itself is
            mutable -- treat it as read-only; mutating it (append/reorder) can
            desync a host that snapshots it against one that re-reads it live.
        close: Release the session's bound document. Idempotent and safe to call
            when nothing has been built yet. Long-running hosts should call this
            when the session ends so a document loaded from a URL has its
            temporary file cleaned up rather than lingering until process exit.
            Not safe to call concurrently with an in-flight ``build_datasheet``
            (see the concurrency note on :func:`create_datasheet_tool_session`).
    """

    defs: list[DatasheetToolDef]
    close: Callable[[], None]


def create_datasheet_tool_defs() -> list[DatasheetToolDef]:
    """Build the datasheet tools as framework-neutral definitions.

    Convenience wrapper over :func:`create_datasheet_tool_session` that returns
    just the ``defs`` list. Use :func:`create_datasheet_tool_session` instead
    when you need to close the bound document at end of session (URL sources
    leave a temporary file behind until closed).

    Returns the same five tools that
    :func:`datasheetindex.tools.registry.create_datasheet_tools_server` exposes
    -- ``build_datasheet``, ``get_section_text``, ``search_text``,
    ``inspect_page``, ``extract_table_markdown`` -- without importing
    ``claude-agent-sdk``.

    ``DatasheetTools.locate_text`` is deliberately **not** among them. It remains
    a supported Python API for coordinate grounding; it is simply not a tool an
    agent has reason to call, since the box it returns covers well under 1% of a
    page and renders back as a picture of the query string.
    """
    return create_datasheet_tool_session().defs


def create_datasheet_tool_session() -> DatasheetToolSession:
    """Build a datasheet tool session: the neutral tool defs plus a ``close``.

    Per-session state (the current :class:`DatasheetTools` bound by
    ``build_datasheet`` and read by the other tools via ``_require()``) lives in
    this factory's closure: one call == one session. The tools start unbound;
    call the ``build_datasheet`` handler with a ``pdf_source`` (local path or
    URL) to load a document, and call :attr:`DatasheetToolSession.close` when the
    session ends.

    Because that state is per-call, **call this factory once per session and do
    not share the returned defs across sessions.** A host that registers one set
    of defs globally and routes several independent conversations through them
    would have every conversation share (and clobber) a single bound document.

    The handlers -- and ``close`` -- are not safe under *concurrent* invocation
    within one session: ``build_datasheet`` mutates the shared bound document, so
    overlapping calls on the same set of defs (or a ``close`` racing an in-flight
    ``build_datasheet``) can race. This matches agent tool-call semantics (tool
    calls are issued serially); a host that fans out concurrent calls against a
    single session should serialize ``build_datasheet`` (and ``close``) itself.
    """
    tools_instance: DatasheetTools | None = None

    def _ok(result: object) -> dict[str, Any]:
        return {
            "content": [{"type": "text", "text": json.dumps(result, default=str)}],
            "is_error": False,
        }

    def _err(msg: str) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": msg}], "is_error": True}

    def _err_exc(exc: BaseException) -> dict[str, Any]:
        """Report a caught exception as ``TypeName: message``.

        ``str(exc)`` alone is the worst case for diagnosability on the very
        exceptions a blanket ``except`` is most likely to catch: ``KeyError``,
        ``IndexError`` and friends stringify to their bare argument, so a tool
        result whose entire text is ``'end_page'`` reads like truncated output
        rather than a failure. Naming the type makes the failure mode legible at
        a glance. Hand-written validation messages keep using ``_err``, so
        "pdf_source is required" does not acquire a useless prefix.
        """
        return _err(f"{type(exc).__name__}: {exc}")

    def _require() -> DatasheetTools:
        if tools_instance is None:
            raise RuntimeError(
                "No datasheet loaded. Call build_datasheet with a pdf_source first."
            )
        return tools_instance

    def _safe_close(instance: DatasheetTools) -> None:
        # Closing a document is best-effort cleanup: its failure must never undo a
        # successful switch (orphaning the freshly built document) nor mask the
        # error that triggered the cleanup.
        try:
            instance.close()
        except Exception:
            pass

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
                    caption_figures=args.get("caption_figures", True),
                    max_figure_captions=args.get(
                        "max_figure_captions", DEFAULT_MAX_FIGURE_CAPTIONS
                    ),
                )
            except Exception:
                # Discard the fresh instance; best-effort close so cleanup failure
                # cannot replace the real build error being raised.
                if not same_source:
                    _safe_close(target)
                raise

            # Commit the switch: rebind BEFORE closing the predecessor so a failure
            # to close it cannot orphan the freshly built document.
            previous = None if same_source else tools_instance
            tools_instance = target
            if previous is not None:
                _safe_close(previous)
            return _ok(tools_instance.get_artifact_manifest())
        except Exception as exc:
            return _err_exc(exc)

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
            return _err_exc(exc)

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
            return _err_exc(exc)

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
                        # Both spellings, same value, deliberately. This envelope
                        # is the Claude Agent SDK's envelope, and that format is
                        # mixed-case by construction: the SDK reads "is_error"
                        # (snake) but item["mimeType"] (camel). Emitting only
                        # "mime_type" made every inspect_page call through
                        # create_datasheet_tools_server raise KeyError('mimeType')
                        # inside the SDK's converter (#13). Emitting only
                        # "mimeType" would break the other direction:
                        # mcp_server._envelope_to_content and any host already
                        # reading the documented snake_case key. Do not "tidy"
                        # this down to one key -- there is no single spelling
                        # that satisfies both, which is why both are here.
                        "mime_type": blocks[0]["mime_type"],
                        "mimeType": blocks[0]["mime_type"],
                    }
                ],
                "is_error": False,
            }
        except Exception as exc:
            return _err_exc(exc)

    async def extract_table_markdown(args: dict[str, Any]) -> dict[str, Any]:
        try:
            md = await asyncio.to_thread(
                _require().extract_table_markdown, args["page"]
            )
            return _ok({"page": args["page"], "markdown": md})
        except Exception as exc:
            # Includes the ImportError when the optional pymupdf4llm is missing.
            return _err_exc(exc)

    def _close() -> None:
        # End-of-session cleanup: release the bound document (idempotent).
        nonlocal tools_instance
        if tools_instance is not None:
            _safe_close(tools_instance)
            tools_instance = None

    defs = [
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
                "The manifest also carries a 'figures' digest of the document's "
                "figure content: total/raster/captioned counts, and a 'pages' "
                "list of {page, figures, caption} rows, one per page carrying "
                "figure entries. Compare 'raster' against 'total' before "
                "trusting a row: entries are also created for 'Figure N' "
                "mentions found in the page text, so a row can name a page "
                "with no image on it -- a List of Figures page, typically. Use "
                "it to decide where inspect_page is worth calling -- a page "
                "with an image may carry content the extracted text does not "
                "have. The digest is bounded (at most "
                "40 page rows, one caption each, 'truncated' says when more "
                "exist); the complete figures array, with normalized regions "
                "for inspect_page(region=...), is in the ToC JSON at "
                "json_path.\n\n"
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
                "model names.\n\n"
                "IMPORTANT - figure captioning cost: Figure captioning runs by "
                "default (caption_figures=True) whenever vision-capable LLM "
                "credentials are configured -- it is a no-op otherwise. Each "
                "captioned figure is one VLM call, so raising "
                "max_figure_captions raises cost proportionally; leave it at "
                "its default unless a document is known to need more."
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
                    "caption_figures": {
                        "type": "boolean",
                        "description": (
                            "Name raster figure regions with a vision model. "
                            "Default true; no-op without credentials."
                        ),
                    },
                    "max_figure_captions": {
                        "type": "integer",
                        "minimum": 0,
                        "description": (
                            "Per-document ceiling on caption calls "
                            f"(default {DEFAULT_MAX_FIGURE_CAPTIONS})."
                        ),
                    },
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
                "ToC nodes, or from search_text hits when the ToC is empty, to "
                "read specific sections. For a single page use the "
                "same value for both. Prefer reading whole sections rather than "
                "page-by-page. The text opens with a position header -- "
                "'=== Page X of N ===' for a single page, '=== Pages X-Y of N ===' "
                "for a range -- so you know where you are in the document. If a "
                "'=== NOTE: ... ===' line follows the header, the range you "
                "asked for cuts content the publisher marked as continued on "
                "an adjacent page: re-read with that page included before "
                "trusting values from it, since a section's ToC page range "
                "does not always contain all of its content. The '===' "
                "wrapping marks this as the tool's own signal, distinct from "
                "a literal 'NOTE:' line that some datasheets carry in their "
                "own body text. The absence of a note only means none was "
                "detected; it is not a guarantee of completeness."
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

    return DatasheetToolSession(defs=defs, close=_close)
