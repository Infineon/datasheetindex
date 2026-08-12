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

#: Attached to a ``search_text`` result only when the search found nothing AND
#: the document carries raster regions -- the one case where a miss is genuinely
#: uninformative. It names the next action, because a limitation stated without
#: a remedy just stops the agent: the observed failure was an agent reporting
#: "the document does not contain SUMITOMO" after one empty search, when the
#: word was in a supplier table exported as an image.
_EMPTY_SEARCH_RASTER_NOTE = (
    "No text-layer match. This document contains raster figures whose contents "
    "are pixels, not text, so a term inside one is unreachable from here. Check "
    "the 'figures' digest returned by build_datasheet for a page whose caption "
    "describes what you are after, then read that page with inspect_page before "
    "concluding the term is absent."
)


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
                    regenerate_toc=args.get("regenerate_toc", False),
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
            tools = _require()
            results = tools.search_text(
                args["query"],
                page=args.get("page"),
                case_sensitive=args.get("case_sensitive", False),
                max_results=args.get("max_results", 20),
            )
            payload: dict[str, Any] = {"query": args["query"], "results": results}
            # The caveat is on this tool's description too, but a description is
            # read once and a zero-hit search is where it matters. Only when the
            # document actually holds pixels a search cannot reach, and only on
            # a miss: a note on every call is noise the agent reads past.
            if not results and tools.has_raster_figures():
                payload["note"] = _EMPTY_SEARCH_RASTER_NOTE
            return _ok(payload)
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
                "Load a datasheet -- building its enriched ToC JSON and "
                "page-matched text file -- and return the manifest. Call this "
                "before the other tools; calling it again with a different "
                "pdf_source switches documents.\n\n"
                "The manifest carries the source, total pages, ToC quality, and "
                "the enriched Table of Contents -- section hierarchy, page "
                "ranges, table counts, footnote markers, cross-references -- "
                "plus a 'figures' digest: total/raster/captioned counts and up "
                "to 40 {page, figures, caption} rows ('truncated' says when "
                "more exist), each row carrying that page's largest figure. A "
                "caption names the kind of content "
                "(table, schematic, plot, photo, block diagram, pinout) and "
                "then its most identifying labels: a table's row labels then "
                "column headings, a plot's axes and plotted quantity. Use it to "
                "choose pages worth inspect_page -- above all when search_text "
                "finds nothing, since a captioned region is pixels no text "
                "search can reach.\n\n"
                "A row with no raster region comes from a 'Figure N' mention in "
                "the page text, and usually means a vector-drawn figure the "
                "index cannot give coordinates for: still worth a full-page "
                "inspect_page. The complete figures array, with regions for "
                "inspect_page(region=...), is in the ToC JSON at json_path.\n\n"
                "'toc_source' is where the ToC came from: 'pdf_outline' (the "
                "PDF's own bookmarks -- pages exact), 'llm_reconstructed' "
                "(rewritten from body text -- every start_page is inferred, "
                "so confirm a section with search_text before reading its "
                "range), or 'none'.\n\n"
                "The returned 'toc' is the outline the PDF itself carries, and "
                "it is occasionally useless -- entries like 'Page 1', 'Page 2' "
                "that name no section. Read it before planning: if the entries "
                "do not identify sections, call build_datasheet again with "
                "regenerate_toc=true to rebuild the outline from the body text. "
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "pdf_source": {
                        "type": "string",
                        "description": "Local path or http(s) URL of the PDF.",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": (
                            "Where to write the artifacts. Omit for a writable default."
                        ),
                    },
                    "output_stem": {
                        "type": "string",
                        "description": (
                            "Base filename for the artifacts. Omit to derive it "
                            "from the source."
                        ),
                    },
                    "include_summaries": {
                        "type": "boolean",
                        "description": (
                            "Add an LLM summary to every ToC node. Off by "
                            "default, and best left off unless the user asks: it "
                            "costs one LLM call per section, and the ToC and page "
                            "text are usually enough. Requires 'model'."
                        ),
                    },
                    "model": {
                        "type": "string",
                        "description": (
                            "Model for summaries and the weak-ToC fallback. Omit "
                            "it unless the user named a model: omitting it uses "
                            "the model this deployment is configured with, and a "
                            "name its gateway does not serve fails the call. Set "
                            "DATASHEETINDEX_MODEL to change that default."
                        ),
                    },
                    "force_rebuild": {
                        "type": "boolean",
                        "description": (
                            "Rebuild even when artifacts on disk already match "
                            "this source."
                        ),
                    },
                    "regenerate_toc": {
                        "type": "boolean",
                        "description": (
                            "Rewrite the table of contents from the body text "
                            "with an LLM, even when the PDF's own outline "
                            "scored well enough to be kept. Requires "
                            "credentials; the call fails if none are "
                            "configured."
                        ),
                    },
                    "caption_figures": {
                        "type": "boolean",
                        "description": (
                            "Name raster figure regions with a vision model. "
                            "Default true; a no-op without credentials."
                        ),
                    },
                    "max_figure_captions": {
                        "type": "integer",
                        "minimum": 0,
                        "description": (
                            "Ceiling on caption calls per document (default "
                            f"{DEFAULT_MAX_FIGURE_CAPTIONS}). Each distinct "
                            "picture costs one vision call, so a higher ceiling "
                            "costs proportionally more."
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
                "Read the extracted text for a page range. Take start_page and "
                "end_page from ToC nodes, or from search_text hits when the ToC "
                "is empty, and prefer whole sections over page-by-page reads.\n\n"
                "The text opens with '=== Page X of N ===' or "
                "'=== Pages X-Y of N ==='. A '=== NOTE: ... ===' line after it "
                "means the range cuts content the publisher marked as continued "
                "on an adjacent page -- re-read including that page before "
                "trusting values from it, since a section's ToC range does not "
                "always hold all of its content. The '===' wrapper is what marks "
                "the line as the tool's own signal rather than a literal 'NOTE:' "
                "in the datasheet's body text. Absence of a note means none was "
                "detected; it is not a guarantee of completeness."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "start_page": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "1-indexed, inclusive.",
                    },
                    "end_page": {
                        "type": "integer",
                        "minimum": 1,
                        "description": (
                            "1-indexed, inclusive. Same as start_page for one page."
                        ),
                    },
                },
                "required": ["start_page", "end_page"],
            },
            handler=get_section_text,
        ),
        DatasheetToolDef(
            name="search_text",
            description=(
                "Locate a term across the datasheet: returns page-aware snippets "
                "with surrounding context, each tagged with the ToC 'breadcrumb' "
                "of the section holding the match (and, for a list query, the "
                "'pattern' that matched).\n\n"
                "Searches the extracted text layer only. A table, schematic or "
                "label placed as an image has no text layer, so a term inside it "
                "cannot be found here: the absence of a match does not prove the "
                "document lacks the term. On an empty result, check the 'figures' "
                "digest from build_datasheet for a page whose caption describes "
                "what you want, then read it with inspect_page."
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
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Default false.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Default 20.",
                    },
                },
                "required": ["query"],
            },
            handler=search_text,
        ),
        DatasheetToolDef(
            name="inspect_page",
            description=(
                "Render a page as a PNG image and look at it. Use when the "
                "extracted text is garbled or missing -- complex table layouts, "
                "block diagrams, pin-outs, timing diagrams, and any page whose "
                "content is a picture rather than text."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "page": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "1-indexed.",
                    },
                    "region": {
                        "type": "object",
                        "description": (
                            "Crop to top/bottom/left/right, each 0.0-1.0. A "
                            "figures entry's 'region' can be passed straight "
                            "through."
                        ),
                    },
                    "detail": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": (
                            "Vision-token cost tier: 'low' (75 dpi) for layout, "
                            "'medium' (100 dpi, the usual choice) for body text "
                            "and table cells, 'high' (150 dpi) for footnotes, "
                            "subscripts and dense schematics."
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
                "Re-extract one page as layout-aware Markdown with proper table "
                "formatting. Use when get_section_text shows a garbled or "
                "misaligned table and you want clean pipe-delimited rows. Costs "
                "fewer tokens than inspect_page (text, not vision) but takes "
                "about 3s per page. The page's running header and footer are "
                "omitted, as they are in get_section_text and search_text."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "page": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "1-indexed, as in the PAGE markers.",
                    },
                },
                "required": ["page"],
            },
            handler=extract_table_markdown,
        ),
    ]

    return DatasheetToolSession(defs=defs, close=_close)
