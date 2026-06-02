"""Tool registration for Agent SDK / MCP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from datasheetindex.core.structure import find_breadcrumb_for_page
from datasheetindex.core.textfile import TextSearchMatch, extract_section_text
from datasheetindex.core.textfile import search_text as search_text_content
from datasheetindex.index import DatasheetIndex
from datasheetindex.llm.client import close_llm_client
from datasheetindex.models import DatasheetArtifacts
from datasheetindex.tools.vision import Detail, inspect_page

if TYPE_CHECKING:
    import pymupdf


@dataclass(frozen=True)
class _BuildOptions:
    output_dir: str
    output_stem: str | None
    include_summaries: bool
    model: str | None


class DatasheetTools:
    """Wraps datasheetindex tools with a bound PDF document."""

    def __init__(self, pdf_path: str) -> None:
        self.pdf_path = pdf_path
        self._index = DatasheetIndex(pdf_path)
        self._artifacts: DatasheetArtifacts | None = None
        self._build_options: _BuildOptions | None = None

    def __enter__(self) -> DatasheetTools:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        self.close()

    @property
    def doc(self) -> pymupdf.Document:
        """Lazy-open the PDF document."""
        return self._index.doc

    @property
    def _doc(self) -> pymupdf.Document | None:
        """Expose current bound document state for compatibility/tests."""
        return self._index._doc

    def close(self) -> None:
        """Close the underlying PDF document."""
        self._index.close()
        self._artifacts = None
        self._build_options = None

    def inspect_page(
        self,
        page: int,
        region: dict[str, float] | None = None,
        dpi: int | None = None,
        detail: Detail = "medium",
    ) -> list[dict]:
        """Render a PDF page as an image for visual inspection.

        Delegates to ``datasheetindex.tools.vision.inspect_page`` with the
        bound document. Defaults to ``detail="medium"`` (100 dpi, ~1150
        vision tokens per page on the Anthropic ``(W*H)/750`` formula)
        because this is the agent-surface wrapper: most loop calls don't
        need 150-dpi footnote fidelity. See ``vision.inspect_page`` for
        the full tier semantics.
        """
        return inspect_page(self.doc, page, region=region, dpi=dpi, detail=detail)

    def extract_table_markdown(self, page: int) -> str:
        """Extract a single page as layout-aware markdown with table structure.

        Requires the ``[layout]`` extra (``pymupdf4llm``). Returns markdown
        with proper table formatting using ``|`` delimiters.
        """
        total = len(self.doc)
        if page < 1 or page > total:
            raise ValueError(f"page must be between 1 and {total}")
        try:
            import pymupdf4llm
        except ImportError:
            raise ImportError(
                "pymupdf4llm is required for table markdown extraction. "
                "Install it with: uv sync --extra layout"
            ) from None
        return pymupdf4llm.to_markdown(self.doc, pages=[page - 1], show_progress=False)

    def build_datasheet(
        self,
        output_dir: str | None = None,
        output_stem: str | None = None,
        include_summaries: bool = False,
        model: str | None = None,
        force_rebuild: bool = False,
    ) -> DatasheetArtifacts:
        """Build and cache datasheet artifacts for later MCP queries."""
        if include_summaries and model is None:
            raise ValueError("--include-summaries requires --model")

        # Resolve once so the cache key is the actual destination path -- two
        # successive calls with output_dir=None must miss the cache if the
        # resolver default has changed between them (e.g. env var rebound).
        from datasheetindex.index import resolve_default_output_dir

        resolved_output_dir = (
            output_dir
            if output_dir is not None and output_dir.strip()
            else resolve_default_output_dir()
        )

        options = _BuildOptions(
            output_dir=resolved_output_dir,
            output_stem=output_stem,
            include_summaries=include_summaries,
            model=model,
        )
        if (
            not force_rebuild
            and self._artifacts is not None
            and self._build_options == options
            and self._artifacts.json_path is not None
            and self._artifacts.json_path.exists()
            and self._artifacts.text_path is not None
            and self._artifacts.text_path.exists()
        ):
            return self._artifacts

        llm_callable = None
        try:
            if model is not None:
                from datasheetindex.llm.client import create_llm_client

                llm_callable = create_llm_client(model=model)

            artifacts = self._index.build(
                output_dir=resolved_output_dir,
                output_stem=output_stem,
                include_summaries=include_summaries,
                llm_callable=llm_callable,
            )
        finally:
            close_llm_client(llm_callable)

        self._artifacts = artifacts
        self._build_options = options
        return artifacts

    def get_artifact_manifest(self) -> dict[str, object]:
        """Return a compact summary of the currently built artifacts."""
        artifacts = self._require_artifacts()
        return {
            "source": artifacts.json_data.get("source"),
            "total_pages": self._total_pages(artifacts),
            "json_path": (
                str(artifacts.json_path) if artifacts.json_path is not None else None
            ),
            "text_path": (
                str(artifacts.text_path) if artifacts.text_path is not None else None
            ),
            "toc_quality": artifacts.json_data.get("toc_quality"),
            "toc": artifacts.json_data.get("toc"),
        }

    def get_section_text(self, start_page: int, end_page: int) -> str:
        """Return extracted text for a page range from the latest build.

        The text opens with a ``=== Pages X-Y of N ===`` position header for
        orientation, followed by the section text WITH ``--- PAGE N ---``
        markers so the agent can orient within the range.
        """
        artifacts = self._require_artifacts()
        total_pages = self._total_pages(artifacts)
        if start_page < 1 or end_page > total_pages or start_page > end_page:
            raise ValueError(
                f"start_page/end_page must satisfy "
                f"1 <= start_page <= end_page <= {total_pages}"
            )
        if start_page == end_page:
            header = f"=== Page {start_page} of {total_pages} ==="
        else:
            header = f"=== Pages {start_page}-{end_page} of {total_pages} ==="
        section = extract_section_text(artifacts.text_content, start_page, end_page)
        return f"{header}\n{section}"

    def search_text(
        self,
        query: str | list[str],
        *,
        page: int | None = None,
        case_sensitive: bool = False,
        max_results: int = 20,
    ) -> list[TextSearchMatch]:
        """Search the built page-matched text and return page-aware snippets.

        ``query`` may be a single pattern or a list of patterns searched in one
        call (each match is tagged with the ``pattern`` that produced it). Every
        match is enriched with the ToC ``breadcrumb`` of the section that
        contains its page, when one is found.
        """
        artifacts = self._require_artifacts()
        total_pages = self._total_pages(artifacts)
        if page is not None and (page < 1 or page > total_pages):
            raise ValueError(f"page must be between 1 and {total_pages}")
        matches = search_text_content(
            artifacts.text_content,
            query,
            page=page,
            case_sensitive=case_sensitive,
            max_results=max_results,
        )
        if artifacts.nodes:
            # Matches commonly cluster on a few pages; resolve each page's
            # breadcrumb once rather than re-walking the ToC per match.
            breadcrumb_by_page: dict[int, str | None] = {}
            for match in matches:
                page_number = match["page"]
                if page_number not in breadcrumb_by_page:
                    breadcrumb_by_page[page_number] = find_breadcrumb_for_page(
                        artifacts.nodes, page_number
                    )
                breadcrumb = breadcrumb_by_page[page_number]
                if breadcrumb:
                    match["breadcrumb"] = breadcrumb
        return matches

    def _require_artifacts(self) -> DatasheetArtifacts:
        if self._artifacts is None:
            raise RuntimeError(
                "No datasheet artifacts available. Call build_datasheet first."
            )
        return self._artifacts

    def _total_pages(self, artifacts: DatasheetArtifacts) -> int:
        total_pages = artifacts.json_data.get("total_pages")
        if not isinstance(total_pages, int):
            raise RuntimeError("Built artifacts are missing total_pages")
        return total_pages


def create_datasheet_tools_server():
    """Create the MCP/tool server that a consuming agent can mount.

    Requires ``claude-agent-sdk`` to be installed. Raises ``ImportError``
    if the SDK is not available. The server starts without a bound PDF;
    call ``build_datasheet`` with a ``pdf_source`` to load a document.
    """
    import asyncio
    import json
    from typing import Any

    try:
        from claude_agent_sdk import (  # type: ignore[import-not-found]  # ty: ignore[unresolved-import]
            create_sdk_mcp_server,
            tool,
        )
    except ImportError:
        raise ImportError(
            "claude-agent-sdk is required for tool server creation. "
            "Install it with: uv pip install claude-agent-sdk"
        ) from None

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

    @tool(
        "build_datasheet",
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
        "model names.",
        {
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
    )
    async def build_datasheet(args: dict[str, Any]) -> dict[str, Any]:
        nonlocal tools_instance
        try:
            pdf_source = args.get("pdf_source", "")
            if not pdf_source:
                return _err("pdf_source is required")
            # Re-bind if source changed
            if tools_instance is None or tools_instance.pdf_path != pdf_source:
                if tools_instance is not None:
                    tools_instance.close()
                tools_instance = DatasheetTools(pdf_source)
            await asyncio.to_thread(
                tools_instance.build_datasheet,
                output_dir=args.get("output_dir"),
                output_stem=args.get("output_stem"),
                include_summaries=args.get("include_summaries", False),
                model=args.get("model"),
                force_rebuild=args.get("force_rebuild", False),
            )
            return _ok(tools_instance.get_artifact_manifest())
        except Exception as exc:
            return _err(str(exc))

    @tool(
        "get_section_text",
        "Read the extracted text for a page range (inclusive, 1-indexed). Use "
        "when you know WHERE to read -- pass start_page/end_page from ToC nodes "
        "to read specific sections. For a single page use the same value for "
        "both. Prefer reading whole sections rather than page-by-page. The text "
        "opens with a '=== Pages X-Y of N ===' header so you know your position "
        "in the document.",
        {
            "type": "object",
            "properties": {
                "start_page": {"type": "integer", "minimum": 1},
                "end_page": {"type": "integer", "minimum": 1},
            },
            "required": ["start_page", "end_page"],
        },
    )
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

    @tool(
        "search_text",
        "Search the full extracted text and return page-aware snippets with "
        "surrounding context. Use when you know WHAT to look for -- a parameter "
        "name, value, or keyword -- to locate it across the datasheet before "
        "reading specific sections. 'query' may be a single string or a list of "
        "strings to search several terms in one call. Each result includes the "
        "ToC 'breadcrumb' of the section containing the match; list searches also "
        "tag each result with the matching 'pattern'. Omit 'page' to search all.",
        {
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
    )
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

    @tool(
        "inspect_page",
        "Render a PDF page as a PNG image for visual inspection. Use when "
        "extracted text is garbled or insufficient -- tables with complex "
        "layouts, block diagrams, pin-out figures, timing diagrams. Optionally "
        "crop with top/bottom/left/right percentages (0.0-1.0). Pick `detail` "
        "to control vision-token cost: 'low' for layout overview, 'medium' "
        "(recommended default) for body text and table cells, 'high' for "
        "footnotes / subscripts / dense schematics.",
        {
            "type": "object",
            "properties": {
                "page": {"type": "integer", "minimum": 1},
                "region": {
                    "type": "object",
                    "description": "Crop region with top/bottom/left/right (0.0-1.0)",
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
    )
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

    @tool(
        "extract_table_markdown",
        "Re-extract a single page as layout-aware Markdown with proper table "
        "formatting. Use when get_section_text shows a garbled or misaligned "
        "table and you need clean | delimited rows for parameter extraction. "
        "Cheaper than inspect_page (text tokens vs vision tokens) but slower "
        "(~3s per page). Pass the 1-indexed page number from the PAGE marker.",
        {
            "type": "object",
            "properties": {
                "page": {"type": "integer", "minimum": 1},
            },
            "required": ["page"],
        },
    )
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

    return create_sdk_mcp_server(
        name="datasheetindex",
        version="1.0.0",
        tools=[
            build_datasheet,
            get_section_text,
            search_text,
            inspect_page,
            extract_table_markdown,
        ],
    )
