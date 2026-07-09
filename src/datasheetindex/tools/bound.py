"""The document-bound datasheet tools, independent of any agent framework.

:class:`DatasheetTools` wraps a single PDF and exposes the operations the tool
handlers delegate to (build, section/text queries, page inspection, coordinate
grounding). It imports no agent-framework code, so both the framework-neutral
tool defs (:mod:`datasheetindex.tools.defs`) and the SDK adapter
(:mod:`datasheetindex.tools.registry`) can build on it with a one-directional
dependency (``registry -> defs -> bound``).

For backward compatibility, ``DatasheetTools`` is re-exported from
:mod:`datasheetindex.tools.registry`, :mod:`datasheetindex.tools`, and the
top-level :mod:`datasheetindex` package.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from datasheetindex.core.locate import TextLocation
from datasheetindex.core.locate import locate_text as locate_text_core
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

    def locate_text(
        self,
        query: str | list[str],
        *,
        page: int | None = None,
        max_results: int = 20,
    ) -> list[TextLocation]:
        """Map a string to its bounding box(es) on a page.

        Works off the live PDF (`self.doc`); unlike `search_text`/`get_section_text`
        it does NOT require `build_datasheet` to have been called.
        """
        return locate_text_core(self.doc, query, page=page, max_results=max_results)

    def extract_table_markdown(self, page: int) -> str:
        """Extract a single page as layout-aware markdown with table structure.

        Requires the ``[layout]`` extra (``pymupdf4llm``). Returns markdown
        with proper table formatting using ``|`` delimiters.
        """
        total = len(self.doc)
        if page < 1 or page > total:
            raise ValueError(f"page must be between 1 and {total}")
        # Imported by name, like the other optional dependencies, so a checker
        # does not require the [layout] extra to be installed.
        try:
            pymupdf4llm = importlib.import_module("pymupdf4llm")
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
