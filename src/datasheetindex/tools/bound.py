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

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from datasheetindex._version import package_version
from datasheetindex.core.artifact_cache import (
    ArtifactRecord,
    is_editable_install,
    read_sidecar,
    remove_sidecar,
    reuse_blocker,
    sha256_file,
    sha256_text,
    sidecar_path,
    write_sidecar,
)
from datasheetindex.core.engine import layout_engine
from datasheetindex.core.locate import TextLocation
from datasheetindex.core.locate import locate_text as locate_text_core
from datasheetindex.core.structure import (
    continuation_at_boundary,
    find_breadcrumb_for_page,
)
from datasheetindex.core.textfile import TextSearchMatch, extract_section_text
from datasheetindex.core.textfile import search_text as search_text_content
from datasheetindex.index import DatasheetIndex
from datasheetindex.llm.client import close_llm_client
from datasheetindex.llm.figure_captions import DEFAULT_MAX_FIGURE_CAPTIONS
from datasheetindex.models import DatasheetArtifacts, TocNode, TocQuality
from datasheetindex.tools.vision import Detail, inspect_page

if TYPE_CHECKING:
    import pymupdf

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _BuildOptions:
    output_dir: str
    output_stem: str | None
    include_summaries: bool
    model: str | None
    caption_figures: bool
    max_figure_captions: int

    def to_dict(self) -> dict[str, object]:
        """The cache key, as recorded in the sidecar.

        Recorded rather than inferred: ``TocNode.to_dict()`` omits empty fields,
        so an absent ``summary`` cannot be told apart from
        ``include_summaries=True`` that produced nothing.

        Built with ``dataclasses.asdict`` rather than a hand-written map so a
        field added to this dataclass is included automatically -- a
        hand-written map can silently omit a new field from the cache key,
        which would let an artifact built one way be served for a request
        that asked for another. A future field that is not JSON-safe now
        fails loudly at ``json.dumps`` instead of silently at the cache key.
        """
        return asdict(self)


_NO_TOC_HINT = (
    "This PDF has no usable table of contents, so there is no section map to "
    "plan from. Orient by reading pages 1-2 with get_section_text, then locate "
    "content with search_text and read around each hit with get_section_text. "
    "inspect_page renders a page as an image when the extracted text is unclear."
)


def _continuation_notes(text_content: str, start_page: int, end_page: int) -> list[str]:
    """Notes for content the requested range cuts at either boundary.

    States only that the publisher marked the adjacent page as continuing. It
    claims nothing about rows or column headers -- a continuation page often
    repeats its headers -- and never asserts that the range is complete.
    """
    notes: list[str] = []
    for title in continuation_at_boundary(text_content, start_page - 1):
        notes.append(
            f'=== NOTE: this range opens inside "{title}", which is '
            f"continued from page {start_page - 1}. ==="
        )
    for title in continuation_at_boundary(text_content, end_page):
        notes.append(
            f'=== NOTE: "{title}" is continued on page {end_page + 1}, '
            f"which is outside this range. ==="
        )
    return notes


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

        The import and the call both happen inside ``layout_engine()``. Doing
        the import outside it would let a concurrent ``classic_tables()``
        restore a stale ``None`` over the freshly installed hook, after which
        every call here raises ``TypeError``.
        """
        total = len(self.doc)
        if page < 1 or page > total:
            raise ValueError(f"page must be between 1 and {total}")
        with layout_engine() as pymupdf4llm:
            return pymupdf4llm.to_markdown(
                self.doc, pages=[page - 1], show_progress=False
            )

    def build_datasheet(
        self,
        output_dir: str | None = None,
        output_stem: str | None = None,
        include_summaries: bool = False,
        model: str | None = None,
        force_rebuild: bool = False,
        caption_figures: bool = True,
        max_figure_captions: int = DEFAULT_MAX_FIGURE_CAPTIONS,
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
            caption_figures=caption_figures,
            max_figure_captions=max_figure_captions,
        )
        if (
            not force_rebuild
            and self._artifacts is not None
            and self._build_options == options
            # A degraded artifact must not be served from memory either. The MCP
            # path holds one instance per document across a session, so this is
            # the commonest retry there is; without this condition the disk rule
            # would never get a chance to run.
            and not self._artifacts.llm_enrichment_incomplete
            and self._artifacts.json_path is not None
            and self._artifacts.json_path.exists()
            and self._artifacts.text_path is not None
            and self._artifacts.text_path.exists()
        ):
            return self._artifacts

        stem = self._index.artifact_stem(output_stem)
        sidecar = sidecar_path(resolved_output_dir, stem)

        if not force_rebuild:
            reused = self._reuse_from_disk(sidecar, options)
            if reused is not None:
                self._artifacts = reused
                self._build_options = options
                return reused

        # Invalidate, write data, publish. Removing the sidecar first means a
        # concurrent reader either finds no sidecar and rebuilds, or finds one
        # and must match both artifact hashes.
        remove_sidecar(sidecar)

        # Fingerprint the source BEFORE the build reads it, not after: a build
        # can take several seconds (enrich_with_table_counts re-opens the path
        # from disk in its own scan workers), and hashing post-build would
        # record whatever replaced the file during that window rather than
        # what PyMuPDF actually built from. Best effort -- a fingerprinting
        # problem here must skip the sidecar, never fail the build.
        source_sha256: str | None = None
        source_size: int | None = None
        try:
            source_path = Path(self._index._resolve_pdf_source())
            source_sha256 = sha256_file(source_path)
            source_size = source_path.stat().st_size
        except Exception:
            logger.debug(
                "Could not fingerprint the source before the build; the "
                "sidecar write will be skipped",
                exc_info=True,
            )

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
                caption_figures=caption_figures,
                max_figure_captions=max_figure_captions,
            )
        finally:
            close_llm_client(llm_callable)

        self._write_build_sidecar(
            sidecar, options, artifacts, source_sha256, source_size
        )

        self._artifacts = artifacts
        self._build_options = options
        return artifacts

    def _reuse_from_disk(
        self, sidecar: Path, options: _BuildOptions
    ) -> DatasheetArtifacts | None:
        """Return artifacts loaded from disk, or None to rebuild.

        Every failure degrades to a rebuild, so the caller needs no error
        handling. Artifact content is validated by hashing the bytes actually
        read: hashing after the read rather than stat-ing before it closes the
        mixed-generation window entirely instead of narrowing it, so a pair
        straddling a concurrent write, or left mixed by a crash between the two
        writes, fails and rebuilds.

        Every rejection goes through one log line with a stable token, so a test
        can assert *which* check rejected a record rather than only that a
        rebuild happened.
        """
        if is_editable_install():
            logger.debug("Not reusing on-disk artifacts: %s", "editable_install")
            return None

        record = read_sidecar(sidecar)
        if record is None:
            logger.debug("Not reusing on-disk artifacts: %s", "no_sidecar")
            return None

        # Resolve first: a fresh instance holding a URL has nothing on disk to
        # hash yet, and a local path may still need the WSL/Windows translation.
        # Hash the resolved file.
        try:
            source_path = self._index._resolve_pdf_source()
        except Exception:
            logger.debug(
                "Not reusing on-disk artifacts: %s",
                "source_unresolvable",
                exc_info=True,
            )
            return None

        try:
            blocker = reuse_blocker(
                record,
                source_path=source_path,
                build_options=options.to_dict(),
                running_version=package_version(),
            )
        except OSError:
            logger.debug("Not reusing on-disk artifacts: %s", "source_unreadable")
            return None
        if blocker is not None:
            logger.debug("Not reusing on-disk artifacts: %s", blocker)
            return None

        directory = Path(options.output_dir)
        json_path = directory / record.json_name
        text_path = directory / record.text_name
        try:
            json_text = json_path.read_text(encoding="utf-8")
            text_content = text_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            logger.debug("Not reusing on-disk artifacts: %s", "artifact_unreadable")
            return None

        if sha256_text(json_text) != record.json_sha256:
            logger.debug("Not reusing on-disk artifacts: %s", "json_hash_mismatch")
            return None
        if sha256_text(text_content) != record.text_sha256:
            logger.debug("Not reusing on-disk artifacts: %s", "text_hash_mismatch")
            return None

        try:
            json_data = json.loads(json_text)
            nodes = [TocNode.from_dict(entry) for entry in json_data["toc"]]
            toc_quality = TocQuality.from_dict(record.toc_quality)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            logger.warning(
                "Not reusing on-disk artifacts: %s",
                "deserialization_failed",
                exc_info=True,
            )
            return None

        logger.info("Reusing valid on-disk artifacts from %s", json_path)
        return DatasheetArtifacts(
            json_path=json_path,
            text_path=text_path,
            json_data=json_data,
            text_content=text_content,
            toc_quality=toc_quality,
            nodes=nodes,
            llm_enrichment_incomplete=record.llm_enrichment_incomplete,
            llm_enrichment_notes=record.llm_enrichment_notes,
        )

    def _write_build_sidecar(
        self,
        sidecar: Path,
        options: _BuildOptions,
        artifacts: DatasheetArtifacts,
        source_sha256: str | None,
        source_size: int | None,
    ) -> None:
        """Record this build's fingerprint. Best effort.

        A sidecar write failure must not fail the build: the artifacts are
        correct and caching is infrastructure, mirroring ``_safe_close`` in
        ``defs.py`` where cleanup failure is logged rather than allowed to
        discard a good result.

        The two deliverables' hashes ARE taken from the files as written
        rather than from in-memory values, so the record cannot disagree with
        what is on disk -- that reasoning holds because they do not exist
        until the build writes them. It does NOT extend to the source: that
        file exists before and throughout the build, so hashing it here,
        after ``self._index.build()`` has already returned, would fingerprint
        whatever is on disk *now*, which is not necessarily what PyMuPDF
        actually read if the source was replaced in place while the build was
        running. ``source_sha256``/``source_size`` are therefore passed in,
        captured by the caller before the build started. As a second guard,
        the source is re-hashed here and compared against that pre-build
        value; a mismatch means the source changed during the build, and the
        sidecar write is skipped rather than recording the wrong generation.
        """
        if source_sha256 is None or source_size is None:
            logger.debug(
                "No pre-build source fingerprint available; skipping the sidecar write"
            )
            return
        try:
            if artifacts.json_path is None or artifacts.text_path is None:
                return
            source_path = Path(self._index._resolve_pdf_source())
            if sha256_file(source_path) != source_sha256:
                logger.warning(
                    "Source %s changed while the build was running; skipping "
                    "the sidecar write so a mismatched generation is never "
                    "recorded",
                    source_path,
                )
                return
            quality = artifacts.toc_quality
            record = ArtifactRecord(
                source_sha256=source_sha256,
                source_size=source_size,
                build_options=options.to_dict(),
                datasheetindex_version=package_version(),
                json_name=artifacts.json_path.name,
                json_sha256=sha256_file(artifacts.json_path),
                text_name=artifacts.text_path.name,
                text_sha256=sha256_file(artifacts.text_path),
                toc_quality=quality.to_dict() if quality is not None else {},
                llm_enrichment_incomplete=artifacts.llm_enrichment_incomplete,
                llm_enrichment_notes=artifacts.llm_enrichment_notes,
            )
            write_sidecar(sidecar, record)
        except Exception:
            logger.warning(
                "Could not write the build sidecar; the artifacts are valid but "
                "will be rebuilt next time",
                exc_info=True,
            )

    def get_artifact_manifest(self) -> dict[str, object]:
        """Return a compact summary of the currently built artifacts.

        When the PDF has no usable ToC the agent has no section map to plan
        from, and the tool descriptions' "pass start_page/end_page from ToC
        nodes" advice is dead. Carry a hint that redirects it to search_text.
        Keyed off the returned ToC being empty -- the outcome the agent faces --
        not off LLM availability: a rejected fallback candidate leaves the ToC
        empty with credentials present.
        """
        artifacts = self._require_artifacts()
        manifest: dict[str, object] = {
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
        if not manifest["toc"]:
            manifest["hint"] = _NO_TOC_HINT
        return manifest

    def get_section_text(self, start_page: int, end_page: int) -> str:
        """Return extracted text for a page range from the latest build.

        The result has three parts, in order:

        1. A position header: ``=== Page X of N ===`` for a single-page read,
           ``=== Pages X-Y of N ===`` for a multi-page range.
        2. Zero or more ``=== NOTE: ... ===`` lines -- present when the
           requested range cuts content the publisher marked as continuing
           onto an adjacent page, at the head of the range, the tail, or both;
           either boundary can carry more than one marker (e.g. a page opening
           with two continued tables). The ``===`` wrapper is what marks the
           line as tool framing rather than document content: real datasheets
           sometimes contain their own literal ``NOTE:`` lines in body text.
        3. The section text, WITH ``--- PAGE N ---`` markers so the agent can
           orient within the range.

        The absence of a NOTE means none was detected. It is not a completeness
        claim: content can spill across a page break with no marker at all.
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
        notes = _continuation_notes(artifacts.text_content, start_page, end_page)
        section = extract_section_text(artifacts.text_content, start_page, end_page)
        return "\n".join([header, *notes, section])

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
