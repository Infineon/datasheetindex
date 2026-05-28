"""Main DatasheetIndex class."""

from __future__ import annotations

import json
import logging
import os
import re
import ssl
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pymupdf

from datasheetindex.core.annotations import (
    enrich_with_cross_references,
    enrich_with_footnote_markers,
)
from datasheetindex.core.preamble import generate_preamble
from datasheetindex.core.quality import assess_toc_quality
from datasheetindex.core.structure import (
    build_tree,
    enrich_with_continued_tables,
    enrich_with_table_counts,
    extract_toc,
)
from datasheetindex.core.textfile import generate_text
from datasheetindex.llm.client import close_llm_client
from datasheetindex.models import DatasheetArtifacts

if TYPE_CHECKING:
    from datasheetindex.llm.client import LlmCallable

logger = logging.getLogger(__name__)

TOC_FALLBACK_THRESHOLD = 0.3
DOWNLOAD_TIMEOUT_SECONDS = 30
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
DOWNLOAD_MAX_SIZE = 100 * 1024 * 1024  # 100 MB
PDF_HEADER_SCAN_BYTES = 1024

_OUTPUT_DIR_ENV_VAR = "DATASHEETINDEX_OUTPUT_DIR"


def resolve_default_output_dir() -> str:
    """Default ``output_dir`` for ``DatasheetIndex.build`` when none is given.

    The CLI and batch entry points pass an explicit ``"output"`` so dev/interactive
    use writes to ``./output/``. Other callers (Python API, MCP servers, hosted
    agents) that omit ``output_dir`` land here, where ``./output`` would fail on
    a read-only container root.

    Resolution: ``$DATASHEETINDEX_OUTPUT_DIR`` if set (lets a deployment pin a
    workspace), otherwise ``<tempdir>/datasheetindex-<uid>`` -- always writable
    and namespaced per-UID so users on a shared host don't collide.
    """
    env_value = os.environ.get(_OUTPUT_DIR_ENV_VAR, "").strip()
    if env_value:
        return env_value
    uid = getattr(os, "getuid", lambda: None)()
    leaf = f"datasheetindex-{uid}" if uid is not None else "datasheetindex"
    return str(Path(tempfile.gettempdir()) / leaf)


def _urlopen_with_ssl_fallback(url: str) -> Any:
    """Open a URL, retrying without SSL verification on certificate errors.

    Semiconductor vendor sites (e.g. mxic.com.tw) sometimes use self-signed
    or improperly chained certificates. We try the secure path first and
    only fall back to unverified SSL when certificate validation fails.
    """
    try:
        return urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS)
    except urllib.error.URLError as exc:
        if not isinstance(exc.reason, ssl.SSLCertVerificationError):
            raise
        logger.warning(
            "SSL certificate verification failed for %s; retrying without verification",
            url,
        )
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return urllib.request.urlopen(
            url, timeout=DOWNLOAD_TIMEOUT_SECONDS, context=ctx
        )


def _is_http_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _sanitize_filename_part(value: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", value)
    return sanitized[:200]


def _looks_like_pdf(data: bytes) -> bool:
    return b"%PDF-" in data[:PDF_HEADER_SCAN_BYTES]


class DatasheetIndex:
    """Pre-processes a datasheet PDF into agent-ready artifacts."""

    def __init__(self, pdf_path: str) -> None:
        self.pdf_path = pdf_path
        self._doc: pymupdf.Document | None = None
        self._resolved_pdf_path: str | None = None
        self._temp_pdf_path: Path | None = None

    def __enter__(self) -> DatasheetIndex:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        self.close()

    @property
    def doc(self) -> pymupdf.Document:
        """Lazy-open the PDF document."""
        if self._doc is None:
            source_path = self._resolve_pdf_source()
            try:
                self._doc = pymupdf.open(source_path)
            except Exception:
                self._cleanup_temp_pdf()
                raise
        return self._doc

    def close(self) -> None:
        """Close the underlying PDF document."""
        if self._doc is not None:
            self._doc.close()
            self._doc = None
        self._cleanup_temp_pdf()

    def _resolve_pdf_source(self) -> str:
        if self._resolved_pdf_path is not None:
            return self._resolved_pdf_path
        if _is_http_url(self.pdf_path):
            self._resolved_pdf_path = self._download_pdf(self.pdf_path)
        else:
            self._resolved_pdf_path = self.pdf_path
        return self._resolved_pdf_path

    def _download_pdf(self, url: str) -> str:
        temp_path: Path | None = None
        header_probe = b""
        try:
            response = _urlopen_with_ssl_fallback(url)
            with response:
                # Validate final URL after redirects to prevent SSRF
                final_url = response.geturl()
                if not _is_http_url(final_url):
                    raise ValueError("URL redirected to a non-HTTP location")

                status = response.getcode()
                if status is not None and status >= 400:
                    raise ValueError(
                        f"Failed to download PDF from URL (status {status})"
                    )

                content_type = response.headers.get("Content-Type", "").lower()
                if content_type and "pdf" not in content_type:
                    raise ValueError(
                        "URL did not return a PDF content type "
                        f"(received: {content_type})"
                    )

                with tempfile.NamedTemporaryFile(
                    suffix=".pdf", delete=False
                ) as temp_file:
                    temp_path = Path(temp_file.name)
                    total_downloaded = 0
                    while True:
                        chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                        if not chunk:
                            break
                        total_downloaded += len(chunk)
                        if total_downloaded > DOWNLOAD_MAX_SIZE:
                            raise ValueError(
                                "Download exceeds maximum size of "
                                f"{DOWNLOAD_MAX_SIZE} bytes"
                            )
                        if len(header_probe) < PDF_HEADER_SCAN_BYTES:
                            remaining = PDF_HEADER_SCAN_BYTES - len(header_probe)
                            header_probe += chunk[:remaining]
                        temp_file.write(chunk)
        except urllib.error.HTTPError as exc:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            raise ValueError(
                f"Failed to download PDF from URL: HTTP {exc.code}"
            ) from exc
        except urllib.error.URLError as exc:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            raise ValueError(f"Failed to download PDF from URL: {exc.reason}") from exc
        except ValueError:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            raise
        except BaseException:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            raise

        if temp_path.stat().st_size == 0:
            temp_path.unlink(missing_ok=True)
            raise ValueError("Downloaded PDF is empty")
        if not _looks_like_pdf(header_probe):
            temp_path.unlink(missing_ok=True)
            raise ValueError("Downloaded content is not a valid PDF")

        self._temp_pdf_path = temp_path
        return str(temp_path)

    def _source_file_name(self) -> str:
        if _is_http_url(self.pdf_path):
            parsed = urllib.parse.urlparse(self.pdf_path)
            name = Path(parsed.path).name
            return name or "downloaded.pdf"
        return Path(self.pdf_path).name

    def _output_stem(self) -> str:
        stem = Path(self._source_file_name()).stem or "datasheet"
        return _sanitize_filename_part(stem)

    def _cleanup_temp_pdf(self) -> None:
        if self._temp_pdf_path is not None:
            self._temp_pdf_path.unlink(missing_ok=True)
            self._temp_pdf_path = None
        if _is_http_url(self.pdf_path):
            self._resolved_pdf_path = None

    def build(
        self,
        output_dir: str | None = None,
        include_summaries: bool = False,
        llm_callable: LlmCallable | None = None,
        output_stem: str | None = None,
    ) -> DatasheetArtifacts:
        """Build the two deliverables: enriched ToC JSON and page-matched text.

        When ``llm_callable`` is provided, low-quality ToCs are regenerated
        via LLM and optional section summaries can be added. ``output_stem``
        optionally overrides the default stem derived from the source filename.

        ``output_dir=None`` (or an empty/whitespace string) resolves to
        ``$DATASHEETINDEX_OUTPUT_DIR`` if set, otherwise a UID-namespaced
        subdirectory under the OS tempdir. The CLI passes its own
        ``"output"`` default explicitly.

        Returns a DatasheetArtifacts with paths, data, and quality info.
        """
        if output_dir is None or not output_dir.strip():
            output_dir = resolve_default_output_dir()
        t_start = time.monotonic()
        doc = self.doc
        t_doc = time.monotonic()
        total_pages = len(doc)
        logger.info("PDF opened (%d pages) in %.1fs", total_pages, t_doc - t_start)

        pdf_name = self._output_stem()
        if output_stem is not None:
            pdf_name = _sanitize_filename_part(output_stem) or "datasheet"

        # 1. Generate page-matched text
        text_content = generate_text(doc)
        t_text = time.monotonic()
        logger.info("Text extraction done in %.1fs", t_text - t_doc)

        # 2. Generate preamble
        preamble = generate_preamble(doc)

        # 3. Extract ToC, build tree, enrich with table counts
        raw_toc = extract_toc(doc)
        nodes = build_tree(raw_toc, total_pages)
        resolved_path = self._resolved_pdf_path
        enrich_with_table_counts(nodes, doc, pdf_path=resolved_path)
        t_tables = time.monotonic()
        logger.info("Table counting done in %.1fs", t_tables - t_text)
        enrich_with_continued_tables(nodes, text_content)
        enrich_with_footnote_markers(nodes, text_content)
        enrich_with_cross_references(nodes, text_content)

        # 4. Assess ToC quality
        toc_quality = assess_toc_quality(nodes, total_pages)
        logger.info(
            "ToC quality: score=%.2f, entries=%d, coverage=%.0f%%",
            toc_quality.score,
            toc_quality.entry_count,
            toc_quality.page_coverage * 100,
        )

        active_llm_callable = llm_callable
        owns_llm_callable = False
        if active_llm_callable is None and toc_quality.score < TOC_FALLBACK_THRESHOLD:
            active_llm_callable = self._try_create_default_llm_client()
            owns_llm_callable = active_llm_callable is not None

        try:
            # 5. LLM fallback: regenerate ToC if quality is poor
            if active_llm_callable and toc_quality.score < TOC_FALLBACK_THRESHOLD:
                t_llm = time.monotonic()
                logger.info(
                    "ToC quality below threshold (%.2f < %.2f), running LLM fallback",
                    toc_quality.score,
                    TOC_FALLBACK_THRESHOLD,
                )
                try:
                    from datasheetindex.llm.toc_fallback import generate_toc_from_text

                    nodes = generate_toc_from_text(
                        text_content, total_pages, active_llm_callable
                    )
                    enrich_with_table_counts(nodes, doc, pdf_path=resolved_path)
                    enrich_with_continued_tables(nodes, text_content)
                    enrich_with_footnote_markers(nodes, text_content)
                    enrich_with_cross_references(nodes, text_content)
                    toc_quality = assess_toc_quality(nodes, total_pages)
                    logger.info(
                        "LLM ToC fallback done in %.1fs", time.monotonic() - t_llm
                    )
                except Exception:
                    logger.warning(
                        "LLM ToC fallback failed; using original ToC",
                        exc_info=True,
                    )

            # 6. LLM summaries: only when explicitly requested
            if active_llm_callable and include_summaries:
                t_sum = time.monotonic()
                from datasheetindex.llm.summarizer import add_summaries

                add_summaries(nodes, text_content, active_llm_callable)
                logger.info("LLM summaries done in %.1fs", time.monotonic() - t_sum)

            logger.info("Total build time: %.1fs", time.monotonic() - t_start)

            # 7. Build JSON structure
            json_data = {
                "source": self._source_file_name(),
                "total_pages": total_pages,
                "preamble": preamble,
                "toc_quality": {
                    "score": toc_quality.score,
                    "entry_count": toc_quality.entry_count,
                    "max_depth": toc_quality.max_depth,
                    "page_coverage": toc_quality.page_coverage,
                    "recommend_summaries": toc_quality.recommend_summaries,
                },
                "toc": [node.to_dict() for node in nodes],
            }

            # 8. Write output files
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)

            json_path = out / f"{pdf_name}.json"
            text_path = out / f"{pdf_name}.txt"

            json_path.write_text(
                json.dumps(json_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            text_path.write_text(text_content, encoding="utf-8")

            return DatasheetArtifacts(
                json_path=json_path,
                text_path=text_path,
                json_data=json_data,
                text_content=text_content,
                toc_quality=toc_quality,
            )
        finally:
            if owns_llm_callable:
                close_llm_client(active_llm_callable)

    def _try_create_default_llm_client(self) -> LlmCallable | None:
        try:
            from datasheetindex.llm.client import create_llm_client

            return create_llm_client(model="gpt-4.1")
        except (ImportError, ValueError, OSError):
            return None
