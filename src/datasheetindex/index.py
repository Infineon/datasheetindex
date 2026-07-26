"""Main DatasheetIndex class."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pymupdf

from datasheetindex.core.annotations import (
    enrich_with_cross_references,
    enrich_with_footnote_markers,
)
from datasheetindex.core.artifact_cache import atomic_write_text
from datasheetindex.core.figures import DEFAULT_MIN_AREA_PCT
from datasheetindex.core.preamble import build_front_matter
from datasheetindex.core.quality import assess_toc_quality
from datasheetindex.core.structure import (
    build_tree,
    enrich_with_continued_tables,
    enrich_with_table_counts,
    extract_toc,
)
from datasheetindex.core.textfile import scan_pages
from datasheetindex.llm.client import close_llm_client, get_vision_client
from datasheetindex.llm.figure_captions import (
    DEFAULT_MAX_FIGURE_CAPTIONS,
    caption_figures_in_place,
    eligible_caption_count,
    validate_max_figure_captions,
)
from datasheetindex.models import DatasheetArtifacts, TocQuality

if TYPE_CHECKING:
    from datasheetindex.llm.client import LlmCallable

logger = logging.getLogger(__name__)

TOC_FALLBACK_THRESHOLD = 0.3
DOWNLOAD_TIMEOUT_SECONDS = 30
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
DOWNLOAD_MAX_SIZE = 100 * 1024 * 1024  # 100 MB
PDF_HEADER_SCAN_BYTES = 1024

_OUTPUT_DIR_ENV_VAR = "DATASHEETINDEX_OUTPUT_DIR"

#: Client provenances that also sanction section summaries. ``build()`` shares
#: one client across three branches, so "a client exists" is not the same
#: statement as "the caller signed up for per-section LLM calls":
#:
#: - ``"caller"`` -- handed in explicitly, so every LLM branch is sanctioned.
#: - ``"toc_fallback"`` -- self-created because the native ToC is too weak to
#:   navigate. That was already an implicit opt-in through 0.24.0, and it is
#:   also the case where summaries help most.
#: - ``"figure_captions"`` -- self-created *only* because the document carries
#:   raster regions. Deliberately absent. Gating summaries on availability
#:   instead let one inserted image turn ``include_summaries=True,
#:   llm_callable=None`` from a no-op into one LLM call per ToC section, a cost
#:   ``max_figure_captions`` does not bound and nothing disclosed.
#:
#: A new construction site is unsanctioned until it is listed here, which is
#: the direction that fails safely.
_SUMMARY_CLIENT_ORIGINS = frozenset({"caller", "toc_fallback"})


def _minimum_fallback_candidate_entries(total_pages: int) -> int:
    """Minimum entry count before an LLM-generated ToC is trusted."""

    if total_pages <= 3:
        return 1
    if total_pages <= 8:
        return 2
    return 3


def _accept_llm_toc_candidate(
    baseline: TocQuality,
    candidate: TocQuality,
    *,
    total_pages: int,
) -> tuple[bool, str]:
    """Decide whether an LLM-generated ToC is safe to replace the baseline."""

    if candidate.entry_count == 0:
        return False, "candidate has no entries"

    if candidate.score <= baseline.score:
        return (
            False,
            "candidate score did not improve "
            f"({candidate.score:.2f} <= {baseline.score:.2f})",
        )

    # The entry-count floor exists to stop a degenerate candidate (a lone node
    # whose end_page build_tree() extended to the last page, scoring well on
    # coverage) from displacing a real ToC. It only applies when there is a real
    # ToC to protect: the fallback's most common trigger is a PDF with no
    # bookmarks at all, and there a thin ToC still beats no ToC.
    if baseline.entry_count > 0:
        min_entries = _minimum_fallback_candidate_entries(total_pages)
        if candidate.entry_count < min_entries:
            return (
                False,
                "candidate has too few entries "
                f"({candidate.entry_count} < {min_entries})",
            )

    if (
        baseline.page_coverage > 0
        and candidate.page_coverage + 0.05 < baseline.page_coverage
    ):
        return (
            False,
            "candidate page coverage dropped materially versus baseline",
        )

    return True, "candidate improved quality without material regressions"


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


# wsl.exe answers a list query immediately; anything slower means the
# subsystem is starting or wedged, and the caller should not wait on it.
WSL_QUERY_TIMEOUT_SECONDS = 5


def _is_windows() -> bool:
    """Whether this is a Windows host.

    A function, not an inline ``sys.platform`` test, so a test can override the
    platform for one module instead of lying to every library in the process
    that reads ``sys.platform``.
    """
    return sys.platform == "win32"


def _wsl_distros() -> list[str]:
    """Names of the installed WSL distributions, or empty if none/unavailable."""
    # stdout goes to a file, not a PIPE, and the wait is process.wait(). Under
    # subprocess.run(capture_output=True), a timeout means kill() followed by
    # communicate(), which waits for EOF on the pipe -- and wsl.exe brokers
    # through the WSL service and can leave a handle-inheriting helper behind,
    # so that EOF may never arrive and the "timeout" would not bound anything.
    # This is the same trap the scan worker avoids for the same reason.
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            out_path = os.path.join(tmpdir, "distros.txt")
            with open(out_path, "wb") as out_handle:
                process = subprocess.Popen(
                    ["wsl.exe", "--list", "--quiet"],
                    stdin=subprocess.DEVNULL,
                    stdout=out_handle,
                    stderr=subprocess.DEVNULL,
                    # Same reason the scan worker gets it: this runs on a
                    # console-less MCP server, and without it every query
                    # flashes a window.
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                try:
                    returncode = process.wait(timeout=WSL_QUERY_TIMEOUT_SECONDS)
                except BaseException:
                    process.kill()
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        process.wait(timeout=WSL_QUERY_TIMEOUT_SECONDS)
                    raise
            with open(out_path, "rb") as handle:
                raw = handle.read()
    except (OSError, subprocess.SubprocessError):
        # Logged, not silent: without this, a wedged WSL service and a host
        # with no WSL at all produce the same empty result and the same
        # unhelpful "no such file", with nothing to tell them apart.
        logger.debug("WSL distribution query failed", exc_info=True)
        return []
    if returncode != 0:
        logger.debug("wsl.exe --list exited %d", returncode)
        return []

    # wsl.exe writes UTF-16LE by default, but UTF-8 when WSL_UTF8=1 is set
    # (WSL 0.64+), and that setting is inherited by this child. UTF-8 output is
    # almost always even-length, so decoding it as UTF-16 does NOT raise -- it
    # silently yields mojibake with no line breaks, collapsing every distro
    # into one junk name that matches nothing. Sniff on a NUL byte instead of
    # trusting a decode error to catch it.
    text = raw.decode("utf-16-le" if b"\x00" in raw else "utf-8", "replace")

    # A name carrying a separator would build a surprising UNC path. These are
    # local machine config and only ever feed os.path.exists, so this is
    # hygiene rather than a security boundary.
    return [
        name
        for name in (line.strip() for line in text.splitlines())
        if name and "\\" not in name and "/" not in name and name not in (".", "..")
    ]


def _windows_paths_for_posix(posix_path: str) -> Iterator[str]:
    """Windows spellings of a POSIX path, most authoritative first.

    ``/mnt/<drive>/...`` is WSL's own mount of a Windows drive, so it maps back
    deterministically and is yielded first. Anything else is assumed to live in
    a distro's filesystem, which Windows reaches over the ``wsl.localhost`` UNC
    share -- PyMuPDF opens those directly, verified against a real datasheet.

    A generator, so a caller that resolves on the ``/mnt`` candidate never
    queries WSL at all. That ordering matters: probing a UNC path against a
    *stopped* distro starts it, which takes tens of seconds and is the one
    unbounded operation on this path.
    """
    if not posix_path.startswith("/"):
        return
    parts = [part for part in posix_path.split("/") if part]
    if not parts:
        return

    # Joined with literal backslashes, never os.path.join: these are Windows
    # paths by construction, and on a POSIX host (where the tests run) join
    # would splice them with forward slashes into "C:\/Users/...".
    if (
        parts[0] == "mnt"
        and len(parts) >= 3
        and len(parts[1]) == 1
        and parts[1].isalpha()
    ):
        yield f"{parts[1].upper()}:\\" + "\\".join(parts[2:])

    tail = "\\".join(parts)
    for distro in _wsl_distros():
        yield f"\\\\wsl.localhost\\{distro}\\{tail}"


# A drive-letter path, in either slash style: "C:\...", "c:/...", or a bare
# "C:\". Anchored, so a POSIX path can never match.
_WINDOWS_DRIVE_RE = re.compile(r"^([A-Za-z]):[\\/](.*)$", re.S)

# The UNC share Windows uses to reach a distro's filesystem, under both the
# current "wsl.localhost" host and the older "wsl$" spelling. Case-insensitive
# because Windows paths are: "\\WSL.localhost\..." is the same share, and an
# anchored case-sensitive pattern would simply fail to match it.
# The distro is captured, not skipped -- see _posix_paths_for_windows.
_WSL_UNC_RE = re.compile(r"^\\\\wsl(?:\.localhost|\$)\\([^\\]+)\\(.*)$", re.I | re.S)


def _posix_paths_for_windows(windows_path: str) -> Iterator[str]:
    """POSIX spellings of a Windows path, for a server running inside WSL.

    The mirror of :func:`_windows_paths_for_posix`, and it earns its place in
    the configuration we actually recommend: with the server in WSL, a user who
    copies a path out of Windows Explorer -- or pastes one from a colleague --
    hands the agent ``C:\\Users\\me\\ds.pdf``. That file is perfectly readable
    from the distro at ``/mnt/c/Users/me/ds.pdf``; nothing was wrong except the
    spelling.

    ``/mnt`` is WSL's default automount root. A distro that has overridden
    ``automount.root`` in ``/etc/wsl.conf`` simply finds no candidate here and
    gets its original path back, which is the same outcome as before.
    """
    unc = _WSL_UNC_RE.match(windows_path)
    if unc is not None:
        # Only unwrap a path that names THIS distro. The share is distro-scoped,
        # so \\wsl.localhost\Debian\home\y\ds.pdf is not our file -- but strip
        # the prefix blindly and it becomes /home/y/ds.pdf, which on a machine
        # with the same user in two distros silently resolves to a *different*
        # document that exists. A wrong answer that looks valid is the one
        # outcome this module refuses elsewhere (see the short-result check in
        # core/structure.py), so it must not be introduced here.
        #
        # Unlike the Windows-side direction, we are not guessing: the path names
        # the distro and WSL exports our own name into the environment.
        distro, tail = unc.group(1), unc.group(2)
        current = os.environ.get("WSL_DISTRO_NAME", "")
        if tail and current and distro.lower() == current.lower():
            yield "/" + tail.replace("\\", "/")
        return

    drive = _WINDOWS_DRIVE_RE.match(windows_path)
    if drive is not None:
        tail = drive.group(2).replace("\\", "/")
        # A bare "C:\" would yield "/mnt/c/", a real directory that exists, so
        # _resolve_local_path would "resolve" to it and the caller's not-found
        # error would name a path the user never passed.
        if tail:
            yield f"/mnt/{drive.group(1).lower()}/{tail}"


def _resolve_local_path(path: str) -> str:
    """Map a path onto this filesystem when the caller's namespace differs.

    VS Code opening a WSL folder from Windows is the case that motivates this:
    the editor, the agent and the files live in the distro, but a
    gallery-installed MCP server runs on the Windows host, where
    ``/home/you/ds.pdf`` simply does not exist. The agent has no way to know
    that and retries the same path, so the tool has to bridge it.

    Translation runs in **both** directions, because the split is symmetric and
    only the direction we happened to hit first would otherwise work. A server
    in WSL -- the setup we recommend, since it also avoids the Windows pool
    deadlock -- is just as likely to be handed ``C:\\Users\\you\\ds.pdf``,
    copied out of Windows Explorer, for a file it can read at ``/mnt/c/...``.

    Only consulted when the literal path does not resolve, so a path that is
    already correct for this host is never rewritten.

    Scope: the *input* PDF only. ``output_dir`` is deliberately not translated;
    a directory that does not exist yet cannot be probed with ``exists()``, and
    the artifact paths returned to the agent would still be in the server's
    namespace. Running the server inside WSL is the fix for that half.
    """
    if not path or os.path.exists(path):
        return path

    candidates = (
        _windows_paths_for_posix(path)
        if _is_windows()
        else _posix_paths_for_windows(path)
    )
    for candidate in candidates:
        if os.path.exists(candidate):
            logger.info("Resolved %s to %s", path, candidate)
            return candidate

    # Fall through unchanged: the caller's own "not found" error names the path
    # the user actually passed, which is the more useful thing to report.
    return path


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
            self._resolved_pdf_path = _resolve_local_path(self.pdf_path)
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

    def artifact_stem(self, output_stem: str | None) -> str:
        """The stem used for ``<stem>.json``, ``<stem>.txt`` and the sidecar.

        ``build()`` and ``DatasheetTools.build_datasheet`` must agree on this
        exactly, or the sidecar would be looked for beside files it does not
        describe. One derivation, two callers.
        """
        if output_stem is not None:
            return _sanitize_filename_part(output_stem) or "datasheet"
        return self._output_stem()

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
        caption_figures: bool = True,
        max_figure_captions: int = DEFAULT_MAX_FIGURE_CAPTIONS,
    ) -> DatasheetArtifacts:
        """Build the two deliverables: enriched ToC JSON and page-matched text.

        When ``llm_callable`` is provided, low-quality ToCs are regenerated
        via LLM and optional section summaries can be added. ``output_stem``
        optionally overrides the default stem derived from the source filename.

        ``include_summaries`` needs a client whose provenance sanctions it (see
        ``_SUMMARY_CLIENT_ORIGINS``): one the caller supplied, or the one the
        weak-ToC fallback creates for itself. A client self-created purely to
        caption figures does **not** enable summaries, so a keyless build
        produces none whether or not the document has figures.

        ``caption_figures`` (default ``True``) names raster figure regions
        with a vision model, capped at ``max_figure_captions`` calls; unlike
        ``include_summaries`` there is no client guard, since the absence of a
        model is handled downstream by leaving captions pending rather than
        by refusing to try.

        ``output_dir=None`` (or an empty/whitespace string) resolves to
        ``$DATASHEETINDEX_OUTPUT_DIR`` if set, otherwise a UID-namespaced
        subdirectory under the OS tempdir. The CLI passes its own
        ``"output"`` default explicitly.

        Returns a DatasheetArtifacts with paths, data, and quality info.
        """
        validate_max_figure_captions(max_figure_captions)
        if output_dir is None or not output_dir.strip():
            output_dir = resolve_default_output_dir()
        t_start = time.monotonic()
        doc = self.doc
        t_doc = time.monotonic()
        total_pages = len(doc)
        logger.info("PDF opened (%d pages) in %.1fs", total_pages, t_doc - t_start)

        pdf_name = self.artifact_stem(output_stem)

        # 1. Generate page-matched text and the figure index in one pass
        scan = scan_pages(doc)
        text_content = scan.text
        t_text = time.monotonic()
        logger.info("Text extraction done in %.1fs", t_text - t_doc)

        # 2. Generate the page-marked front matter and its per-page signals
        front_matter = build_front_matter(doc)
        preamble = front_matter.text

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

        # Reasons an eligible LLM step did not produce its result. Read by the
        # artifact cache, which refuses to reuse a degraded build -- otherwise a
        # single transient gateway error would cost this document its ToC for
        # the life of the output directory.
        enrichment_notes: list[str] = []

        # Two branches can need a client, and they share one. Captioning is the
        # second: without it here, a caller with credentials configured but no
        # explicit model would silently never caption, since the weak-ToC
        # branch is the only other construction site. A client of its own would
        # double the connection cost and leak on the path where only captions
        # need one, so both branches use ``active_llm_callable`` and the single
        # ``close_llm_client`` in the ``finally``. A caller-supplied callable
        # suppresses construction entirely -- including the probe ``bound.py``
        # hands in, which it owns and closes itself.
        effective_cap = max_figure_captions if caption_figures else 0
        has_caption_candidates = eligible_caption_count(scan.figures, effective_cap) > 0
        needs_toc_fallback = toc_quality.score < TOC_FALLBACK_THRESHOLD
        active_llm_callable = llm_callable
        owns_llm_callable = False
        # Where the client came from, which is a different question from
        # whether there is one. Only the summaries branch asks; see
        # _SUMMARY_CLIENT_ORIGINS.
        llm_client_origin: str | None = "caller" if llm_callable is not None else None
        if active_llm_callable is None and (
            needs_toc_fallback or has_caption_candidates
        ):
            active_llm_callable = self._try_create_default_llm_client()
            owns_llm_callable = active_llm_callable is not None
            if owns_llm_callable:
                llm_client_origin = (
                    "toc_fallback" if needs_toc_fallback else "figure_captions"
                )
            if active_llm_callable is None and needs_toc_fallback:
                logger.info(
                    "ToC quality below threshold but no LLM client is available; "
                    "these artifacts will not be cached for reuse"
                )
                enrichment_notes.append("toc_fallback_no_client")

        try:
            # 5. LLM fallback: regenerate ToC if quality is poor
            if active_llm_callable and needs_toc_fallback:
                t_llm = time.monotonic()
                logger.info(
                    "ToC quality below threshold (%.2f < %.2f), running LLM fallback",
                    toc_quality.score,
                    TOC_FALLBACK_THRESHOLD,
                )
                try:
                    from datasheetindex.llm.toc_fallback import generate_toc_from_text

                    candidate_nodes = generate_toc_from_text(
                        text_content, total_pages, active_llm_callable
                    )
                    enrich_with_table_counts(
                        candidate_nodes,
                        doc,
                        pdf_path=resolved_path,
                    )
                    enrich_with_continued_tables(candidate_nodes, text_content)
                    enrich_with_footnote_markers(candidate_nodes, text_content)
                    enrich_with_cross_references(candidate_nodes, text_content)
                    candidate_quality = assess_toc_quality(candidate_nodes, total_pages)
                    accept_candidate, candidate_reason = _accept_llm_toc_candidate(
                        toc_quality,
                        candidate_quality,
                        total_pages=total_pages,
                    )
                    if accept_candidate:
                        nodes = candidate_nodes
                        toc_quality = candidate_quality
                        logger.info(
                            "LLM ToC fallback accepted in %.1fs (%s)",
                            time.monotonic() - t_llm,
                            candidate_reason,
                        )
                    else:
                        logger.warning(
                            "LLM ToC fallback rejected; using original ToC (%s)",
                            candidate_reason,
                        )
                except Exception:
                    logger.warning(
                        "LLM ToC fallback failed; using original ToC",
                        exc_info=True,
                    )
                    enrichment_notes.append("toc_fallback_raised")

            # 6. LLM summaries: only when explicitly requested, and only on a
            # client whose PROVENANCE sanctions them. Do not simplify this back
            # to ``active_llm_callable and include_summaries``: that is the
            # regression _SUMMARY_CLIENT_ORIGINS exists to name.
            if (
                include_summaries
                and active_llm_callable is not None
                and llm_client_origin in _SUMMARY_CLIENT_ORIGINS
            ):
                t_sum = time.monotonic()
                from datasheetindex.llm.summarizer import add_summaries

                add_summaries(nodes, text_content, active_llm_callable)
                logger.info("LLM summaries done in %.1fs", time.monotonic() - t_sum)

            # 6b. Caption raster figure regions the text layer never named
            vision_client = get_vision_client(active_llm_callable)
            caption_outcome = caption_figures_in_place(
                doc,
                scan.figures,
                vision_client=vision_client,
                max_figure_captions=effective_cap,
            )
            if caption_outcome.failed:
                enrichment_notes.append("figure_caption_failed")

            logger.info("Total build time: %.1fs", time.monotonic() - t_start)

            # 7. Build JSON structure
            json_data = {
                "source": self._source_file_name(),
                "total_pages": total_pages,
                "preamble": preamble,
                "preamble_pages": [p.to_dict() for p in front_matter.pages],
                "toc_quality": {
                    "score": toc_quality.score,
                    "entry_count": toc_quality.entry_count,
                    "max_depth": toc_quality.max_depth,
                    "page_coverage": toc_quality.page_coverage,
                    "recommend_summaries": toc_quality.recommend_summaries,
                },
                "toc": [node.to_dict() for node in nodes],
                "figures": scan.figures,
                "figures_excluded": {
                    "below_min_area_pct": scan.excluded_below_min_area,
                    "min_area_pct": DEFAULT_MIN_AREA_PCT,
                },
                "figure_captions_excluded": {
                    "above_max": caption_outcome.excluded_above_max,
                    "max_figure_captions": effective_cap,
                },
            }

            # 8. Write output files
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)

            json_path = out / f"{pdf_name}.json"
            text_path = out / f"{pdf_name}.txt"

            # Atomic: temp then os.replace, so a crashed or failing build leaves
            # the previous generation intact rather than a truncated file. It
            # does not by itself guarantee a coherent pair -- the JSON write
            # can succeed and the text write then fail or race a reader. What
            # closes that gap is the sidecar's hash-after-read validation in
            # tools/bound.py, which rejects a pair whose bytes do not match
            # what it recorded.
            atomic_write_text(
                json_path, json.dumps(json_data, indent=2, ensure_ascii=False)
            )
            atomic_write_text(text_path, text_content)

            return DatasheetArtifacts(
                json_path=json_path,
                text_path=text_path,
                json_data=json_data,
                text_content=text_content,
                toc_quality=toc_quality,
                nodes=nodes,
                llm_enrichment_incomplete=bool(enrichment_notes),
                llm_enrichment_notes=tuple(enrichment_notes),
                figure_captions_pending=caption_outcome.pending,
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
