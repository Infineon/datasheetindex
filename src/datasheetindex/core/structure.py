"""ToC extraction and enriched tree JSON generation."""

from __future__ import annotations

import concurrent.futures
import contextlib
import json
import logging
import multiprocessing
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from datasheetindex.core.boilerplate import flag_boilerplate
from datasheetindex.core.engine import classic_tables
from datasheetindex.core.textfile import extract_page_text, extract_section_text
from datasheetindex.models import TocNode

if TYPE_CHECKING:
    from multiprocessing.context import BaseContext

    import pymupdf

logger = logging.getLogger(__name__)

# Page-level find_tables() does not scale past a handful of processes, and each
# worker costs real memory. Cap the fan-out so a large PDF on a many-core host
# cannot spawn one process per page.
_MAX_PARALLEL_WORKERS = 8

# The delegating parent must outlast the child's own scan deadline, or it
# always wins the race (its clock starts first, before the child has even
# imported pymupdf) and kills a child that was about to report a usable
# traceback. The margin covers process start plus that import.
_HELPER_GRACE_SECONDS = 30.0

# A killed process is reaped immediately; this only bounds the window in
# which the kill has not landed yet, so the escape path is never a bare wait.
_KILL_WAIT_SECONDS = 10.0

# Absolute ceiling on a scan deadline. Past roughly this point the caller's
# own request timeout has already fired, so a longer deadline buys nothing
# and only delays the sequential fallback that would have produced an answer.
_SCAN_TIMEOUT_CEILING_SECONDS = 600.0

_CGROUP_ROOT = Path("/sys/fs/cgroup")
_PROC_SELF_CGROUP = Path("/proc/self/cgroup")


def extract_toc(doc: pymupdf.Document) -> list[list]:
    """Extract the raw ToC from a PDF document.

    Returns a list of ``[level, title, page_number]`` entries from PyMuPDF's
    ``get_toc()``.
    """
    return doc.get_toc()


def build_tree(raw_toc: list[list], total_pages: int) -> list[TocNode]:
    """Build a hierarchical tree of TocNode from raw ToC entries.

    Uses a stack to track the current nesting path. Each raw entry is
    ``[level, title, start_page]`` where level >= 1.
    """
    if not raw_toc:
        return []

    root_nodes: list[TocNode] = []
    # Stack holds (level, node) pairs for current ancestry
    stack: list[tuple[int, TocNode]] = []

    for entry in raw_toc:
        level, title, start_page = validate_toc_entry(entry)
        node = TocNode(title=title, level=level, start_page=start_page)

        # Pop stack until we find the parent level
        while stack and stack[-1][0] >= level:
            stack.pop()

        if stack:
            # Attach as child of the top of stack
            stack[-1][1].nodes.append(node)
        else:
            # Top-level node
            root_nodes.append(node)

        stack.append((level, node))

    compute_end_pages(root_nodes, total_pages)
    assign_node_ids(root_nodes)
    assign_breadcrumbs(root_nodes)
    flag_boilerplate(root_nodes)
    return root_nodes


def validate_toc_entry(entry: list) -> tuple[int, str, int]:
    """Validate and normalize a raw ToC entry."""
    if len(entry) < 3:
        raise ValueError("Each ToC entry must include [level, title, start_page]")

    level = int(entry[0])
    title = str(entry[1])
    start_page = int(entry[2])

    if level < 1:
        raise ValueError(f"Invalid ToC level {level}; expected >= 1")
    if start_page < 1:
        raise ValueError(f"Invalid start_page {start_page}; expected >= 1")

    return level, title, start_page


def compute_end_pages(nodes: list[TocNode], parent_end: int) -> None:
    """Recursively compute end_page for each node.

    A node's end_page is determined by:
    - If it has a next sibling: next sibling's start_page - 1
    - Otherwise: the parent's end_page
    """
    for i, node in enumerate(nodes):
        if i + 1 < len(nodes):
            # Clamp: when siblings share a start page, end >= start
            node.end_page = max(nodes[i + 1].start_page - 1, node.start_page)
        else:
            # Clamp malformed ToCs where child starts after parent's inferred end.
            node.end_page = max(parent_end, node.start_page)

        if node.nodes:
            compute_end_pages(node.nodes, node.end_page)


def assign_node_ids(nodes: list[TocNode], counter: list[int] | None = None) -> None:
    """Assign depth-first sequential 4-digit zero-padded IDs."""
    if counter is None:
        counter = [1]

    for node in nodes:
        node.node_id = f"{counter[0]:04d}"
        counter[0] += 1
        if node.nodes:
            assign_node_ids(node.nodes, counter)


BREADCRUMB_SEPARATOR = " > "


def assign_breadcrumbs(nodes: list[TocNode], parent_path: str = "") -> None:
    """Recursively assign the full ancestry path to each node's ``breadcrumb``.

    The breadcrumb is the chain of titles from the root to the node, joined
    by ``" > "`` and including the node's own title.
    """
    for node in nodes:
        title = node.title.strip()
        node.breadcrumb = (
            f"{parent_path}{BREADCRUMB_SEPARATOR}{title}" if parent_path else title
        )
        if node.nodes:
            assign_breadcrumbs(node.nodes, node.breadcrumb)


def find_breadcrumb_for_page(nodes: list[TocNode], page: int) -> str | None:
    """Return the breadcrumb of the deepest ToC section containing ``page``.

    Walks the enriched ``TocNode`` tree and returns the ``breadcrumb`` of the
    most specific node whose ``[start_page, end_page]`` range covers ``page``.
    Returns ``None`` when no section covers the page or no covering node carries
    a breadcrumb.

    When sibling sections have overlapping ranges that cover the same page at
    the same depth (e.g. siblings sharing a start page), the first one in
    document order wins -- a nested child is strictly deeper than its parent, so
    "most specific" still selects the deepest covering section.
    """
    best: TocNode | None = None

    def _walk(items: list[TocNode]) -> None:
        nonlocal best
        for node in items:
            if node.start_page <= page <= node.end_page and (
                best is None or node.level > best.level
            ):
                best = node
            if node.nodes:
                _walk(node.nodes)

    _walk(nodes)
    if best is None or not best.breadcrumb:
        return None
    return best.breadcrumb


def _count_tables_on_page(args: tuple[str, int]) -> tuple[int, int]:
    """Count tables on a single page. Runs in a subprocess.

    Each worker opens the PDF independently because PyMuPDF document
    objects cannot be pickled across process boundaries.

    The classic_tables() guard is a no-op today -- workers start under
    forkserver/spawn and never import pymupdf4llm. It is here so that "classic"
    is a property of this function rather than an accident of the worker's
    import graph: a future set_forkserver_preload, or a PyMuPDF that activates
    layout on import, must not silently reintroduce the divergence.
    """
    import pymupdf as _pymupdf

    pdf_path, page_idx = args
    doc = _pymupdf.open(pdf_path)
    try:
        with classic_tables():
            tables = doc[page_idx].find_tables()  # type: ignore[attr-defined]
        return page_idx, len(tables.tables)
    finally:
        doc.close()


def _subprocess_init() -> None:
    """Redirect stdin/stdout to devnull in worker subprocesses.

    Workers start under a non-fork method (see :func:`_mp_context`), so they do
    not inherit the parent's open file descriptors -- except 0, 1 and 2, which
    survive ``exec`` on every platform. When the parent is an MCP stdio server,
    an inherited stdin/stdout collides with JSON-RPC communication, causing
    deadlocks. Only 0 and 1 are redirected; stderr is deliberately left intact
    so worker tracebacks still reach the parent's logs.
    """
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)  # stdin
    os.dup2(devnull, 1)  # stdout
    os.close(devnull)


def _cpus_from_quota(quota: int, period: int) -> int | None:
    """Effective CPU count for a bandwidth quota, or ``None`` if unlimited.

    A non-positive quota means unlimited; a non-positive period is malformed
    (and would divide by zero).
    """
    if quota <= 0 or period <= 0:
        return None
    return max(1, quota // period)


def _own_cgroup_relpath(proc_cgroup: Path, controller: str | None) -> str:
    """This process's cgroup path, relative to the hierarchy root.

    Parses ``/proc/self/cgroup``. *controller* is ``None`` for the cgroup v2
    unified hierarchy (whose lines start with ``0::``), or a v1 controller name
    such as ``cpu``. Returns ``""`` (the root) when no match is found -- which
    is also what a container in its own cgroup namespace reports.
    """
    try:
        lines = proc_cgroup.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""

    for line in lines:
        hierarchy_id, _, rest = line.partition(":")
        controllers, _, path = rest.partition(":")
        if controller is None:
            if hierarchy_id == "0":
                return path.strip("/")
        elif controller in controllers.split(","):
            return path.strip("/")
    return ""


def _cgroup_chain(base: Path, relpath: str) -> list[Path]:
    """*base* and each descendant along *relpath*, root first."""
    chain = [base]
    current = base
    for part in relpath.split("/"):
        if part and part != "..":
            current = current / part
            chain.append(current)
    return chain


def _read_cgroup_cpu_quota(
    root: Path | None = None, proc_cgroup: Path | None = None
) -> int | None:
    """Effective CPU count from the cgroup CPU bandwidth quota.

    Resolves the process's *own* cgroup via ``/proc/self/cgroup`` rather than
    reading the hierarchy root, because a quota is commonly applied to a nested
    cgroup (e.g. systemd's ``CPUQuota=`` on ``/system.slice/x.service``) while
    the root always reports unlimited. A quota on any ancestor also constrains
    us, so the tightest one along the chain wins. In a container's own cgroup
    namespace the chain collapses to the root, which is where its quota lives.

    Returns ``None`` when no quota applies or none can be read.
    """
    root = _CGROUP_ROOT if root is None else root
    proc_cgroup = _PROC_SELF_CGROUP if proc_cgroup is None else proc_cgroup

    quotas: list[int] = []

    # cgroup v2: "<quota> <period>", or "max <period>" when unlimited.
    for directory in _cgroup_chain(root, _own_cgroup_relpath(proc_cgroup, None)):
        try:
            quota_s, period_s = (
                (directory / "cpu.max").read_text(encoding="utf-8").split()
            )
            if quota_s == "max":
                continue
            cpus = _cpus_from_quota(int(quota_s), int(period_s))
        except (OSError, ValueError):
            continue
        if cpus is not None:
            quotas.append(cpus)

    if quotas:
        return min(quotas)

    # cgroup v1: quota and period live in separate files under the cpu
    # controller, which is co-mounted as "cpu,cpuacct". The plain "cpu" name is
    # only a compat symlink, and some runtimes bind-mount the real name alone.
    v1_relpath = _own_cgroup_relpath(proc_cgroup, "cpu")
    for v1_root in (root / "cpu", root / "cpu,cpuacct"):
        for directory in _cgroup_chain(v1_root, v1_relpath):
            try:
                quota = int(
                    (directory / "cpu.cfs_quota_us").read_text(encoding="utf-8")
                )
                period = int(
                    (directory / "cpu.cfs_period_us").read_text(encoding="utf-8")
                )
                cpus = _cpus_from_quota(quota, period)
            except (OSError, ValueError):
                continue
            if cpus is not None:
                quotas.append(cpus)

    return min(quotas) if quotas else None


def _available_cpus() -> int:
    """CPUs this process may actually use.

    ``os.cpu_count()`` reports the host's cores and ignores the cgroup CPU
    quota, so in a 1-CPU container on a 128-core node it overstates the budget
    by 128x. Affinity does not help either: a CFS bandwidth quota is not an
    affinity mask. Take the tighter of the two signals.
    """
    cpus = os.process_cpu_count() or 1  # affinity-aware, quota-blind
    quota = _read_cgroup_cpu_quota()
    if quota is not None:
        cpus = min(cpus, quota)
    return max(1, cpus)


def _mp_context() -> BaseContext:
    """A start method that does not fork the calling process.

    ``fork()`` gives the child only the calling thread but a full copy of every
    mutex, including any held by threads that do not exist in the child -- such
    a child can deadlock on its first allocation. It also copies every open file
    descriptor, and an inherited ``flock`` fd keeps the lock held no matter which
    process acquired it. A library cannot know whether its caller is threaded,
    so it must not fork one. This also keeps behaviour identical across Python
    3.13 (which defaults to ``fork``) and 3.14 (which defaults to
    ``forkserver``).

    ``spawn`` is available on every platform CPython supports, so there is no
    path back to ``fork`` here.
    """
    methods = multiprocessing.get_all_start_methods()
    return multiprocessing.get_context(
        "forkserver" if "forkserver" in methods else "spawn"
    )


def _parallel_enabled_by_env() -> bool:
    """Whether DATASHEETINDEX_PARALLEL permits the parallel scan.

    Accepts the spellings a user actually reaches for. Matching only the
    literal "0" would silently ignore DATASHEETINDEX_PARALLEL=false, leaving
    the escape hatch looking broken to the person who most needs it.
    """
    value = os.environ.get("DATASHEETINDEX_PARALLEL", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _is_windows() -> bool:
    """Whether this is a Windows host.

    A function, not an inline ``sys.platform`` test, so a test can override the
    platform for one module instead of lying to every library in the process
    that reads ``sys.platform``.
    """
    return sys.platform == "win32"


def _abandon_pool(pool: Any) -> None:
    """Drop a pool without waiting on it.

    Typed loosely on purpose: this needs only ``shutdown`` and, where it
    exists, ``kill_workers``. Pinning it to ``Executor`` would force the tests
    that cover both branches to subclass an ABC to say something simple.

    ``shutdown(wait=False)`` returns promptly, but the executor's manager
    thread keeps joining its workers, and ``concurrent.futures`` joins *that*
    thread from an ``atexit`` hook -- so a wedged worker still hangs the
    process on the way out, merely later, which for a long-lived MCP server is
    indistinguishable from the bug being fixed. ``kill_workers()`` (3.14+)
    terminates them first; on 3.13, our floor, no such API exists and
    ``shutdown`` is the best available.
    """
    kill_workers = getattr(pool, "kill_workers", None)
    if kill_workers is not None:
        kill_workers()
        return
    pool.shutdown(wait=False, cancel_futures=True)


def _read_worker_stderr(err_path: str) -> str:
    """Tail of the scan worker's stderr, for a failure message."""
    try:
        with open(err_path, "rb") as handle:
            raw = handle.read()
    except OSError:
        return "(worker stderr unavailable)"
    return raw.decode("utf-8", "replace").strip()[-2000:]


def _scan_timeout(total_pages: int) -> float:
    """Deadline for a whole-document scan, in seconds.

    Bounds every parallel path so a wedged pool degrades to a slow sequential
    scan instead of hanging forever. One second per page is roughly 5x the
    sequential cost of a slow page and ~40x the parallel cost, so this only
    fires on a genuine stall, never on a document that is merely large.

    Capped, because the fallback has to be reachable *within the call*: a
    1000-page document would otherwise wait ~1000s, then run a sequential scan
    on top, long past any MCP client's request timeout -- a deadline nobody
    lives to see is the hang it replaced, wearing a hat.
    """
    return min(max(120.0, float(total_pages)), _SCAN_TIMEOUT_CEILING_SECONDS)


def _build_table_count_cache_pool(pdf_path: str, total_pages: int) -> dict[int, int]:
    """Scan all pages for tables using a process pool in *this* process.

    Not used on Windows -- see :func:`_build_table_count_cache_helper`.
    """
    workers = min(_available_cpus(), total_pages, _MAX_PARALLEL_WORKERS)
    args = [(pdf_path, i) for i in range(total_pages)]
    pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=_subprocess_init,
        mp_context=_mp_context(),
    )
    try:
        # Materialised inside the try, and deliberately not under `with`.
        # pool.map returns a LAZY iterator: consuming it after the block let
        # __exit__ run shutdown(wait=True) before a single result had been
        # read, so a stalled worker blocked in shutdown rather than raising.
        cache = dict(
            pool.map(_count_tables_on_page, args, timeout=_scan_timeout(total_pages))
        )
    except BaseException:
        # Never wait on the way out. A wedged pool is precisely the case the
        # timeout exists to escape, and shutdown(wait=True) would block on it
        # forever -- turning a recoverable failure back into a hang.
        _abandon_pool(pool)
        raise
    # wait=True is correct here: every result has already been read, so the
    # workers are idle. A worker wedging in its own atexit after returning
    # results would still block; that is accepted, not overlooked.
    pool.shutdown(wait=True)
    return cache


def _build_table_count_cache_helper(pdf_path: str, total_pages: int) -> dict[int, int]:
    """Scan all pages by delegating to a stdio-detached child process.

    The Windows path. A pool created inside an MCP stdio server deadlocks
    there -- its workers freeze before their interpreter initialises and only
    unblock when the server exits (modelcontextprotocol/python-sdk#817). A
    plain process pools normally, so we make one: this child is spawned with
    stdin and stdout on devnull, which severs it from the server's JSON-RPC
    handles, and it builds the pool itself.

    Measured on a 148-page datasheet under a real Windows MCP server: 31.4s,
    against an unbounded hang for the in-process pool.
    """
    if not sys.executable:
        raise RuntimeError("no interpreter to launch the scan worker with")
    if getattr(sys, "frozen", False):
        # "-m" against a frozen host re-runs the application itself with our
        # module name as argv, which for a frozen MCP server means starting a
        # second server. Fail fast into the sequential fallback instead.
        raise RuntimeError("cannot launch the scan worker from a frozen interpreter")

    # ignore_cleanup_errors: on the kill path a worker may still hold a handle
    # here, and a PermissionError raised during unwinding would mask the
    # TimeoutExpired the caller actually needs to see.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        out_path = os.path.join(tmpdir, "table_counts.json")
        err_path = os.path.join(tmpdir, "scan_worker.err")

        # stderr goes to a FILE, never a PIPE. The child's own pool workers
        # inherit fd 2 (_subprocess_init redirects only 0 and 1, so worker
        # tracebacks survive), which means they hold the write end. A PIPE
        # would make communicate() wait for an EOF that kill() cannot deliver,
        # because kill() reaches only the direct child -- reproducing the exact
        # hang this module exists to remove.
        with open(err_path, "wb") as err_handle:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "datasheetindex.core._scan_worker",
                    pdf_path,
                    str(total_pages),
                    out_path,
                ],
                stdin=subprocess.DEVNULL,
                # Severing stdin/stdout is the whole point; do not "improve"
                # either into a PIPE. Results come back through out_path.
                stdout=subprocess.DEVNULL,
                stderr=err_handle,
                # CREATE_NO_WINDOW: the server usually has no console, and
                # without this each scan flashes one. Absent off Windows.
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            try:
                # wait(), not communicate(): it reaps only the direct child and
                # is indifferent to grandchildren holding inherited handles.
                # The grace margin lets the child hit its OWN deadline first and
                # report a real traceback; without it the parent's clock, which
                # starts earlier, always wins and every stall looks identical.
                process.wait(timeout=_scan_timeout(total_pages) + _HELPER_GRACE_SECONDS)
            except BaseException as exc:
                # BaseException, not only TimeoutExpired: a KeyboardInterrupt
                # out of wait() would otherwise leave the child alive with its
                # whole pool, holding a handle into the temp directory we are
                # about to remove -- the orphan this module exists to prevent,
                # merely relocated.
                #
                # kill() signals only this child. Its pool workers are not ours
                # to signal and may briefly outlive it; they are released by the
                # parent's death, which is what strands them in the first place.
                process.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=_KILL_WAIT_SECONDS)
                if isinstance(exc, subprocess.TimeoutExpired):
                    # Reading the worker's stderr back here is the whole payoff
                    # of the grace margin: a bare TimeoutExpired says something
                    # stalled but never what, and the answer is already sitting
                    # in err_path.
                    raise RuntimeError(
                        f"scan worker timed out after {exc.timeout}s: "
                        f"{_read_worker_stderr(err_path)}"
                    ) from exc
                raise

        if process.returncode != 0:
            raise RuntimeError(
                f"scan worker exited {process.returncode}: "
                f"{_read_worker_stderr(err_path)}"
            )

        with open(out_path, encoding="utf-8") as handle:
            counts = json.load(handle)

    # int() on the value as well as the key: a non-int count would survive
    # into _apply_table_counts and fail there, outside the try that would
    # have fallen back.
    cache = {int(page): int(count) for page, count in counts.items()}
    if len(cache) != total_pages:
        # Downstream, a missing page is indistinguishable from a page with no
        # tables: _apply_table_counts defaults it to 0. Raise so the sequential
        # fallback runs, rather than shipping a wrong answer that looks valid.
        raise RuntimeError(
            f"scan worker returned {len(cache)} of {total_pages} page counts"
        )
    return cache


def _build_table_count_cache_parallel(
    pdf_path: str, total_pages: int
) -> dict[int, int]:
    """Scan all pages for tables in parallel, however that is safe here.

    Windows cannot pool from inside an MCP stdio server, so it delegates to a
    detached child. Everywhere else the in-process pool is proven and one
    process cheaper, so it stays.
    """
    if _is_windows():
        return _build_table_count_cache_helper(pdf_path, total_pages)
    return _build_table_count_cache_pool(pdf_path, total_pages)


def _build_table_count_cache_sequential(
    doc: pymupdf.Document,
) -> dict[int, int]:
    """Scan all pages for tables sequentially (fallback).

    Pinned to the classic detector: this runs in the caller's process, which may
    have imported pymupdf4llm and installed the layout hook.
    """
    cache: dict[int, int] = {}
    with classic_tables():
        for page_idx in range(len(doc)):
            tables = doc[page_idx].find_tables()  # type: ignore[attr-defined]
            cache[page_idx] = len(tables.tables)
    return cache


def enrich_with_table_counts(
    nodes: list[TocNode],
    doc: pymupdf.Document,
    pdf_path: str | None = None,
) -> list[TocNode]:
    """Count tables on each node's page range using PyMuPDF find_tables().

    Counts always come from PyMuPDF's classic geometric detector, whether or not
    this process has imported pymupdf4llm, and whichever path below runs. They
    are therefore a stable property of the document. Expect false positives on
    plots and block diagrams; this is a navigational hint, not a precise count.

    When *pdf_path* is provided, pages are scanned in parallel across
    multiple processes for a significant speedup on large documents.
    Falls back to sequential scanning when the path is unavailable
    (e.g. in-memory test PDFs) or if multiprocessing fails.

    Modifies nodes in-place and returns them for convenience.
    """
    # Workers never fork the caller (see _mp_context), so every platform pays a
    # fixed process-startup cost that outweighs parallelism on small documents.
    # Only parallelize above this threshold. Measured on ~200ms/page documents:
    # ~1.4x at 12 pages, ~3.7x at 40.
    _PARALLEL_PAGE_THRESHOLD = 12

    total_pages = len(doc)
    cache: dict[int, int] | None = None

    # Both parallel paths keep worker stdin/stdout off the parent's, so
    # parallelism is safe even when that stdout is an MCP JSON-RPC pipe.
    # DATASHEETINDEX_PARALLEL=0 is the escape hatch if a host still trips over
    # process creation; the scan then runs sequentially, only slower.
    _can_parallel = (
        pdf_path is not None
        and total_pages >= _PARALLEL_PAGE_THRESHOLD
        and _parallel_enabled_by_env()
    )

    if _can_parallel and pdf_path is not None:
        try:
            cache = _build_table_count_cache_parallel(pdf_path, total_pages)
        except Exception:
            # Warning, not debug: a pool that fails to start degrades silently
            # to a ~3x slower scan, and that invisibility is how the fan-out
            # bug this fallback guards went unnoticed in production.
            logger.warning(
                "Parallel table counting failed, falling back to sequential",
                exc_info=True,
            )

    if cache is None:
        cache = _build_table_count_cache_sequential(doc)

    _apply_table_counts(nodes, cache, total_pages)
    return nodes


def _apply_table_counts(
    nodes: list[TocNode], cache: dict[int, int], total_pages: int
) -> None:
    """Distribute pre-computed per-page table counts to ToC nodes."""
    for node in nodes:
        count = 0
        start_idx = node.start_page - 1
        end_idx = node.end_page - 1
        for page_idx in range(start_idx, end_idx + 1):
            if 0 <= page_idx < total_pages:
                count += cache.get(page_idx, 0)

        node.table_count = count
        node.has_tables = count > 0

        if node.nodes:
            _apply_table_counts(node.nodes, cache, total_pages)


_CONTINUED_TABLE_RE = re.compile(
    r"(Table\s+[\d\-\.]+\s+.+?)\s*\((?:[Cc]ontinued|[Cc]ont\.)\)"
)


def enrich_with_continued_tables(
    nodes: list[TocNode], text_content: str
) -> list[TocNode]:
    """Detect multi-page tables by scanning for "(Continued)" markers.

    Modifies nodes in-place and returns them for convenience.
    """
    _continued_tables_recursive(nodes, text_content)
    return nodes


def _continued_tables_recursive(nodes: list[TocNode], text_content: str) -> None:
    """Walk the tree and populate continued_tables for each node."""
    for node in nodes:
        section_text = extract_section_text(
            text_content, node.start_page, node.end_page
        )
        matches = _CONTINUED_TABLE_RE.findall(section_text)
        # Deduplicate while preserving order
        seen: set[str] = set()
        deduplicated: list[str] = []
        for title in matches:
            normalized = title.strip()
            if normalized not in seen:
                seen.add(normalized)
                deduplicated.append(normalized)
        node.continued_tables = deduplicated

        if node.nodes:
            _continued_tables_recursive(node.nodes, text_content)


# The page-boundary continuation signal. Deliberately separate from
# _CONTINUED_TABLE_RE above: that one defines TocNode.continued_tables (tables
# captioned "Table N ... (Continued)"); this one answers the different question
# "does content continue across this page break", for any publisher's wording.
#
# The upper bound on the title's length distinguishes "a title" from "a
# paragraph of prose that happens to end in (continued)" -- it is not a claim
# that titles are short, and it does no discriminating work against the known
# false positives (those are rejected by the positional guard below, and are
# short lines anyway). 200 is chosen to comfortably admit a vendor's full
# parameterised caption repeated on the continuation page, e.g. the 92-char
# "Table 12. Electrical characteristics (VDD = 3.3 V, TA = 25 degC, unless
# otherwise specified)", which a tighter bound would silently drop.
_CONTINUATION_RE = re.compile(
    r"[ \t]*(\S.{2,200}?)[ \t]*\((?:continued|cont\.)\)[ \t]*$",
    re.IGNORECASE,
)

# A table that resumes does so at the top of the page, below the running header.
# Measured across the Infineon and TI datasheets: genuine continuations sit at
# nonblank line 3, while the mid-page "NOTES: (continued)" blocks on TI's
# mechanical-drawing pages sit at lines 19-48. This positional guard is the
# whole correctness property -- see the design spec.
_OPENING_BLOCK_LINES = 5


def continuation_at_boundary(text_content: str, page: int) -> list[str]:
    """Titles of content that continues from ``page`` onto ``page + 1``.

    A title is returned when page+1 opens with a continuation marker inside its
    opening block. Returns empty for an out-of-range ``page`` -- ``page < 1``
    (no such boundary exists) or a ``page`` at/after the last one (nothing
    follows) -- and when page+1 carries no qualifying marker.

    Silence is not a completeness claim: content can spill across a page break
    with no marker at all.
    """
    if page < 1:
        return []

    next_text = extract_page_text(text_content, page + 1)
    if not next_text:
        return []

    titles: list[str] = []
    seen: set[str] = set()
    nonblank = [line for line in next_text.splitlines() if line.strip()]
    for line in nonblank[:_OPENING_BLOCK_LINES]:
        match = _CONTINUATION_RE.match(line)
        if match is None:
            continue
        title = " ".join(match.group(1).split())
        if title not in seen:
            seen.add(title)
            titles.append(title)
    return titles
