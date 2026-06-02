"""ToC extraction and enriched tree JSON generation."""

from __future__ import annotations

import concurrent.futures
import logging
import os
import re
from typing import TYPE_CHECKING

from datasheetindex.core.boilerplate import flag_boilerplate
from datasheetindex.core.textfile import extract_section_text
from datasheetindex.models import TocNode

if TYPE_CHECKING:
    import pymupdf

logger = logging.getLogger(__name__)


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
    """
    import pymupdf as _pymupdf

    pdf_path, page_idx = args
    doc = _pymupdf.open(pdf_path)
    try:
        tables = doc[page_idx].find_tables()  # type: ignore[attr-defined]
        return page_idx, len(tables.tables)
    finally:
        doc.close()


def _subprocess_init() -> None:
    """Redirect stdin/stdout to devnull in worker subprocesses.

    On Windows, child processes inherit the parent's file descriptors.
    When the parent is an MCP stdio server, inherited stdin/stdout
    collide with JSON-RPC communication, causing deadlocks.
    """
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)  # stdin
    os.dup2(devnull, 1)  # stdout
    os.close(devnull)


def _build_table_count_cache_parallel(
    pdf_path: str, total_pages: int
) -> dict[int, int]:
    """Scan all pages for tables using multiprocessing."""
    workers = min(os.cpu_count() or 1, total_pages)
    args = [(pdf_path, i) for i in range(total_pages)]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, initializer=_subprocess_init
    ) as pool:
        results = pool.map(_count_tables_on_page, args)
    return dict(results)


def _build_table_count_cache_sequential(
    doc: pymupdf.Document,
) -> dict[int, int]:
    """Scan all pages for tables sequentially (fallback)."""
    cache: dict[int, int] = {}
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

    When *pdf_path* is provided, pages are scanned in parallel across
    multiple processes for a significant speedup on large documents.
    Falls back to sequential scanning when the path is unavailable
    (e.g. in-memory test PDFs) or if multiprocessing fails.

    Modifies nodes in-place and returns them for convenience.
    """
    # Subprocess spawn overhead on Windows (~3.5s) outweighs parallelism
    # gains for small documents. Only parallelize above this threshold.
    _PARALLEL_PAGE_THRESHOLD = 12

    total_pages = len(doc)
    cache: dict[int, int] | None = None

    # Worker subprocesses redirect stdin/stdout to devnull via
    # _subprocess_init, so parallelism is safe even when the parent's
    # stdout is an MCP JSON-RPC pipe.
    _can_parallel = pdf_path is not None and total_pages >= _PARALLEL_PAGE_THRESHOLD

    if _can_parallel and pdf_path is not None:
        try:
            cache = _build_table_count_cache_parallel(pdf_path, total_pages)
        except Exception:
            logger.debug(
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
