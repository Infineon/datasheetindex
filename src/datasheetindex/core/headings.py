"""Where each ToC heading sits inside its start page, in page-text offsets.

Page ranges tile the document: a section ends on the page before its next
sibling starts. When that sibling's heading sits partway down its page, the
text above it belongs to the earlier section, and a read of the earlier range
used to drop it silently (BME280 "1.3 Pressure sensor specification" lost half
its spec table). The fix is applied at read time, so the artifact -- ranges,
table counts, the text file -- is unchanged; this module supplies the one fact
the read needs: where on its page each heading starts.

Offsets index the page text exactly as ``textfile.page_texts`` returns it,
which is the string ``TextSearchMatch.start`` indexes too. Never compute them
on a stripped or normalized copy: a marker inserted at an offset from a
different string lands inside the previous line.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from datasheetindex.core.textfile import page_texts
from datasheetindex.models import flatten_nodes

if TYPE_CHECKING:
    from datasheetindex.models import TocNode

# Printed numbering a bookmark may leave out: "5.2", "A.1", "IV.", "7)".
_NUMBERING = r"(?:[A-Z]?\d+(?:\.\d+)*[.)]?|[A-Z][.)]|[IVXLC]+[.)])"


@lru_cache(maxsize=1024)
def _heading_pattern(title: str) -> re.Pattern[str] | None:
    tokens = title.split()
    if not tokens:
        return None
    body = r"\s+".join(re.escape(token) for token in tokens)
    # Group 1 starts at the beginning of the line, so a marker inserted there
    # precedes any indentation instead of splitting it. A heading must begin a
    # line: a title mentioned mid-sentence ("see 3 Thermal limits") is not
    # where the section starts, and stopping there would cut the tail short --
    # the one error this module exists to prevent.
    return re.compile(
        rf"(?:^|\n)([ \t]*(?:{_NUMBERING}[ \t]+)?{body})(?=\s|$)",
        re.IGNORECASE,
    )


def locate_heading(page_text: str, title: str) -> int | None:
    """Offset of the line where ``title`` starts as a heading, or ``None``.

    Tolerates case, whitespace and line wrapping inside the title, and printed
    numbering the bookmark omits. The first line-starting occurrence wins.
    """
    pattern = _heading_pattern(title)
    if pattern is None:
        return None
    match = pattern.search(page_text)
    return match.start(1) if match else None


@dataclass(frozen=True)
class PageHeadings:
    """The headings of the sections that start on one page."""

    #: ``(offset, node)`` for each heading found below other content, by offset.
    mid: tuple[tuple[int, TocNode], ...] = ()
    #: Some section's heading is the first content of the page.
    opens: bool = False
    #: Some section starts here but its heading was not found on the page.
    unlocated: bool = False


def map_headings(nodes: list[TocNode], text_content: str) -> dict[int, PageHeadings]:
    """Locate every node's heading on its start page, keyed by page."""
    texts = page_texts(text_content)
    by_page: dict[int, list[TocNode]] = defaultdict(list)
    for node in flatten_nodes(nodes):
        by_page[node.start_page].append(node)

    result: dict[int, PageHeadings] = {}
    for page, page_nodes in by_page.items():
        text = texts.get(page, "")
        mid: list[tuple[int, TocNode]] = []
        opens = unlocated = False
        for node in page_nodes:
            offset = locate_heading(text, node.title)
            if offset is None:
                unlocated = True
            elif text[:offset].strip():
                mid.append((offset, node))
            else:
                opens = True
        # sorted() is stable, so equal offsets keep document order.
        mid.sort(key=lambda item: item[0])
        result[page] = PageHeadings(tuple(mid), opens, unlocated)
    return result


def next_page_head(
    text_content: str, page: int, headings: dict[int, PageHeadings]
) -> str | None:
    """The framed text that continues a read ending on ``page - 1``, if any.

    Nothing when no section starts on ``page`` (its content is the previous
    section's, and the ranges already cover it) or when a heading opens it.
    Otherwise the text above the earliest mid-page heading. When a section
    starts here but no heading was found, the whole page: a page of overlap
    is the safe error, a lost tail the unsafe one.
    """
    entry = headings.get(page)
    if entry is None or entry.opens:
        return None
    text = page_texts(text_content).get(page, "")
    if entry.mid:
        offset, node = entry.mid[0]
        framing = (
            f"=== Continued from the top of page {page}, up to the next section "
            f'heading "{node.title.strip()}" ==='
        )
        return f"{framing}\n{text[:offset].rstrip()}"
    if entry.unlocated:
        framing = (
            f"=== Continued: all of page {page}. A section starts on it, but its "
            "heading could not be located ==="
        )
        return f"{framing}\n{text}"
    return None


def mark_section_start(
    section: str, page: int, headings: dict[int, PageHeadings]
) -> str:
    """Mark where the read's own text begins on its first page, ``page``.

    Only the first page needs it: there, rows above the first mid-page heading
    are the previous section's tail. Later headings in a read are its own
    structure, already visible as headings, and marking them all cost a median
    5-10% of read characters on the corpus. No marker when a heading opens the
    page, since the text above a later heading is then the opening section's.

    ``section`` is the raw slice ``extract_section_text`` returns. The offset is
    rebased past the ``--- PAGE N ---`` marker and the line breaks ``page_texts``
    strips after it, so it lands where search offsets point.
    """
    entry = headings.get(page)
    if entry is None or entry.opens or not entry.mid:
        return section
    page_marker = f"--- PAGE {page} ---"
    found = section.find(page_marker)
    if found == -1:
        return section
    body = found + len(page_marker)
    rest = section[body:]
    offset, node = entry.mid[0]
    position = body + len(rest) - len(rest.lstrip("\r\n")) + offset
    # The title, not the breadcrumb: one line per read, and on short section
    # reads the breadcrumb doubled the marker's cost for no extra signal.
    line = (
        f"=== SECTION STARTS: {node.title.strip()} "
        "(text above is the previous section's) ===\n"
    )
    return section[:position] + line + section[position:]


def section_at(
    nodes: list[TocNode],
    headings: dict[int, PageHeadings],
    page: int,
    offset: int,
) -> TocNode | None:
    """The section owning ``offset`` on ``page``: the last heading before it.

    Walks headings in document order, so the result is the most recently
    started section -- which is how an outline divides text. A heading that
    opens its page or was not found counts as starting at offset 0.
    """
    mid_offsets = {id(node): at for at, node in headings.get(page, PageHeadings()).mid}
    owner: TocNode | None = None
    for node in flatten_nodes(nodes):
        if node.start_page < page or (
            node.start_page == page and mid_offsets.get(id(node), 0) <= offset
        ):
            owner = node
    return owner
