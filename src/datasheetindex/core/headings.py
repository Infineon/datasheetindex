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

from datasheetindex.core.textfile import extract_page_text, page_texts
from datasheetindex.models import flatten_nodes

if TYPE_CHECKING:
    from datasheetindex.models import TocNode

# Printed numbering a bookmark may leave out: "5.2", "A.1", "IV.", "7)".
_NUMBERING = r"(?:[A-Z]?\d+(?:\.\d+)*[.)]?|[A-Z][.)]|[IVXLC]+[.)])"


@lru_cache(maxsize=1024)
def _heading_patterns(title: str) -> tuple[re.Pattern[str], ...]:
    tokens = title.split()
    if not tokens:
        return ()
    body = r"\s+".join(re.escape(token) for token in tokens)
    # Group 1 starts at the beginning of the line, so a marker inserted there
    # precedes any indentation instead of splitting it. A heading must begin a
    # line: a title mentioned mid-sentence ("see 3 Thermal limits") is not
    # where the section starts, and stopping there would cut the tail short --
    # the one error this module exists to prevent.
    head = rf"(?:^|\n)([ \t]*(?:{_NUMBERING}[ \t]+)?{body})"
    return (
        # The title alone on its line. Tried first: a short title such as
        # "Pressure" also starts table rows ("Pressure 300 1100 hPa") above the
        # real heading.
        re.compile(rf"{head}(?=[ \t]*(?:\n|$))", re.IGNORECASE),
        # The title starting a line that continues, as when PyMuPDF merges a
        # heading with the next line's text.
        re.compile(rf"{head}(?=\s|$)", re.IGNORECASE),
    )


def locate_heading(page_text: str, title: str) -> int | None:
    """Offset of the line where ``title`` starts as a heading, or ``None``.

    Tolerates case, whitespace and line wrapping inside the title, and printed
    numbering the bookmark omits. A line holding only the title wins; failing
    that, the first line that starts with it.
    """
    for pattern in _heading_patterns(title):
        match = pattern.search(page_text)
        if match:
            return match.start(1)
    return None


@dataclass(frozen=True)
class PageHeadings:
    """The headings of the sections that start on one page."""

    #: ``(offset, node)`` for each heading found below other content, by offset.
    mid: tuple[tuple[int, TocNode], ...] = ()
    #: Sections whose heading is the first content of the page.
    top: tuple[TocNode, ...] = ()
    #: Sections that start here but whose heading was not found on the page.
    missing: tuple[TocNode, ...] = ()

    @property
    def opens(self) -> bool:
        return bool(self.top)

    @property
    def unlocated(self) -> bool:
        return bool(self.missing)


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
        top: list[TocNode] = []
        missing: list[TocNode] = []
        for node in page_nodes:
            offset = locate_heading(text, node.title)
            if offset is None:
                missing.append(node)
            elif text[:offset].strip():
                mid.append((offset, node))
            else:
                top.append(node)
        # sort() is stable, so equal offsets keep document order.
        mid.sort(key=lambda item: item[0])
        result[page] = PageHeadings(tuple(mid), tuple(top), tuple(missing))
    return result


def next_page_head(
    text_content: str, page: int, headings: dict[int, PageHeadings]
) -> str | None:
    """The framed text that continues a read ending on ``page - 1``, if any.

    The text above the earliest heading found partway down ``page``. Nothing
    when no section starts there (the ranges already cover it), when a heading
    opens it, or when no heading was located (see ``next_page_note``). The
    page marker follows the framing line because agents cite pages from the
    markers: without it the tail's rows would be cited as ``page - 1``.
    """
    entry = headings.get(page)
    if entry is None or entry.opens or not entry.mid:
        return None
    offset, node = entry.mid[0]
    head = extract_page_text(text_content, page)[:offset].rstrip()
    framing = (
        f"=== Continued from the top of page {page}, up to the next section "
        f'heading "{node.title.strip()}" ==='
    )
    return f"{framing}\n--- PAGE {page} ---\n{head}"


def next_page_note(page: int, headings: dict[int, PageHeadings]) -> str | None:
    """A NOTE when a section starts on ``page`` but no heading could be placed.

    There is no safe cut point then, and appending the whole page doubled
    every read on a ToC whose titles do not match the printed headings (a
    paraphrased LLM ToC). Saying where the section may continue keeps the miss
    visible and leaves the fetch to the agent.
    """
    entry = headings.get(page)
    if entry is None or entry.opens or entry.mid or not entry.unlocated:
        return None
    return (
        f"=== NOTE: a section starts on page {page}, but its heading could not "
        "be located there, so this range may end before its section does. Read "
        f"page {page} too if values seem to be missing. ==="
    )


def mark_section_start(
    section: str, page: int, headings: dict[int, PageHeadings]
) -> str:
    """Mark where the read's own text begins on its first page, ``page``.

    Only the first page needs it: there, rows above the first mid-page heading
    are the previous section's tail. Later headings in a read are its own
    structure, already visible as headings, and marking them all cost a median
    5-10% of read characters on the corpus. No marker when a heading opens the
    page or could not be located, since the text above a later heading may
    then be the read's own section.

    ``section`` is the raw slice ``extract_section_text`` returns. The offset is
    rebased past the ``--- PAGE N ---`` marker and the line breaks ``page_texts``
    strips after it, so it lands where search offsets point.
    """
    entry = headings.get(page)
    # An unlocated heading may be the one that opens the page, and then the
    # text above the mid-page heading is the read's own: claim nothing.
    if entry is None or entry.opens or entry.unlocated or not entry.mid:
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
    flat_nodes: list[TocNode],
    headings: dict[int, PageHeadings],
    page: int,
    offset: int,
) -> TocNode | None:
    """The section owning ``offset`` on ``page``: the latest heading before it.

    ``flat_nodes`` is ``flatten_nodes(nodes)``, passed in so a search resolves
    it once rather than per match. Candidates are sections whose range reaches
    the previous page, plus those whose heading on ``page`` opens it or sits
    above ``offset``. The latest by position wins, and among equal positions
    the later in document order, which is the deeper one. Position, not ToC
    order, so an outline bookmarked out of page order cannot pick a section
    that ended long before. A heading that could not be located has no
    position and is skipped: counting it at offset 0 handed a child the text
    above its own parent's heading.
    """
    entry = headings.get(page, PageHeadings())
    placed = {id(node): 0 for node in entry.top}
    placed.update({id(node): at for at, node in entry.mid})
    best: TocNode | None = None
    best_key: tuple[int, int, int] | None = None
    for index, node in enumerate(flat_nodes):
        if node.start_page < page:
            if node.end_page < page - 1:
                continue
            key = (node.start_page, 0, index)
        elif node.start_page == page and placed.get(id(node), offset + 1) <= offset:
            key = (page, placed[id(node)], index)
        else:
            continue
        if best_key is None or key > best_key:
            best, best_key = node, key
    return best
