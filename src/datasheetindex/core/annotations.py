"""Footnote and cross-reference detection for TocNode enrichment."""

from __future__ import annotations

import re

from datasheetindex.core.textfile import extract_section_text
from datasheetindex.models import TocNode, flatten_nodes

# --- Footnote patterns ---
_BARE_NUMERIC_RE = re.compile(r"(?<!\d)([1-9]\d?)\)")
_NOTE_STYLE_RE = re.compile(r"(Note\s+\d+)")
_PAREN_NUMERIC_RE = re.compile(r"\((\d+)\)")

# --- Cross-reference patterns ---
_FIGURE_REF_RE = re.compile(
    r"(?:[Ss]ee|[Rr]efer\s+to)\s+((?:Figure|Fig\.)\s+[\d\-\.]+)"
)
_TABLE_REF_RE = re.compile(r"(?:[Ss]ee|[Rr]efer\s+to)\s+(Table\s+[\d\-\.]+)")
_SECTION_REF_RE = re.compile(
    r"(?:[Ss]ee|[Rr]efer\s+to)\s+[Ss]ection\s+([\d]+(?:\.[\d]+)+)"
)

_PAGE_MARKER_RE = re.compile(r"--- PAGE \d+ ---")


def _strip_page_markers(text: str) -> str:
    """Remove PAGE markers to avoid false positives in footnote detection."""
    return _PAGE_MARKER_RE.sub("", text)


def enrich_with_footnote_markers(
    nodes: list[TocNode], text_content: str
) -> list[TocNode]:
    """Detect footnote markers in each node's page range text.

    Modifies nodes in-place and returns them for convenience.
    """
    _footnotes_recursive(nodes, text_content)
    return nodes


def _footnotes_recursive(nodes: list[TocNode], text_content: str) -> None:
    """Walk the tree and populate footnote_markers for each node."""
    for node in nodes:
        section_text = extract_section_text(
            text_content, node.start_page, node.end_page
        )
        clean_text = _strip_page_markers(section_text)

        markers: list[str] = []
        seen: set[str] = set()

        # Bare numeric: 1), 2), etc.
        for m in _BARE_NUMERIC_RE.finditer(clean_text):
            marker = m.group(1) + ")"
            if marker not in seen:
                seen.add(marker)
                markers.append(marker)

        # Note style: Note 1, Note 2, etc.
        for m in _NOTE_STYLE_RE.finditer(clean_text):
            marker = m.group(1)
            if marker not in seen:
                seen.add(marker)
                markers.append(marker)

        # Parenthesized: (1), (2), ... only N <= 20
        for m in _PAREN_NUMERIC_RE.finditer(clean_text):
            num = int(m.group(1))
            if num <= 20:
                marker = f"({m.group(1)})"
                if marker not in seen:
                    seen.add(marker)
                    markers.append(marker)

        node.footnote_markers = markers

        if node.nodes:
            _footnotes_recursive(node.nodes, text_content)


def _build_section_map(nodes: list[TocNode]) -> dict[str, str]:
    """Build a mapping from section numbers to node_ids.

    Parses section numbers from node titles (e.g., "5.2 Electrical Specs"
    -> section "5.2").
    """
    section_map: dict[str, str] = {}
    for node in flatten_nodes(nodes):
        # Try to extract a leading section number like "5.2" or "1.2.3"
        m = re.match(r"([\d]+(?:\.[\d]+)+)", node.title.strip())
        if m:
            section_map[m.group(1)] = node.node_id
    return section_map


def enrich_with_cross_references(
    nodes: list[TocNode], text_content: str
) -> list[TocNode]:
    """Detect cross-references (see Figure/Table/Section) in each node.

    Modifies nodes in-place and returns them for convenience.
    """
    section_map = _build_section_map(nodes)
    _crossrefs_recursive(nodes, text_content, section_map)
    return nodes


def _crossrefs_recursive(
    nodes: list[TocNode], text_content: str, section_map: dict[str, str]
) -> None:
    """Walk the tree and populate cross_references for each node."""
    for node in nodes:
        section_text = extract_section_text(
            text_content, node.start_page, node.end_page
        )

        refs: list[dict] = []
        seen: set[tuple[str, str]] = set()

        # Figure references
        for m in _FIGURE_REF_RE.finditer(section_text):
            key = ("figure", m.group(1))
            if key not in seen:
                seen.add(key)
                refs.append(
                    {
                        "text": m.group(0),
                        "type": "figure",
                        "target": m.group(1),
                    }
                )

        # Table references
        for m in _TABLE_REF_RE.finditer(section_text):
            key = ("table", m.group(1))
            if key not in seen:
                seen.add(key)
                refs.append(
                    {
                        "text": m.group(0),
                        "type": "table",
                        "target": m.group(1),
                    }
                )

        # Section references
        for m in _SECTION_REF_RE.finditer(section_text):
            section_num = m.group(1)
            key = ("section", section_num)
            if key not in seen:
                seen.add(key)
                ref: dict = {
                    "text": m.group(0),
                    "type": "section",
                    "target": section_num,
                }
                if section_num in section_map:
                    ref["target_node_id"] = section_map[section_num]
                refs.append(ref)

        node.cross_references = refs

        if node.nodes:
            _crossrefs_recursive(node.nodes, text_content, section_map)
