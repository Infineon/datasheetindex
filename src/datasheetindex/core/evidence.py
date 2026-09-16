"""Artifact-local source evidence and deterministic grounding helpers.

Evidence is written to its own ``<stem>.evidence.jsonl`` next to the ToC JSON,
never into it: one record per text block makes the index 40-50x the size of
the navigation tree, which must stay small enough for an agent to read.

The file is JSON Lines so that it can be navigated without parsing all of it:
a header line, then one element per line in page order (sections, text blocks
in reading order, figures, tables). ``grep`` for a phrase or an ``element_id``
returns whole records, and each text block carries its own ``text``, so a hit
already names its geometry and section without slicing the text artifact by
offset.

Geometry convention, for every element type: ``bbox`` is ``[x0, y0, x1, y1]``
in PDF points in the *displayed* (rotated) page space -- the space of
``page.rect``, ``find_tables()`` and ``inspect_page`` -- and ``region`` is that
box normalized to ``page.rect``. Text ranges are half-open ``{start, end}``
character offsets; page numbers are 1-indexed and inclusive.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    import pymupdf

EVIDENCE_SCHEMA_VERSION = 1


class EvidenceRange(TypedDict):
    """Half-open character range in the page-matched text artifact."""

    start: int
    end: int


class EvidenceElement(TypedDict, total=False):
    """A source element that can support an agent's claim."""

    element_id: str
    element_type: str
    page: int
    text_range: EvidenceRange
    page_text_range: EvidenceRange
    start_page: int
    end_page: int
    bbox: list[float]
    region: dict[str, float]
    breadcrumb: str
    node_id: str
    source_kind: str
    text: str
    caption: str
    caption_source: str


EVIDENCE_FORMAT = "datasheetindex-evidence"

#: Within a page: the section a reader enters first, then its text in reading
#: order, then the figures and tables drawn on it.
_TYPE_ORDER = {"section": 0, "text_block": 1, "figure_caption": 2, "figure": 3}


def evidence_file_name(stem: str) -> str:
    """The evidence artifact's file name, beside ``<stem>.json``."""

    return f"{stem}.evidence.jsonl"


def serialize_evidence(elements: Iterable[EvidenceElement]) -> str:
    """Render the evidence file: a header line, then one element per line."""

    import json

    ordered = sorted(
        elements,
        key=lambda element: (
            element.get("page", 0),
            _TYPE_ORDER.get(element.get("element_type", ""), len(_TYPE_ORDER)),
            element.get("element_id", ""),
        ),
    )
    header = {
        "format": EVIDENCE_FORMAT,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "element_count": len(ordered),
    }
    lines = [header, *ordered]
    return "".join(
        json.dumps(line, ensure_ascii=False, separators=(",", ":")) + "\n"
        for line in lines
    )


def parse_evidence(content: str) -> list[dict[str, Any]]:
    """Read ``serialize_evidence`` output back; raises on a foreign file."""

    import json

    lines = content.splitlines()
    header = json.loads(lines[0]) if lines else {}
    if header.get("format") != EVIDENCE_FORMAT:
        raise ValueError("not a datasheetindex evidence file")
    if header.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise ValueError(
            f"evidence schema {header.get('schema_version')} is not "
            f"{EVIDENCE_SCHEMA_VERSION}"
        )
    return [json.loads(line) for line in lines[1:] if line]


def displayed_bbox(bbox: Sequence[float], page: pymupdf.Page) -> list[float]:
    """Map an unrotated-page box (``get_text``, ``get_image_info``) to page.rect space.

    Identity on an unrotated page. On a rotated one the raw box names a
    different part of the rendered page -- or none of it -- so a region built
    from it crops the wrong area in ``inspect_page``.
    """

    import pymupdf

    rect = pymupdf.Rect(bbox) * page.rotation_matrix
    return [rect.x0, rect.y0, rect.x1, rect.y1]


def _range(start: int, end: int) -> EvidenceRange:
    return {"start": start, "end": end}


def _region(
    bbox: Sequence[float], page_width: float, page_height: float
) -> dict[str, float] | None:
    x0, y0, x1, y1 = (float(value) for value in bbox)
    if page_width <= 0 or page_height <= 0:
        return None
    x0 = max(0.0, min(page_width, x0))
    x1 = max(0.0, min(page_width, x1))
    y0 = max(0.0, min(page_height, y0))
    y1 = max(0.0, min(page_height, y1))
    if x0 >= x1 or y0 >= y1:
        return None
    return {
        "left": x0 / page_width,
        "right": x1 / page_width,
        "top": y0 / page_height,
        "bottom": y1 / page_height,
    }


def text_element(
    *,
    element_id: str,
    page: int,
    canonical_start: int,
    canonical_end: int,
    page_start: int,
    page_end: int,
    text: str,
    bbox: Sequence[float],
    page_width: float,
    page_height: float,
) -> EvidenceElement:
    """Build an evidence record for one retained, ordered text block.

    ``text`` is exactly the artifact slice ``text_range`` names; it is carried
    so the record can be found and read on its own.
    """

    element: EvidenceElement = {
        "element_id": element_id,
        "element_type": "text_block",
        "page": page,
        "text": text,
        "text_range": _range(canonical_start, canonical_end),
        "page_text_range": _range(page_start, page_end),
        "bbox": [float(value) for value in bbox],
        "source_kind": "literal",
    }
    region = _region(bbox, page_width, page_height)
    if region is not None:
        element["region"] = region
    return element


def range_overlaps(element: Mapping[str, Any], start: int, end: int) -> bool:
    """Whether a canonical text span intersects an element's text range."""

    text_range = element.get("text_range")
    if not isinstance(text_range, Mapping):
        return False
    element_start = text_range.get("start")
    element_end = text_range.get("end")
    if not isinstance(element_start, int) or not isinstance(element_end, int):
        return False
    return start < element_end and end > element_start


def ground_range(
    elements: Iterable[Mapping[str, Any]], start: int, end: int
) -> list[dict[str, Any]]:
    """Return elements intersecting a canonical artifact text span.

    The returned dictionaries are copies, so callers can add claim-specific
    metadata without mutating the cached artifact.
    """

    if start < 0 or end < start:
        raise ValueError("evidence range must be a non-negative half-open span")
    return [
        dict(element) for element in elements if range_overlaps(element, start, end)
    ]


def ground_page_span(
    elements: Iterable[Mapping[str, Any]], page: int, start: int, end: int
) -> list[dict[str, Any]]:
    """Ground a page-local search span against page-local element ranges."""

    if start < 0 or end < start:
        raise ValueError("evidence range must be a non-negative half-open span")
    matches: list[dict[str, Any]] = []
    for element in elements:
        if element.get("page") != page:
            continue
        page_range = element.get("page_text_range")
        if not isinstance(page_range, Mapping):
            continue
        element_start = page_range.get("start")
        element_end = page_range.get("end")
        if (
            isinstance(element_start, int)
            and isinstance(element_end, int)
            and start < element_end
            and end > element_start
        ):
            matches.append(dict(element))
    return matches


def link_to_toc(
    elements: list[EvidenceElement], nodes: Iterable[Any]
) -> list[EvidenceElement]:
    """Attach each page element to its deepest containing ToC node.

    Uses ``structure.find_node_for_page`` so an element and a ``search_text``
    hit on the same page always name the same section.
    """

    # Deferred: structure imports this module at load time.
    from datasheetindex.core.structure import find_node_for_page

    node_list = list(nodes)
    node_by_page: dict[int, Any] = {}
    for element in elements:
        if element.get("element_type") == "section":
            continue
        page = element.get("page")
        if not isinstance(page, int):
            continue
        if page not in node_by_page:
            node_by_page[page] = find_node_for_page(node_list, page)
        node = node_by_page[page]
        if node is None:
            continue
        if node.breadcrumb:
            element["breadcrumb"] = node.breadcrumb
        if node.node_id:
            element["node_id"] = node.node_id
    return elements


def section_elements(
    nodes: Iterable[Any], *, source_kind: str = "literal"
) -> list[EvidenceElement]:
    """Represent ToC nodes as source elements with stable artifact-local IDs."""

    elements: list[EvidenceElement] = []

    def walk(items: Iterable[Any]) -> None:
        for node in items:
            elements.append(
                {
                    "element_id": f"section-{node.node_id}",
                    "element_type": "section",
                    "page": node.start_page,
                    "start_page": node.start_page,
                    "end_page": node.end_page,
                    "source_kind": source_kind,
                    "node_id": node.node_id,
                    "breadcrumb": node.breadcrumb,
                }
            )
            walk(getattr(node, "nodes", ()))

    walk(nodes)
    return elements


def figure_elements(
    figures: Iterable[Mapping[str, object]],
) -> list[EvidenceElement]:
    """Evidence records for the figure index, leaving ``figures`` untouched.

    The records are self-contained -- the evidence file is read on its own --
    so they repeat the figure's geometry and caption. Text-layer caption
    entries have no geometry, and their records carry none.
    """

    counters: dict[tuple[int, str], int] = {}
    elements: list[EvidenceElement] = []
    for figure in figures:
        page = figure.get("page")
        kind = str(figure.get("kind", "figure"))
        if not isinstance(page, int):
            continue
        key = (page, kind)
        ordinal = counters.get(key, 0)
        counters[key] = ordinal + 1
        record: EvidenceElement = {
            "element_id": f"p{page:04d}-{kind}-{ordinal:03d}",
            "element_type": "figure" if kind == "raster" else "figure_caption",
            "page": page,
            "source_kind": (
                "generated"
                if figure.get("caption_source") in {"generated", "llm"}
                else "literal"
            ),
        }
        bbox = figure.get("bbox")
        if isinstance(bbox, list) and all(
            isinstance(value, (int, float)) for value in bbox
        ):
            record["bbox"] = [float(value) for value in bbox]
        region = figure.get("region")
        if isinstance(region, dict) and all(
            isinstance(key, str) and isinstance(value, (int, float))
            for key, value in region.items()
        ):
            record["region"] = {key: float(value) for key, value in region.items()}
        caption = figure.get("caption")
        if isinstance(caption, str):
            record["caption"] = caption
        caption_source = figure.get("caption_source")
        if isinstance(caption_source, str):
            record["caption_source"] = caption_source
        elements.append(record)
    return elements


def table_element(
    *,
    element_id: str,
    page: int,
    bbox: Sequence[float],
    page_width: float,
    page_height: float,
) -> EvidenceElement:
    """Build an evidence record for one whole-table region."""

    element: EvidenceElement = {
        "element_id": element_id,
        "element_type": "table",
        "page": page,
        "bbox": [float(value) for value in bbox],
        "source_kind": "literal",
    }
    region = _region(bbox, page_width, page_height)
    if region is not None:
        element["region"] = region
    return element
