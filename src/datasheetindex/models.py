"""Data models for datasheetindex artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TocNode:
    """A node in the enriched Table of Contents tree."""

    title: str
    level: int
    start_page: int
    end_page: int = 0
    node_id: str = ""
    breadcrumb: str = ""
    boilerplate_category: str = ""
    has_tables: bool = False
    table_count: int = 0
    summary: str = ""
    continued_tables: list[str] = field(default_factory=list)
    footnote_markers: list[str] = field(default_factory=list)
    cross_references: list[dict] = field(default_factory=list)
    nodes: list[TocNode] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        result: dict = {
            "node_id": self.node_id,
            "title": self.title,
            "level": self.level,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "has_tables": self.has_tables,
            "table_count": self.table_count,
        }
        if self.breadcrumb:
            result["breadcrumb"] = self.breadcrumb
        if self.boilerplate_category:
            result["boilerplate_category"] = self.boilerplate_category
        if self.summary:
            result["summary"] = self.summary
        if self.continued_tables:
            result["continued_tables"] = self.continued_tables
        if self.footnote_markers:
            result["footnote_markers"] = self.footnote_markers
        if self.cross_references:
            result["cross_references"] = self.cross_references
        if self.nodes:
            result["nodes"] = [child.to_dict() for child in self.nodes]
        return result

    @classmethod
    def from_dict(cls, data: dict) -> TocNode:
        """Rebuild a node from ``to_dict`` output.

        ``to_dict`` omits empty fields, so every optional key needs a default
        here or a reloaded artifact would differ from the one that produced it.
        """
        return cls(
            title=data["title"],
            level=data["level"],
            start_page=data["start_page"],
            end_page=data.get("end_page", 0),
            node_id=data.get("node_id", ""),
            breadcrumb=data.get("breadcrumb", ""),
            boilerplate_category=data.get("boilerplate_category", ""),
            has_tables=data.get("has_tables", False),
            table_count=data.get("table_count", 0),
            summary=data.get("summary", ""),
            continued_tables=list(data.get("continued_tables", [])),
            footnote_markers=list(data.get("footnote_markers", [])),
            cross_references=list(data.get("cross_references", [])),
            nodes=[cls.from_dict(child) for child in data.get("nodes", [])],
        )


@dataclass
class TocQuality:
    """Quality assessment for the extracted ToC."""

    score: float = 0.0
    entry_count: int = 0
    max_depth: int = 0
    page_coverage: float = 0.0
    recommend_summaries: bool = False
    details: str = ""

    def to_dict(self) -> dict:
        """Serialize every field, ``details`` included.

        Deliberately not what ``index.py`` writes into the ToC JSON's
        ``toc_quality`` block, which omits ``details`` and must stay
        byte-identical. This one exists for the build sidecar.
        """
        return {
            "score": self.score,
            "entry_count": self.entry_count,
            "max_depth": self.max_depth,
            "page_coverage": self.page_coverage,
            "recommend_summaries": self.recommend_summaries,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TocQuality:
        """Rebuild from ``to_dict`` output."""
        return cls(
            score=data.get("score", 0.0),
            entry_count=data.get("entry_count", 0),
            max_depth=data.get("max_depth", 0),
            page_coverage=data.get("page_coverage", 0.0),
            recommend_summaries=data.get("recommend_summaries", False),
            details=data.get("details", ""),
        )


@dataclass
class DatasheetArtifacts:
    """Output of the DatasheetIndex build process."""

    json_path: Path | None = None
    text_path: Path | None = None
    json_data: dict = field(default_factory=dict)
    text_content: str = ""
    toc_quality: TocQuality | None = None
    # The typed enriched ToC tree (same content as ``json_data["toc"]`` before
    # serialization). Retained so tools can resolve structure via TocNode
    # attributes instead of reaching into the serialized dict shape.
    nodes: list[TocNode] = field(default_factory=list)
    # True when LLM work this build was eligible for did not produce its
    # result -- no callable was obtainable, or the call ran and raised. A
    # rejected fallback candidate is a completed decision and does NOT set
    # this. Read by the artifact cache, which refuses to reuse a degraded
    # build, from disk or from memory.
    llm_enrichment_incomplete: bool = False
    llm_enrichment_notes: tuple[str, ...] = ()


def flatten_nodes(nodes: list[TocNode]) -> list[TocNode]:
    """Collect all nodes into a flat list via depth-first traversal."""
    result: list[TocNode] = []
    for node in nodes:
        result.append(node)
        if node.nodes:
            result.extend(flatten_nodes(node.nodes))
    return result
