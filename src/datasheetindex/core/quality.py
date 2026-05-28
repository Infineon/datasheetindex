"""Page-level quality scoring and ToC quality assessment."""

from __future__ import annotations

from datasheetindex.models import TocNode, TocQuality, flatten_nodes


def assess_toc_quality(nodes: list[TocNode], total_pages: int) -> TocQuality:
    """Score the quality of an extracted ToC.

    Weighted score from 4 factors:
    - Entry count (30%): enough entries to be useful, not too many
    - Page coverage (30%): how much of the document is covered
    - Hierarchy depth (20%): multi-level structure is better
    - Title quality (20%): meaningful titles vs numeric/short ones
    """
    if not nodes or total_pages == 0:
        return TocQuality(
            score=0.0,
            entry_count=0,
            max_depth=0,
            page_coverage=0.0,
            recommend_summaries=True,
            details="No ToC entries found",
        )

    flat = flatten_nodes(nodes)
    entry_count = len(flat)
    max_depth = max(n.level for n in flat)

    # --- Entry count score (30%) ---
    # Sweet spot: 5-30 entries
    if entry_count < 3:
        entry_score = 0.2
    elif entry_count <= 5:
        entry_score = 0.6
    elif entry_count <= 30:
        entry_score = 1.0
    elif entry_count <= 60:
        entry_score = 0.7
    else:
        entry_score = 0.4

    # --- Page coverage score (30%) ---
    covered_pages: set[int] = set()
    for node in flat:
        for p in range(node.start_page, node.end_page + 1):
            covered_pages.add(p)
    page_coverage = len(covered_pages) / total_pages if total_pages > 0 else 0.0
    coverage_score = min(page_coverage, 1.0)

    # --- Hierarchy depth score (20%) ---
    if max_depth >= 3:
        depth_score = 1.0
    elif max_depth == 2:
        depth_score = 0.7
    else:
        depth_score = 0.3

    # --- Title quality score (20%) ---
    good_titles = 0
    for node in flat:
        title = node.title.strip()
        if len(title) >= 3 and not title.replace(".", "").isdigit():
            good_titles += 1
    title_score = good_titles / entry_count if entry_count > 0 else 0.0

    # Weighted total
    score = (
        0.3 * entry_score + 0.3 * coverage_score + 0.2 * depth_score + 0.2 * title_score
    )
    score = round(score, 3)

    recommend_summaries = score < 0.5 or entry_count > 40

    details_parts = [
        f"entry_score={entry_score:.2f}",
        f"coverage_score={coverage_score:.2f}",
        f"depth_score={depth_score:.2f}",
        f"title_score={title_score:.2f}",
    ]

    return TocQuality(
        score=score,
        entry_count=entry_count,
        max_depth=max_depth,
        page_coverage=round(page_coverage, 3),
        recommend_summaries=recommend_summaries,
        details=", ".join(details_parts),
    )
