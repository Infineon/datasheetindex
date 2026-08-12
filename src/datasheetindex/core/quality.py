"""Page-level quality scoring and ToC quality assessment."""

from __future__ import annotations

from datasheetindex.core.furniture import normalize_key
from datasheetindex.models import TocNode, TocQuality, flatten_nodes


def _informativeness(flat: list[TocNode]) -> float:
    """Fraction of entries that are distinguishable from one another.

    Keyed on the breadcrumb rather than the bare title, because two chapters
    legitimately share a subsection name and the ancestry path separates them:
    measured across a 24-document corpus, bare titles cost the ESP32 technical
    reference manual 27 points (0.727 against 0.995) while the breadcrumb form
    loses almost nothing. ``breadcrumb`` defaults to ``""`` on a directly
    constructed node, so the title is the fallback.

    Digit masking cannot tell "a template plus a counter" from "legitimately
    numbered siblings" -- ``Port P1``/``Port P2`` collide exactly as
    ``Page 1``/``Page 2`` do -- and that is accepted rather than fixed: on real
    documents such collisions are always a minority (worst measured 21%),
    whereas an enumerated outline collapses essentially every entry.
    """
    keys = {normalize_key(node.breadcrumb or node.title) for node in flat}
    return len(keys) / len(flat)


def assess_toc_quality(nodes: list[TocNode], total_pages: int) -> TocQuality:
    """Score the quality of an extracted ToC.

    Weighted score from 4 factors:
    - Entry count (30%): enough entries to be useful, not too many
    - Page coverage (30%): how much of the document is covered
    - Hierarchy depth (20%): multi-level structure is better
    - Title quality (20%): meaningful titles vs numeric/short ones

    The weighted total is then multiplied by ``_informativeness`` -- the
    fraction of entries distinguishable from one another -- which gates the
    score rather than joining it as a fifth weighted factor. See the comment
    above the multiplication for why.
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

    # Informativeness gates the rest rather than joining it as a fifth weighted
    # factor. Three of the four factors above measure whether the outline
    # *spans* the document, and an enumerated outline maximises all three by
    # construction: one entry per page gives a plausible entry count and
    # perfect page coverage. Forcing title_score to zero still leaves such an
    # outline at 0.48-0.66, so a 20% component cannot reach the 0.3 fallback
    # threshold -- and no threshold can, because the ordering is inverted:
    # 'Page 1..20' scores 0.860 against a real 89-entry outline's 0.820. An
    # outline whose entries cannot be told apart cannot route a reader
    # anywhere, however completely it covers the pages, so this multiplies.
    informativeness = _informativeness(flat)
    score = (
        0.3 * entry_score + 0.3 * coverage_score + 0.2 * depth_score + 0.2 * title_score
    ) * informativeness
    score = round(score, 3)

    recommend_summaries = score < 0.5 or entry_count > 40

    details_parts = [
        f"entry_score={entry_score:.2f}",
        f"coverage_score={coverage_score:.2f}",
        f"depth_score={depth_score:.2f}",
        f"title_score={title_score:.2f}",
        f"informativeness={informativeness:.2f}",
    ]

    return TocQuality(
        score=score,
        entry_count=entry_count,
        max_depth=max_depth,
        page_coverage=round(page_coverage, 3),
        recommend_summaries=recommend_summaries,
        details=", ".join(details_parts),
    )
