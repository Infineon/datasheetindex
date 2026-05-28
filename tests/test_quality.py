"""Tests for ToC quality scoring."""

from datasheetindex.core.quality import assess_toc_quality
from datasheetindex.models import TocNode


def test_empty_toc_scores_zero():
    quality = assess_toc_quality([], total_pages=10)
    assert quality.score == 0.0
    assert quality.entry_count == 0
    assert quality.recommend_summaries is True


def test_zero_pages_scores_zero():
    quality = assess_toc_quality([], total_pages=0)
    assert quality.score == 0.0


def test_good_toc_scores_high():
    """A well-structured ToC with good coverage should score > 0.6."""
    nodes = [
        TocNode(
            title="Overview",
            level=1,
            start_page=1,
            end_page=4,
            nodes=[
                TocNode(title="Introduction", level=2, start_page=1, end_page=2),
                TocNode(title="Features", level=2, start_page=3, end_page=4),
            ],
        ),
        TocNode(
            title="Electrical Characteristics",
            level=1,
            start_page=5,
            end_page=8,
            nodes=[
                TocNode(
                    title="Absolute Maximum Ratings",
                    level=2,
                    start_page=5,
                    end_page=6,
                ),
                TocNode(
                    title="Operating Conditions",
                    level=2,
                    start_page=7,
                    end_page=8,
                ),
            ],
        ),
        TocNode(
            title="Pin Configuration",
            level=1,
            start_page=9,
            end_page=10,
        ),
    ]
    quality = assess_toc_quality(nodes, total_pages=10)
    assert quality.score > 0.6
    assert quality.entry_count == 7
    assert quality.max_depth == 2


def test_numeric_titles_penalized():
    """Entries with purely numeric titles should lower the title score."""
    nodes = [
        TocNode(title="1", level=1, start_page=1, end_page=3),
        TocNode(title="2", level=1, start_page=4, end_page=6),
        TocNode(title="3", level=1, start_page=7, end_page=10),
    ]
    quality = assess_toc_quality(nodes, total_pages=10)
    # Title score should be 0 since all titles are numeric/short
    # Full page coverage still gives some score, but it should be lower
    # than a good ToC with meaningful titles
    assert quality.score < 0.6


def test_large_toc_recommends_summaries():
    """A ToC with many entries should recommend summaries."""
    nodes = [
        TocNode(title=f"Section {i}", level=1, start_page=i, end_page=i)
        for i in range(1, 50)
    ]
    quality = assess_toc_quality(nodes, total_pages=50)
    assert quality.recommend_summaries is True
    assert quality.entry_count == 49


def test_single_entry_low_score():
    """A ToC with only one entry should have a relatively low score."""
    nodes = [TocNode(title="Only Section", level=1, start_page=1, end_page=10)]
    quality = assess_toc_quality(nodes, total_pages=10)
    assert quality.score < 0.7
    assert quality.entry_count == 1
