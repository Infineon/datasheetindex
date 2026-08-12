"""Tests for ToC quality scoring."""

import pytest

from datasheetindex.core.quality import assess_toc_quality
from datasheetindex.index import TOC_FALLBACK_THRESHOLD
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
    """A ToC with many entries should recommend summaries.

    These titles are enumerated, so informativeness collapses them to one key
    and the score falls to 0.016 -- which satisfies ``score < 0.5`` as well.
    The test below is the one that isolates the entry-count rule.
    """
    nodes = [
        TocNode(title=f"Section {i}", level=1, start_page=i, end_page=i)
        for i in range(1, 50)
    ]
    quality = assess_toc_quality(nodes, total_pages=50)
    assert quality.recommend_summaries is True
    assert quality.entry_count == 49


def _distinct_titles(count):
    """``count`` titles that survive ``normalize_key``'s digit masking.

    Enumerated titles collapse to a single key, which is exactly what the
    informativeness factor is for -- so a test about the *entry count* must not
    use them, or it measures informativeness instead.
    """
    from string import ascii_uppercase as letters

    return [
        f"Register {letters[i // len(letters)]}{letters[i % len(letters)]} control"
        for i in range(count)
    ]


def test_a_high_scoring_toc_still_recommends_summaries_past_the_entry_cap():
    """Isolates ``entry_count > 40`` from the ``score < 0.5`` clause beside it.

    ``test_large_toc_recommends_summaries`` above satisfies both clauses since
    informativeness landed, so deleting ``or entry_count > 40`` left it green.
    These 45 titles are all distinct, so the score clears 0.5 and the entry cap
    is the only rule that can still recommend summaries.
    """
    nodes = [
        TocNode(title=title, level=1, start_page=i + 1, end_page=i + 1)
        for i, title in enumerate(_distinct_titles(45))
    ]
    quality = assess_toc_quality(nodes, total_pages=45)

    assert quality.entry_count == 45
    assert quality.score >= 0.5, (
        "the score clause must not be able to satisfy this test on its own"
    )
    assert quality.recommend_summaries is True


def test_single_entry_low_score():
    """A ToC with only one entry should have a relatively low score."""
    nodes = [TocNode(title="Only Section", level=1, start_page=1, end_page=10)]
    quality = assess_toc_quality(nodes, total_pages=10)
    assert quality.score < 0.7
    assert quality.entry_count == 1


def _enumerated(count, template="Page %d"):
    return [
        TocNode(title=template % i, level=1, start_page=i, end_page=i)
        for i in range(1, count + 1)
    ]


def test_page_enumeration_falls_below_the_fallback_threshold():
    """One entry per page maximises entry count and page coverage by
    construction, so the structural factors cannot catch this."""
    for pages in (20, 134):
        quality = assess_toc_quality(_enumerated(pages), total_pages=pages)
        assert quality.score < TOC_FALLBACK_THRESHOLD, pages


def test_section_enumeration_falls_below_the_threshold():
    quality = assess_toc_quality(_enumerated(49, "Section %d"), total_pages=49)
    assert quality.score < TOC_FALLBACK_THRESHOLD


_DISTINCT_TOPICS = [
    "Introduction",
    "System Overview",
    "Electrical Characteristics",
    "Absolute Maximum Ratings",
    "Operating Conditions",
    "Pin Configuration",
    "Package Information",
    "Ordering Information",
    "Functional Description",
    "Timer Module",
    "UART Interface",
    "SPI Interface",
    "I2C Interface",
    "ADC Module",
    "DAC Module",
    "PWM Generator",
    "GPIO Configuration",
    "Interrupt Controller",
    "Clock System",
    "Reset Behavior",
    "Power Management",
    "Low Power Modes",
    "Memory Map",
    "Flash Programming",
    "Bootloader",
    "Watchdog Timer",
    "Real Time Clock",
    "CAN Controller",
    "USB Interface",
    "Ethernet MAC",
    "Security Features",
    "Cryptography Engine",
    "Random Number Generator",
    "Temperature Sensor",
    "Comparator Block",
    "Operational Amplifier",
    "Voltage Reference",
    "Brown Out Detector",
    "Debug Interface",
    "Revision History",
]


def test_a_half_enumerated_outline_is_degraded_but_not_condemned():
    """The factor is continuous, not a cliff.

    Uses distinct topic titles rather than a "Topic %d description" template:
    that template collides under digit masking exactly like "Page %d" does,
    so a uniformly-numbered "real" half does not actually exercise a
    half-real, half-enumerated split.
    """
    assert len(_DISTINCT_TOPICS) == 40
    real = [
        TocNode(title=topic, level=1, start_page=i, end_page=i)
        for i, topic in enumerate(_DISTINCT_TOPICS, start=1)
    ]
    nodes = real + [
        TocNode(title=f"Page {i}", level=1, start_page=i, end_page=i)
        for i in range(41, 81)
    ]
    quality = assess_toc_quality(nodes, total_pages=80)
    assert quality.score < 0.5
    assert quality.score > TOC_FALLBACK_THRESHOLD


def test_same_named_subsections_under_different_parents_are_not_penalised():
    """The false positive the breadcrumb keying exists to prevent. Two chapters
    can both contain 'Register description' and both are real."""
    nodes = [
        TocNode(
            title="1 Timer",
            level=1,
            start_page=1,
            end_page=4,
            breadcrumb="1 Timer",
            nodes=[
                TocNode(
                    title="1.1 Register description",
                    level=2,
                    start_page=1,
                    end_page=4,
                    breadcrumb="1 Timer > 1.1 Register description",
                )
            ],
        ),
        TocNode(
            title="2 UART",
            level=1,
            start_page=5,
            end_page=10,
            breadcrumb="2 UART",
            nodes=[
                TocNode(
                    title="2.1 Register description",
                    level=2,
                    start_page=5,
                    end_page=10,
                    breadcrumb="2 UART > 2.1 Register description",
                )
            ],
        ),
    ]
    quality = assess_toc_quality(nodes, total_pages=10)
    assert quality.score == 0.82


def test_numbered_siblings_survive():
    """Digit masking collapses Port P1..P8, which are genuinely distinct
    sections -- the ti_msp430f5529 shape. Even a toy outline where this
    collision is a *majority* must not be condemned.

    The Ports subtree alone is 8 of its own 9 entries colliding. Adding
    three more chapters here does not dilute that into a minority -- it is
    still 7 of 12 entries colliding (0.417 informativeness, score 0.392) --
    so this only shows that a majority-collision outline can still clear
    the 0.3 line, not that it is diluted away. The minority claim itself is
    a real-document fact, not something this fixture demonstrates: on
    ti_msp430f5529, measured directly, the score goes 0.820 -> 0.741,
    nowhere near the 0.3 fallback line, because in the real document the
    Ports subtree sits among far more distinguishable entries (worst
    measured collision fraction across the corpus is 21%, not this
    fixture's 58%).
    """
    other_chapters = [
        TocNode(title=title, level=1, start_page=p, end_page=p)
        for p, title in enumerate(
            ["1 Introduction", "2 System Overview", "4 Revision History"],
            start=41,
        )
    ]
    nodes = [
        TocNode(
            title="3 Ports",
            level=1,
            start_page=1,
            end_page=40,
            breadcrumb="3 Ports",
            nodes=[
                TocNode(
                    title=f"3.{i} Port P{i} input/output",
                    level=2,
                    start_page=i * 4,
                    end_page=i * 4 + 3,
                    breadcrumb=f"3 Ports > 3.{i} Port P{i} input/output",
                )
                for i in range(1, 9)
            ],
        )
    ] + other_chapters
    quality = assess_toc_quality(nodes, total_pages=43)
    assert quality.score > TOC_FALLBACK_THRESHOLD


def test_nodes_without_a_breadcrumb_fall_back_to_the_title():
    """TocNode.breadcrumb defaults to '' and tests construct nodes directly."""
    nodes = [
        TocNode(title="Alpha", level=1, start_page=1, end_page=5),
        TocNode(title="Beta", level=1, start_page=6, end_page=10),
    ]
    quality = assess_toc_quality(nodes, total_pages=10)
    assert quality.score > 0.5


@pytest.mark.real_pdf
def test_the_bundled_datasheet_is_unaffected():
    """The only assertion tying this heuristic to a real document. A change to
    normalize_key that reintroduces collisions must fail loudly here.

    0.82 is measured, not a round target: the outline's 89 entries produce 89
    distinct breadcrumb keys, so informativeness is exactly 1.000 and the
    score is unchanged by this factor.

    **Skipped on a clean clone.** The bundled PSoC datasheet is gitignored and
    absent from the CI checkout that gates releases, so this protection does
    not exist there. The synthetic cases above are what actually run in CI.
    """
    from pathlib import Path

    import pymupdf

    from datasheetindex.core.structure import (
        build_tree,
        compute_end_pages,
        extract_toc,
    )

    pdf = Path(__file__).resolve().parent.parent / (
        "infineon-psoc-6-mcu-cy8c62x8-cy8c62xa-datasheet-datasheet-en.pdf"
    )
    if not pdf.exists():
        pytest.skip("bundled PSoC datasheet not present")

    doc = pymupdf.open(str(pdf))
    try:
        nodes = build_tree(extract_toc(doc), doc.page_count)
        compute_end_pages(nodes, doc.page_count)
        quality = assess_toc_quality(nodes, doc.page_count)
    finally:
        doc.close()
    assert quality.score == 0.82
