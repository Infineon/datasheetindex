"""Tests for boilerplate classification of ToC nodes."""

import pytest

from datasheetindex.core.boilerplate import (
    classify_title,
    flag_boilerplate,
)
from datasheetindex.models import TocNode


@pytest.mark.parametrize(
    "title,expected",
    [
        # legal
        ("Disclaimer", "legal"),
        ("Legal Disclaimer", "legal"),
        ("Important Notice", "legal"),
        ("Important Information", "legal"),
        ("Trademarks", "legal"),
        ("Copyright Notice", "legal"),
        ("Patents", "legal"),
        ("Terms and Conditions", "legal"),
        ("Safety Precautions", "legal"),
        ("ESD Caution", "legal"),
        # ordering
        ("Ordering Information", "ordering"),
        ("Ordering Guide", "ordering"),
        ("Part Numbers", "ordering"),
        ("Part Numbering Information", "ordering"),
        ("Marking Information", "ordering"),
        ("Device Marking", "ordering"),
        ("How to Order", "ordering"),
        # revision
        ("Revision History", "revision"),
        ("Document History", "revision"),
        ("Change Log", "revision"),
        ("Revisions", "revision"),
        ("Version Control", "revision"),
        ("History of Changes", "revision"),
        # contact
        ("Contact Information", "contact"),
        ("Sales Offices", "contact"),
        ("Worldwide Sales", "contact"),
        ("Where to Buy", "contact"),
        ("Customer Support", "contact"),
        # toc
        ("Table of Contents", "toc"),
        ("Contents", "toc"),
        ("List of Figures", "toc"),
        ("List of Tables", "toc"),
        ("Index", "toc"),
        # glossary
        ("Glossary", "glossary"),
        ("Abbreviations", "glossary"),
        ("Acronyms and Abbreviations", "glossary"),
        ("Terminology", "glossary"),
        ("Definitions", "glossary"),
    ],
)
def test_classify_title_positive(title, expected):
    assert classify_title(title) == expected


@pytest.mark.parametrize(
    "title",
    [
        # Substantive sections that mention boilerplate keywords but aren't
        # themselves boilerplate.
        "Trademark Licensing Strategy",
        "Glossary of Register Names",
        "Pin Configuration and Description",
        "Electrical Characteristics",
        "Block Diagram",
        "Operating Conditions",
        "Communication Protocol",
        "Power Management",
        "Functional Description",
        "Application Information",
        "Order of Operations",  # contains "order" but not ordering info
        "Revision A Functional Updates",  # describes content for rev A, not history
        # Bare-word regression: these used to false-match `legal` because the
        # qualifier was optional. They are common substantive titles in real
        # vendor datasheets (NXP, ST, Renesas use "Information" as a chapter
        # title; some datasheets have a bare "Notice" chapter).
        "Information",
        "Notice",
        "Notices",
        "Liability",
        # Single-letter prefix regression: leading "A " followed only by
        # whitespace must not be stripped, or "A Glossary of Terms" would
        # misclassify as `glossary`.
        "A Glossary of Common Terms",
        # Empty / nearly empty
        "",
        "   ",
        "1.",
    ],
)
def test_classify_title_negative(title):
    assert classify_title(title) == ""


@pytest.mark.parametrize(
    "title,expected",
    [
        # With section number prefixes
        ("12 Revision History", "revision"),
        ("12.1 Revision History", "revision"),
        ("Appendix A: Ordering Information", "ordering"),
        ("Chapter 3 Contents", "toc"),
        ("A. Trademarks", "legal"),
        # With trailing punctuation
        ("Disclaimer:", "legal"),
        ("Glossary.", "glossary"),
        # Mixed case
        ("REVISION HISTORY", "revision"),
        ("ordering information", "ordering"),
    ],
)
def test_classify_title_with_prefixes_and_punctuation(title, expected):
    assert classify_title(title) == expected


def test_flag_boilerplate_top_level_only():
    nodes = [
        TocNode(title="Electrical Characteristics", level=1, start_page=1),
        TocNode(title="Revision History", level=1, start_page=10),
    ]
    flag_boilerplate(nodes)
    assert nodes[0].boilerplate_category == ""
    assert nodes[1].boilerplate_category == "revision"


def test_flag_boilerplate_children_inherit_from_boilerplate_parent():
    """Subsections of a boilerplate parent inherit the parent's category."""
    child = TocNode(title="Page 1 Changes", level=2, start_page=10)
    parent = TocNode(
        title="Revision History",
        level=1,
        start_page=10,
        nodes=[child],
    )
    flag_boilerplate([parent])
    assert parent.boilerplate_category == "revision"
    assert child.boilerplate_category == "revision"


def test_flag_boilerplate_children_classified_independently():
    """Children of non-boilerplate parents are classified on their own merits."""
    child_boilerplate = TocNode(title="Revision History", level=2, start_page=70)
    child_substantive = TocNode(title="DC Specifications", level=2, start_page=10)
    parent = TocNode(
        title="Electrical Characteristics",
        level=1,
        start_page=10,
        nodes=[child_substantive, child_boilerplate],
    )
    flag_boilerplate([parent])
    assert parent.boilerplate_category == ""
    assert child_substantive.boilerplate_category == ""
    assert child_boilerplate.boilerplate_category == "revision"


def test_flag_boilerplate_to_dict_round_trip():
    """Boilerplate category should serialize through to_dict."""
    node = TocNode(title="Disclaimer", level=1, start_page=1, end_page=2)
    flag_boilerplate([node])
    d = node.to_dict()
    assert d["boilerplate_category"] == "legal"


def test_flag_boilerplate_child_with_own_category_wins_over_parent():
    """Cross-category: a `glossary` child under a `revision` parent stays
    `glossary`, not `revision`."""
    child = TocNode(title="Glossary", level=2, start_page=12)
    parent = TocNode(title="Revision History", level=1, start_page=10, nodes=[child])
    flag_boilerplate([parent])
    assert parent.boilerplate_category == "revision"
    assert child.boilerplate_category == "glossary"


def test_flag_boilerplate_deep_inheritance():
    """Three-level inheritance under a true boilerplate parent."""
    leaf = TocNode(title="Detail line", level=3, start_page=12)
    middle = TocNode(title="Sub Heading", level=2, start_page=11, nodes=[leaf])
    top = TocNode(title="Revision History", level=1, start_page=10, nodes=[middle])
    flag_boilerplate([top])
    assert top.boilerplate_category == "revision"
    assert middle.boilerplate_category == "revision"
    assert leaf.boilerplate_category == "revision"


def test_flag_boilerplate_empty_title_does_not_propagate():
    """An empty-title parent has no own classification and contributes none
    to its children. Children are classified on their own merits."""
    child_a = TocNode(title="Electrical Characteristics", level=2, start_page=2)
    child_b = TocNode(title="Disclaimer", level=2, start_page=3)
    parent = TocNode(title="", level=1, start_page=1, nodes=[child_a, child_b])
    flag_boilerplate([parent])
    assert parent.boilerplate_category == ""
    assert child_a.boilerplate_category == ""
    assert child_b.boilerplate_category == "legal"
