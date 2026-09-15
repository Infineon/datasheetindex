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
        # TI writes the orderable part-number addendum and the package
        # drawings under one compound heading. It classifies as `ordering`,
        # not `mechanical`, because the addendum is the per-variant table:
        # `flag_boilerplate` must be able to suppress it on a family
        # datasheet. Variant-evidence ranking is deliberately narrower because
        # this appendix does not carry feature differences on every TI family.
        ("Mechanical, Packaging, and Orderable Information", "ordering"),
        ("Orderable Information", "ordering"),
        ("Package Option Addendum", "ordering"),
        # The marking legend maps package markings back to part numbers, so it
        # is per-part identifying information and must ride `ordering` -- the
        # only category multi_variant lifts. Sibling of the existing
        # "Device Marking" branch; Microchip spells it this way.
        ("Package Marking Information", "ordering"),
        ("Package Marking", "ordering"),
        # mechanical -- package drawings and dimensions, no part numbers
        ("Packaging Information", "mechanical"),
        ("Packaging", "mechanical"),
        ("Package Outline", "mechanical"),
        ("Package Dimensions", "mechanical"),
        ("Package Drawings", "mechanical"),
        ("Mechanical Data", "mechanical"),
        ("Mechanical Drawings", "mechanical"),
        ("Tape and Reel Information", "mechanical"),
        ("Thermal Pad Mechanical Data", "mechanical"),
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
        # `mechanical` regression, measured on the corpus. Raspberry Pi puts
        # "2.3. Recommended operating conditions" under "2. Mechanical
        # specification", and RP2040's "Chapter 5. Electrical and Mechanical"
        # is the electrical chapter outright. Flagging either would make a
        # core parameter section inherit a deprioritize hint.
        "Mechanical specification",
        "Mechanical Specifications",
        "Electrical and Mechanical",
        "Mechanical and Electrical Characteristics",
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
        # A "(continued)" suffix marks the same section resumed on a later
        # page and must not change what it is. Microchip repeats the heading
        # this way -- micro_pic16f887 carries "19.1 Package Marking
        # Information (Continued)" twice -- and an anchored pattern misses
        # every one of them, so the continuation inherits its parent instead.
        ("19.1 Package Marking Information (Continued)", "ordering"),
        ("Revision History (continued)", "revision"),
        ("Ordering Information (Continued)", "ordering"),
        ("Packaging Information (continued)", "mechanical"),
        # Punctuation *after* the closing paren must not defeat the strip:
        # `_CONTINUATION_RE` is anchored on the end of the string, so a
        # trailing period or colon -- ordinary in vendor headings -- used to
        # leave "(continued" in place and drop the title back to unclassified.
        ("Ordering Information (Continued).", "ordering"),
        ("Ordering Information (Continued):", "ordering"),
        ("Revision History (continued) -", "revision"),
        # The apostrophe spellings, which the comment claims and the pattern
        # did not match. Both the straight and the typographic apostrophe.
        ("Package Marking Information (Cont'd)", "ordering"),
        ("Package Marking Information (Cont\u2019d)", "ordering"),
        ("Revision History (cont.)", "revision"),
        # Microchip's per-part table. The existing `product identification`
        # branch is anchored, so the trailing "System" defeated it.
        ("Product Identification System", "ordering"),
        ("Product Identification", "ordering"),
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


class TestOrderingOnMultiVariantDatasheets:
    """The ordering section is authoritative when the document covers a family.

    ``boilerplate_category`` exists to tell an agent what to *deprioritize*.
    On a multi-variant datasheet the per-part table lives in the ordering
    section, so flagging it steers the agent away from the only section that
    can answer a per-part question -- the observed failure this guards.
    """

    def test_ordering_is_flagged_on_a_single_part_datasheet(self):
        nodes = [TocNode(title="Ordering Information", level=1, start_page=9)]
        flag_boilerplate(nodes, multi_variant=False)
        assert nodes[0].boilerplate_category == "ordering"

    def test_ordering_is_not_flagged_on_a_multi_variant_datasheet(self):
        nodes = [TocNode(title="Ordering Information", level=1, start_page=69)]
        flag_boilerplate(nodes, multi_variant=True)
        assert nodes[0].boilerplate_category == ""

    def test_other_categories_still_flagged_on_a_multi_variant_datasheet(self):
        """Only `ordering` is authoritative. Legal text is boilerplate either way."""
        nodes = [
            TocNode(title="Disclaimer", level=1, start_page=70),
            TocNode(title="Revision History", level=1, start_page=71),
        ]
        flag_boilerplate(nodes, multi_variant=True)
        assert nodes[0].boilerplate_category == "legal"
        assert nodes[1].boilerplate_category == "revision"

    def test_children_do_not_inherit_a_suppressed_ordering_flag(self):
        """Suppression must reach subsections, which is where the tables are."""
        nodes = [
            TocNode(
                title="Ordering Information",
                level=1,
                start_page=69,
                nodes=[TocNode(title="Part Number Matrix", level=2, start_page=70)],
            )
        ]
        flag_boilerplate(nodes, multi_variant=True)
        assert nodes[0].nodes[0].boilerplate_category == ""

    def test_default_preserves_existing_behaviour(self):
        """Callers that pass no flag must be unaffected."""
        nodes = [TocNode(title="Ordering Information", level=1, start_page=9)]
        flag_boilerplate(nodes)
        assert nodes[0].boilerplate_category == "ordering"


class TestMechanicalCategory:
    """Package drawings are boilerplate on every datasheet, family or not.

    Split from `ordering` because the two are read for different questions:
    `ordering` answers "which part number", `mechanical` answers "what are the
    package dimensions". Only `ordering` is per-variant authoritative, so only
    `ordering` is suppressed on a family datasheet.
    """

    def test_mechanical_is_flagged_on_a_single_part_datasheet(self):
        nodes = [TocNode(title="11 Packaging Information", level=1, start_page=30)]
        flag_boilerplate(nodes, multi_variant=False)
        assert nodes[0].boilerplate_category == "mechanical"

    def test_mechanical_stays_flagged_on_a_multi_variant_datasheet(self):
        """Drawings are not the per-part table; the addendum is, and that is
        `ordering`."""
        nodes = [TocNode(title="11 Packaging Information", level=1, start_page=30)]
        flag_boilerplate(nodes, multi_variant=True)
        assert nodes[0].boilerplate_category == "mechanical"

    def test_the_ti_compound_section_is_suppressed_on_a_family_datasheet(self):
        """The observed TI shape: one chapter carrying both, 12 of 24 corpus
        documents. Suppression must reach it, or the orderable addendum is
        flagged skippable on exactly the datasheets that need it."""
        nodes = [
            TocNode(
                title="11 Mechanical, Packaging, and Orderable Information",
                level=1,
                start_page=31,
                nodes=[
                    TocNode(
                        title="11.1 Package Option Addendum", level=2, start_page=32
                    )
                ],
            )
        ]
        flag_boilerplate(nodes, multi_variant=True)
        assert nodes[0].boilerplate_category == ""
        assert nodes[0].nodes[0].boilerplate_category == ""

    def test_the_ti_compound_section_is_flagged_on_a_single_part_datasheet(self):
        nodes = [
            TocNode(
                title="11 Mechanical, Packaging, and Orderable Information",
                level=1,
                start_page=31,
            )
        ]
        flag_boilerplate(nodes, multi_variant=False)
        assert nodes[0].boilerplate_category == "ordering"

    def test_a_classified_subsection_keeps_its_own_category(self):
        """A node's own classification wins over the parent's, as for every
        other category."""
        nodes = [
            TocNode(
                title="11 Packaging Information",
                level=1,
                start_page=30,
                nodes=[TocNode(title="11.1 Revision History", level=2, start_page=34)],
            )
        ]
        flag_boilerplate(nodes)
        assert nodes[0].nodes[0].boilerplate_category == "revision"

    def test_an_unclassified_subsection_inherits_mechanical(self):
        """Inheritance is the documented design, and under a packaging chapter
        it is right: the unlabelled subsections there are drawings. Pinned so
        the behaviour is a decision rather than an accident -- measured across
        the corpus, every inheriting child is a genuine drawing section
        ("Package dimensions", "Package Details", the MA/PN package codes)."""
        nodes = [
            TocNode(
                title="11 Packaging Information",
                level=1,
                start_page=30,
                nodes=[TocNode(title="11.2 Package Details", level=2, start_page=32)],
            )
        ]
        flag_boilerplate(nodes)
        assert nodes[0].nodes[0].boilerplate_category == "mechanical"

    def test_the_marking_legend_is_lifted_on_a_family_datasheet(self):
        """The one inheriting child that is not a drawing. On micro_pic16f887
        -- a family part -- "19.1 Package Marking Information" sits under
        "19.0 Packaging Information" and used to inherit `mechanical`, which
        multi_variant does not lift, so the legend mapping markings to variants
        carried a deprioritize hint on exactly the datasheet that needs it."""
        nodes = [
            TocNode(
                title="19.0 Packaging Information",
                level=1,
                start_page=298,
                nodes=[
                    TocNode(
                        title="19.1 Package Marking Information",
                        level=2,
                        start_page=298,
                    ),
                    TocNode(title="19.2 Package Details", level=2, start_page=301),
                ],
            )
        ]
        flag_boilerplate(nodes, multi_variant=True)
        assert nodes[0].boilerplate_category == "mechanical"
        assert nodes[0].nodes[0].boilerplate_category == ""
        assert nodes[0].nodes[1].boilerplate_category == "mechanical"


def test_suppressed_ordering_does_not_inherit_its_parent_category():
    """Suppression must not fall through to the parent's category.

    Clearing the node's own classification and then letting it inherit puts
    the flag straight back on, re-creating the exact mis-steer the
    suppression exists to remove -- just with a different category name.
    """
    nodes = [
        TocNode(
            title="Appendix A: Revision History",
            level=1,
            start_page=60,
            nodes=[TocNode(title="Ordering Information", level=2, start_page=61)],
        )
    ]
    flag_boilerplate(nodes, multi_variant=True)
    assert nodes[0].nodes[0].boilerplate_category == ""
