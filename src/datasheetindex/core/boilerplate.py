"""Boilerplate detection for ToC nodes.

Flags sections whose title matches well-known datasheet boilerplate so that
agents can deprioritize them during navigation. Detection is intentionally
title-only and pattern-based -- no LLM call, no text scanning -- because
~80% of datasheet boilerplate is title-detectable and we want this to stay
free in the happy path.

Categories:
    legal     -- disclaimers, important notices, trademarks, copyright, patents
    ordering  -- ordering info, part numbers, marking information
    revision  -- revision/change/document history
    contact   -- sales offices, support contacts, "where to buy"
    toc       -- table of contents, list of figures/tables, index
    glossary  -- glossary, abbreviations, acronyms, terminology

Scope: English titles only. Non-ASCII headings (e.g. "免責事項", "Mentions
légales") are intentionally not classified -- adding multilingual coverage
without measurable false-positive rates from real-world non-English datasheets
would invite regressions.
"""

from __future__ import annotations

import re

from datasheetindex.models import TocNode

# Strip leading section numbering / prefixes before matching.
# Matches "Appendix A:", "Chapter 1.", "12.3.4 ", "A. " before the real title.
# A bare single capital letter must be followed by punctuation (`.`, `:`, `)`,
# `-`), never bare whitespace -- otherwise "A Glossary of Terms" would have
# its leading "A" stripped and then misclassify as `glossary`.
_LEADING_PREFIX_RE = re.compile(
    r"""
    ^\s*
    (?:
        (?:chapter|section|appendix|annex)\s+[A-Za-z0-9]+ [\s:.\)\-]+
      | [0-9]+(?:\.[0-9]+)* [\s:.\)\-]+
      | [A-Z] [:.\)\-]+ \s*
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Each pattern matches the *normalized* title (lowercased, prefix stripped,
# trailing punctuation removed). Anchored on both ends to avoid matching
# substantive sections that merely mention a keyword (e.g. "Trademark Licensing
# Strategy" is not "Trademarks").
_BOILERPLATE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "legal",
        re.compile(
            # Each branch must be unambiguous: bare `information` / `notice` /
            # `liability` are common substantive titles, so they require an
            # explicit qualifier (`legal`, `important`, `product`, ...).
            r"^("
            r"disclaimers?"
            r"|legal\s+(disclaimer|notices?|information)"
            r"|important\s+(notices?|information|notes?)"
            r"|terms?\s+(and|&)\s+conditions?"
            r"|safety\s+(precautions?|guidelines?|notices?|information|warnings?)"
            r"|trademarks?(\s+(notice|acknowledgments?|information))?"
            r"|copyrights?(\s+notice)?"
            r"|patents?(\s+notice)?"
            r"|product\s+liability"
            r"|limitations?\s+of\s+liability"
            r"|warranty(\s+disclaimer)?"
            r"|export\s+control"
            r"|esd\s+(caution|warning|notice)"
            r")$"
        ),
    ),
    (
        "ordering",
        re.compile(
            r"^("
            r"ordering\s+(information|guide|details?|codes?)"
            r"|order(ing)?\s+(information|number|numbers)"
            r"|part\s+(number|numbers|numbering)(\s+information)?"
            r"|marking\s+(information|codes?)"
            r"|product\s+(identification|marking|naming)"
            r"|device\s+(marking|ordering)"
            r"|how\s+to\s+order"
            r")$"
        ),
    ),
    (
        "revision",
        re.compile(
            r"^("
            r"(revision|document|change|version)\s+(history|record|records|log|control|status)"
            r"|revisions?"
            r"|change\s+log"
            r"|history\s+of\s+(revisions?|changes?)"
            r")$"
        ),
    ),
    (
        "contact",
        re.compile(
            r"^("
            r"(contact|sales|support)\s+(information|us|offices?|contacts?)"
            r"|customer\s+(information|us|offices?|contacts?|support|service|care)"
            r"|technical\s+support"
            r"|worldwide\s+(sales|offices|support)"
            r"|where\s+to\s+(buy|contact|get\s+help)"
            r"|regional\s+(sales|offices)"
            r")$"
        ),
    ),
    (
        "toc",
        re.compile(
            r"^("
            r"(table\s+of\s+)?contents?"
            r"|list\s+of\s+(figures?|tables?|illustrations?|equations?)"
            r"|index"
            r")$"
        ),
    ),
    (
        "glossary",
        re.compile(
            r"^("
            r"glossary"
            r"|abbreviations?(\s+(and|&)\s+acronyms?)?"
            r"|acronyms?(\s+(and|&)\s+abbreviations?)?"
            r"|terminology"
            r"|definitions?"
            r"|nomenclature"
            r")$"
        ),
    ),
]

# Vocabulary that signals legal boilerplate in *running prose*. Used by the
# front-matter `legal_hits` signal, and deliberately NOT shared with the
# `legal` branch of `_BOILERPLATE_PATTERNS` above, which serves a different
# question. That pattern is anchored to a whole title and several of its
# branches require a qualifier (`product liability`, `important notices`),
# because bare `liability` and `information` are common substantive section
# titles. In prose the judgement inverts: a bare "liability" in a footer
# sentence is exactly the signal. One list cannot serve both without either
# weakening the title matcher -- which publishes a flag in the artifact -- or
# under-counting prose. The partial duplication is the cheaper failure.
_LEGAL_VOCABULARY: tuple[str, ...] = (
    r"disclaimers?",
    r"warrant(?:y|ies)",
    r"liabilit(?:y|ies)",
    r"liable",
    r"trademarks?",
    r"copyrights?",
    r"patents?",
    r"indemnif\w*",
    r"terms\s+and\s+conditions",
    r"limitations?\s+of\s+liability",
    r"export\s+control",
    r"subject\s+to\s+change\s+without\s+notice",
    r"no\s+license",
    r"as\s+is",
    r"at\s+your\s+own\s+risk",
)

_LEGAL_PROSE_RE = re.compile(
    r"\b(?:" + "|".join(_LEGAL_VOCABULARY) + r")\b", re.IGNORECASE
)


def _normalize_title(title: str) -> str:
    """Strip leading numbering/prefixes and trailing punctuation, lowercase."""
    s = title.strip()
    # Strip leading section/chapter/number prefixes (may have several layers,
    # e.g. "Appendix A: 1. Ordering" — strip repeatedly).
    while True:
        new = _LEADING_PREFIX_RE.sub("", s, count=1)
        if new == s:
            break
        s = new
    s = s.strip(" \t:.,-)")
    return s.lower()


def classify_title(title: str) -> str:
    """Return the boilerplate category for a title, or ``""`` if none matches."""
    normalized = _normalize_title(title)
    if not normalized:
        return ""
    for category, pattern in _BOILERPLATE_PATTERNS:
        if pattern.match(normalized):
            return category
    return ""


def flag_boilerplate(nodes: list[TocNode]) -> list[TocNode]:
    """Recursively set ``boilerplate_category`` on each node.

    Classification rules:
    - A node's own title classification always wins. So a substantive
      subsection like "Electrical Characteristics" under a misclassified
      "Information" parent stays unflagged, instead of inheriting `legal`.
    - A node with no own classification inherits its parent's category,
      on the principle that unlabelled subsections of a "Revision History"
      appendix are themselves revision-history content.
    - Top-level nodes with no own classification stay empty.

    Modifies nodes in-place and returns them for convenience.
    """
    _flag_recursive(nodes, parent_category="")
    return nodes


def _flag_recursive(nodes: list[TocNode], parent_category: str) -> None:
    for node in nodes:
        own = classify_title(node.title)
        if own:
            node.boilerplate_category = own
        elif parent_category:
            node.boilerplate_category = parent_category
        else:
            node.boilerplate_category = ""
        if node.nodes:
            _flag_recursive(node.nodes, node.boilerplate_category)


def count_legal_hits(text: str) -> int:
    """Count legal-boilerplate vocabulary matches in running prose.

    A count, not a verdict: a cover letter scores high and a features page
    scores zero, but what to do about that is the caller's decision.
    """
    return len(_LEGAL_PROSE_RE.findall(text))
