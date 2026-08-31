"""Detection of multi-variant datasheets from the document's title block.

A *multi-variant* datasheet covers more than one part number whose features
differ, so body text describing "the device" may not describe the part the
user asked about. The failure this guards against is specific and observed:
an agent asked whether one part has a peripheral, answering "yes" from a
family-level features section, when the per-part table said no for that part.

Detection is deliberately **title-only**. The title is the one place a vendor
*curates* the document's scope -- writing ``ADS111x`` or ``LM111, LM211,
LM311`` precisely when the document covers a family. Page-1 body text was
measured as an alternative and rejected: it fired on 22 of 57 otherwise-missed
documents but recovered only 2 genuine families, because package order codes
(``TXB0104RGY``), companion parts (``CC1190``) and tokens that are not part
numbers at all (``RGB888``, ``PT100``, pin names like ``AIN0P``) are
indistinguishable from feature variants without vendor-specific grammars.
Precision fell from 1.00 to ~0.41 for a gain of 2 documents in 115.

Measured against hand-assigned ground truth on a 25-document corpus spanning
6 vendors: **precision 1.00, recall 0.85** (11 of 13 families found, 0 false
positives in 12 single-part documents). An independent 115-document sample
fetched from one vendor is not hand-labelled, so it yields no recall figure,
but its firing rate (52%) matches the corpus base rate (52%) -- evidence the
rate is not an artifact of corpus selection.

Precision is the property that matters. The flag suppresses a boilerplate hint
and adds a caution to tool output, so a false positive costs noise while a
false negative costs a confident wrong answer. That asymmetry is why the rules
below are all conservative, and why a miss must degrade to the always-on
caution in the tool descriptions rather than to silence.

The residual 15% is not reachable by a better rule. Both corpus misses show
why: one is a family technical reference manual whose title names no part at
all, and one is an ``LM158/LM258/LM358`` datasheet whose title is the
descriptor alone. Separating ``ADS1255`` (a sibling) from ``TXB0104RGY`` (an
order code) needs world knowledge. That judgment belongs to the agent, which
holds the actual question; the library supplies the cheap, certain half.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pymupdf

logger = logging.getLogger(__name__)

# A part-number-shaped token: contains a digit AND a letter, 4-18 chars.
# The digit requirement rejects ordinary words; the letter requirement rejects
# bare page and table numbers.
_PART_TOKEN = re.compile(
    r"\b(?=[A-Za-z0-9\-]*[0-9])(?=[A-Za-z0-9\-]*[A-Za-z])"
    r"[A-Za-z0-9][A-Za-z0-9\-]{3,17}\b"
)

# "PIC16F882/883/887", "OPA340/2340": one part token continued by bare numbers.
# Checked before _PART_TOKEN pairs because the continuations are not themselves
# part-shaped (they carry no letter).
_SLASH_LIST = re.compile(
    r"\b(?=[A-Za-z0-9]*[0-9])[A-Za-z][A-Za-z0-9]{2,16}(?:/[0-9]{2,5}){1,8}\b"
)

# "ADS111x", "MSP430F552x": a lowercase x standing in for the varying digit.
# Vendors use this in a title only to denote a family.
_WILDCARD = re.compile(
    r"\b(?=[A-Za-z0-9\-]*[0-9])[A-Za-z][A-Za-z0-9\-]*x[A-Za-z0-9\-]*\b"
)

_SERIES = re.compile(r"\b(series|family)\b", re.IGNORECASE)

# The agent is shown ``family`` verbatim inside a note, so it is bounded.
_MAX_FAMILY_CHARS = 120


@dataclass(frozen=True)
class VariantSignal:
    """Evidence that a datasheet covers a product family.

    ``family`` is the matched text, shown to the agent verbatim. ``rule``
    names which pattern fired, so a surprising flag can be traced to its
    cause without re-running the detector.
    """

    family: str
    rule: str


def _bounded(parts: list[str]) -> str:
    """Join matched tokens, de-duplicated in order, within the char budget."""
    seen: list[str] = []
    for part in parts:
        if part not in seen:
            seen.append(part)
    joined = ", ".join(seen)
    if len(joined) <= _MAX_FAMILY_CHARS:
        return joined
    return joined[: _MAX_FAMILY_CHARS - 3].rstrip(", ") + "..."


def _near_identical(a: str, b: str) -> bool:
    """Same length and leading letters, differing in 1-2 positions.

    Catches the grade families a vendor names in full -- LM111/LM211/LM311,
    SN54HC590A/SN74HC590A -- which differ at the FIRST digit, where a
    shared-prefix test sees two unrelated tokens.
    """
    if len(a) != len(b) or a == b:
        return False
    if a[:2] != b[:2]:
        return False
    # Lengths are equal by the guard above, so strict= changes nothing here.
    diff = sum(1 for x, y in zip(a, b, strict=True) if x != y)
    return 1 <= diff <= 2


def detect_variants(title: str) -> VariantSignal | None:
    """Return a signal if ``title`` states a product family, else ``None``.

    Rules are ordered by how specific their evidence is, so ``rule`` names the
    strongest thing found rather than the first thing checked.
    """
    if not title or not title.strip():
        return None

    slash = _SLASH_LIST.findall(title)
    if slash:
        return VariantSignal(family=_bounded(slash), rule="slash-list")

    wildcard = _WILDCARD.findall(title)
    if wildcard:
        return VariantSignal(family=_bounded(wildcard), rule="wildcard")

    tokens = _PART_TOKEN.findall(title)
    for i, a in enumerate(tokens):
        for b in tokens[i + 1 :]:
            if a != b and a[:3] == b[:3]:
                return VariantSignal(family=_bounded([a, b]), rule="list")
            if _near_identical(a, b):
                return VariantSignal(family=_bounded([a, b]), rule="near-identical")

    if tokens and _SERIES.search(title):
        return VariantSignal(family=_bounded(tokens), rule="series")

    return None


# Spans within this many points of the largest are part of the title block.
# A datasheet title is often set in two sizes (part number, then descriptor),
# and the gap to body text is far larger than the gap between them.
_TITLE_SIZE_SLACK = 1.5

# Enough for a part-number list plus a descriptor; past that a page is
# title-set throughout and the extra spans are not title.
_MAX_TITLE_SPANS = 8


def title_text(doc: pymupdf.Document) -> str:
    """Return the document's title text: page-1 largest-font block + metadata.

    Both sources are read because they miss different documents -- measured
    recall for the stated family is 85% from the page-1 block and 62% from
    metadata, and neither is a superset of the other.
    """
    parts: list[str] = []

    meta = (doc.metadata or {}).get("title") or ""
    if meta.strip():
        parts.append(" ".join(meta.split()))

    if len(doc) > 0:
        spans: list[tuple[float, str]] = []
        try:
            for block in doc[0].get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = " ".join(span["text"].split())
                        if text:
                            spans.append((round(span["size"], 1), text))
        except Exception:
            # Advisory signal: a page that will not render costs the caution,
            # never the build. Metadata read above is kept -- losing one
            # source must not lose the other.
            logger.debug("Could not read page 1 for the title block", exc_info=True)
            spans = []
        if spans:
            top = max(size for size, _ in spans)
            biggest = [text for size, text in spans if size >= top - _TITLE_SIZE_SLACK]
            parts.append(" ".join(biggest[:_MAX_TITLE_SPANS]))

    return " ".join(p for p in parts if p).strip()
