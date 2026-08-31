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

**Recall is the property that matters, and the asymmetry is measured.** This
paragraph used to lead with the opposite, and that error cost four rounds of
review: every guard added to raise precision over-reached and killed real
families -- a casing rule took ATmega and nRF, a leading-token rule took every
title led by a filename or document number, and vocabulary entries for `usb`
and `rs` took the Microchip hub and Recom converter families.

The numbers. A **false positive** was measured directly: forcing the flag onto
a genuinely single-part datasheet and asking a plainly answerable question gave
6/6 correct answers at 4.3 turns, against 6/6 at 4.2 turns with no flag. It
costs a tenth of a turn. A **false negative** returns the document to the
unmodified library's behaviour, which answered the motivating per-part question
correctly 0 times in 9.

So the rules below are conservative about *rejecting*, not about firing, and
two consequences follow for anyone maintaining them:

- An exclusion that plausibly doubles as a vendor prefix is not worth its
  precision. Prefer the noisy answer.
- A miss must still degrade to the always-on caution in the tool descriptions
  rather than to silence, since that caution is all a missed document gets.

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
#
# The negative lookahead rejects document metadata, which has the identical
# shape: "Rev1/2020", "Ver2/2023", "Doc3/2019". Those routinely sit in a title
# block, and without this a single-part datasheet is flagged as a family whose
# name is a revision string -- printed verbatim to the agent in the read-time
# note.
_SLASH_LIST = re.compile(
    r"\b(?!(?:rev|ver|vers|version|doc|rel|iss|issue|draft)[0-9])"
    r"(?=[A-Za-z0-9]*[0-9])[A-Za-z][A-Za-z0-9]{2,16}(?:/[0-9]{2,5}){1,8}\b",
    re.IGNORECASE,
)

# "ADS111x", "MSP430F552x": a lowercase x standing in for the varying digit.
# Vendors use this in a title only to denote a family.
#
# The pattern alone is far too broad -- it matches any token holding a digit
# and a letter x, so "Cortex-M4" and the lowercase filename "max31855" both
# satisfy it. `_is_wildcard_token` applies the missing half of the rule.
_WILDCARD = re.compile(
    r"\b(?=[A-Za-z0-9\-]*[0-9])[A-Za-z][A-Za-z0-9\-]*x[A-Za-z0-9\-]*\b"
)

# "ESP32 Series". Anchored to the token it qualifies by `_series_family` --
# unanchored, "SOT23 Series Current Monitor" names a package as the family.
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
    # Drop whole entries, never characters. The agent is shown this string
    # inside an instruction telling it to trust the names, so a cut mid-token
    # would present a part number that does not exist.
    kept: list[str] = []
    used = 0
    for part in seen:
        cost = len(part) + (2 if kept else 0)
        if used + cost > _MAX_FAMILY_CHARS - 5:
            # Skip it, do not abandon the list: a later, shorter entry may
            # still fit, and dropping it would shrink the family for no
            # reason.
            continue
        kept.append(part)
        used += cost
    if not kept:
        # Not even one entry fits. Naming no part beats naming half of one:
        # `_variant_note` omits the parenthetical when this is empty, so the
        # note still says the datasheet covers a family, which is true, and
        # never presents a part-number prefix that identifies nothing.
        return ""
    return ", ".join(kept) + ", ..."


# Below this, a token sharing a 3-character prefix with another differs only in
# its last character or two and carries almost no evidence -- "DDR3"/"DDR4",
# "USB2"/"USB3" are bus widths, not a product family. Every family the corpus
# states in full is longer (1N4001, LM111, ADS1113).
_MIN_PAIR_TOKEN_CHARS = 5


def _is_wildcard_token(token: str) -> bool:
    """Whether ``token`` is a part number with ``x`` marking what varies.

    A wildcard token is written in vendor part-number casing -- uppercase
    letters and digits -- with a lowercase ``x`` substituted for the varying
    character: ``ADS111x``, ``OPAx340``, ``SNx4HC595``, ``TLV906xS``,
    ``xx555``, ``PSC3P5xD``. So every lowercase character in it must be an
    ``x``.

    Ordinary words fail that test, which is the point: "Cortex-M4" and
    "32-bit" hold an x or a digit but also other lowercase letters, and the
    lowercase filename "max31855" fails for the same reason. Without this an
    Arm MCU datasheet naming its core in the title -- extremely common -- is
    published as a family called "Cortex-M4".
    """
    return "x" in token and all(c == "x" or not c.islower() for c in token)


def _series_family(title: str, tokens: list[str]) -> str | None:
    """The family named by an "X Series" phrase, if the title carries one.

    The keyword must directly qualify a token, and that token must be the
    title's *principal* part -- the leading one, or a part the leading one
    contains. Two constraints, because each alone gets a real title wrong:

    - Adjacency alone reads "MAX4173 Low-Cost SOT23 Series Current Monitor"
      as a family named after a package code.
    - Leading position alone loses the shape ``title_text`` actually
      produces. It concatenates the PDF metadata title with the page-1
      block, so the family token appears twice and the copy carrying
      "Series" is rarely first: "ESP32 Datasheet ESP32 Series Datasheet",
      or "Infineon-XMC1400-DataSheet-v01_02-EN XMC1400 Series Datasheet",
      whose leading token is the filename. Containment covers both, and
      also the order-code case ("ADS1115IDGSR ADS1115 Series").
    """
    if not tokens:
        return None
    lead = tokens[0].casefold()
    for token in tokens:
        folded = token.casefold()
        if folded != lead and folded not in lead:
            continue
        if _is_non_part(token):
            # "Arm Cortex-M4 Series Technical Reference Manual" otherwise
            # publishes the core as the family.
            continue
        # Every occurrence, not just the first: the concatenated title
        # repeats the token, and it is the later copy that carries "Series".
        for match in re.finditer(re.escape(token), title, re.IGNORECASE):
            after = title[match.end() :].lstrip(" \t,-")
            if _SERIES.match(after):
                return token
    return None


def _is_part_pair(a: str, b: str) -> bool:
    """Whether two prefix-sharing tokens are two parts rather than one.

    Rejects the two ways this rule misfires on a real title block:

    - **One token contains the other.** ``TPS7A4901DGNR`` beside
      ``TPS7A4901`` is a part and its own order code, not two parts --
      and the pair co-occurs routinely, because ``title_text``
      concatenates the PDF metadata title with the page-1 block. Package
      and temperature suffixes are the one thing this detector must never
      read as a feature family; they share one die and one feature set.
    - **Both tokens are too short to be evidence.** See the constant above.
    """
    if _is_non_part(a) or _is_non_part(b):
        return False
    lower_a, lower_b = a.casefold(), b.casefold()
    if lower_a.startswith(lower_b) or lower_b.startswith(lower_a):
        return False
    return min(len(a), len(b)) >= _MIN_PAIR_TOKEN_CHARS


def _near_identical(a: str, b: str) -> bool:
    """Same length and leading letters, differing in 1-2 positions.

    Catches the grade families a vendor names in full -- LM111/LM211/LM311,
    SN54HC590A/SN74HC590A -- which differ at the FIRST digit, where a
    shared-prefix test sees two unrelated tokens.
    """
    if _is_non_part(a) or _is_non_part(b):
        return False
    a, b = a.casefold(), b.casefold()
    if len(a) != len(b) or a == b:
        return False
    if len(a) < _MIN_PAIR_TOKEN_CHARS:
        # Same threshold as the prefix rule, for the same reason: "DDR3"/"DDR4"
        # and "USB2"/"USB3" satisfy every structural test this applies.
        return False
    if a[:2] != b[:2]:
        return False
    # Lengths are equal by the guard above, so strict= changes nothing here.
    diff = sum(1 for x, y in zip(a, b, strict=True) if x != y)
    return 1 <= diff <= 2


# Vocabulary that takes a part number's shape -- letters, an optional
# separator, digits -- while naming something else. Two such tokens in one
# title share a prefix, clear the length floor and do not contain each other,
# so every structural test the pair rules apply accepts them.
#
# This list is deliberately dumb and explicit. Two structural proxies were
# tried first and each over-reached: requiring single-case tokens lost the
# ATmega, ATtiny and nRF families, whose vendors write mixed case as house
# style, and requiring the matched group to contain the title's leading part
# token lost every title led by a filename, a descriptor or a document number
# ("SBAS444H ADS1113 ADS1114"). Both are false negatives, which degrade the
# ordering-flag suppression and the read-time note to nothing. Naming the
# vocabulary has neither failure mode: an entry can only ever be wrong about
# the word it names.
#
# Matched against the token's LEADING alphabetic run, so "Cortex-M7" is
# tested as "cortex" and "RV32IMAC" as "rv". Extend it when a real title
# produces a false positive; do not replace it with a structural rule
# without re-running the corpus and the TI title sample.
_NON_PART_PREFIXES = frozenset(
    {
        # CPU cores and instruction sets
        "cortex",
        "neoverse",
        "ethos",
        "mali",
        "xtensa",
        "hexagon",
        "tricore",
        "powerpc",
        "andes",
        "starfive",
        "risc",
        "rv",
        # interface and bus standards. Deliberately short: "usb", "rs",
        # "ddr" and "lpddr" were here and were removed after "usb" was
        # found to kill the real USB2512B/USB2513B/USB2514B hub family and
        # "rs" the Recom RS-2405S/RS-1212D converters. `DDR3`/`USB2`, the
        # shapes they were added for, fall under _MIN_PAIR_TOKEN_CHARS
        # anyway, so they bought nothing and cost real recall. An entry
        # that plausibly doubles as a vendor prefix does not belong here:
        # a false positive was measured at 6/6 correct and +0.1 turns,
        # while a false negative loses the whole benefit for a document.
        "pcie",
        "sata",
        "hdmi",
        "mipi",
        # package families
        "lqfp",
        "tqfp",
        "qfn",
        "vqfn",
        "hvqfn",
        "bga",
        "sot",
        "soic",
        "tssop",
        "msop",
        "dfn",
        "wlcsp",
        "sop",
        "dip",
        "plcc",
        # document furniture
        "figure",
        "table",
        "chapter",
        "section",
        "note",
        "errata",
        "page",
        "item",
        "appendix",
        "annex",
        "grade",
        "class",
        "um",
    }
)


def _is_non_part(token: str) -> bool:
    """Whether ``token`` names something other than a part.

    Compares the token's *leading* alphabetic run -- "Cortex" out of
    "Cortex-M7" -- not every letter in it, which would give "cortexm" and
    match nothing.
    """
    lead = re.match(r"[A-Za-z]+", token)
    return bool(lead) and lead.group().casefold() in _NON_PART_PREFIXES


def _prefix_group(tokens: list[str]) -> list[str]:
    """Every token sharing a 3-character prefix with another, in order."""
    group = [
        a
        for i, a in enumerate(tokens)
        if any(
            a.casefold() != b.casefold()
            and a[:3].casefold() == b[:3].casefold()
            and _is_part_pair(a, b)
            for j, b in enumerate(tokens)
            if i != j
        )
    ]
    return group if len(group) >= 2 else []


def _near_identical_group(tokens: list[str]) -> list[str]:
    """Every token near-identical to another, in order."""
    group = [
        a
        for i, a in enumerate(tokens)
        if any(_near_identical(a, b) for j, b in enumerate(tokens) if i != j)
    ]
    return group if len(group) >= 2 else []


def detect_variants(title: str) -> VariantSignal | None:
    """Return a signal if ``title`` states a product family, else ``None``.

    Rules are ordered by how specific their evidence is, so ``rule`` names the
    strongest thing found rather than the first thing checked.
    """
    if not title or not title.strip():
        return None

    slash = [t for t in _SLASH_LIST.findall(title) if not _is_non_part(t)]
    if slash:
        return VariantSignal(family=_bounded(slash), rule="slash-list")

    wildcard = [t for t in _WILDCARD.findall(title) if _is_wildcard_token(t)]
    if wildcard:
        return VariantSignal(family=_bounded(wildcard), rule="wildcard")

    # Case-insensitively unique, because `title_text` concatenates the PDF
    # metadata title with the page-1 block and vendors routinely set one in
    # full caps. Two casings of one token are one part, not a family.
    tokens: list[str] = []
    seen: set[str] = set()
    for token in _PART_TOKEN.findall(title):
        if token.casefold() not in seen:
            seen.add(token.casefold())
            tokens.append(token)

    # Collect every token that pairs, not just the first two. The family text
    # is the agent's only view of who the caution covers: naming "1N4001,
    # 1N4002" on a document listing 1N4001-1N4007 invites an agent asked about
    # 1N4007 to conclude its part is out of scope.
    # Both groups contribute to the family text, while ``rule`` names the
    # first that fired. A real title can implicate parts through each: in
    # "LM393B, LM2903B, LM193, LM293, LM393 and LM2903", LM2903B pairs with
    # LM293 by shared prefix while LM193 reaches LM393 only by near-identity.
    # Reporting one group alone names a family missing parts the document
    # covers, and an agent asked about an omitted part can read the caution
    # as not applying to it.
    prefix_group = _prefix_group(tokens)
    near_group = _near_identical_group(tokens)
    if prefix_group or near_group:
        rule = "list" if prefix_group else "near-identical"
        ordered = [t for t in tokens if t in prefix_group or t in near_group]
        return VariantSignal(family=_bounded(ordered), rule=rule)

    series = _series_family(title, tokens)
    if series is not None:
        return VariantSignal(family=series, rule="series")

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
