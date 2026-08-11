"""Running header/footer ("page furniture") detection.

Pure functions over strings and counts. This module never touches a
``pymupdf.Page``, reads no environment and does no I/O, so it is testable
without a PDF and cannot reach the process-global layout engine.

The method is a simplified Lin page-association (SPIE 2003): a block is
furniture when the same normalized text recurs, in a page-edge band, on a
large share of the document's pages. ``core/textfile.py`` owns the geometry
half of that decision; this module owns the text and counting half.

See "Running header/footer stripping" under Deliverable 2 in
``docs/datasheetindex_architecture.md`` for the measurements behind every
constant here, the known limits, and the alternatives that were rejected.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence

#: Raw block text longer than this is body prose, never furniture. This is the
#: ONLY size guard. There is deliberately no line-count rule: PyMuPDF's
#: ``get_text("blocks")`` groups a whole footer into one block of several short
#: lines -- the PSoC 6 footer is a single 4-line, 41-character block -- so
#: excluding multi-line blocks discards real footers. Measured across seven
#: documents, a ">= 3 lines" rule missed genuine footers on five of them.
MAX_FURNITURE_CHARS = 200

#: Share of a document's pages a key must appear on to count as furniture.
#: Measured: real furniture recurs on 52-100% of pages; the two values below
#: 92% are both on one document. Lowering this to 0.33 starts deleting running
#: section headings ("6 Electrical specifications" on 47 of the PSoC's 134
#: pages), so 0.5 is the last value at which the survey corpus stays clean.
PAGE_FRACTION = 0.5

#: Absolute floor on the page count, so a 1- or 2-page document can never
#: produce furniture. With no recurrence evidence, keeping the text is the
#: honest answer.
MIN_PAGES = 3

_WHITESPACE_RE = re.compile(r"\s+")
_DIGIT_RUN_RE = re.compile(r"\d+")

#: A block opening with a caption keyword is content, even when it recurs.
#: Insurance rather than the active mechanism: block granularity already keeps
#: ``Table N (continued)`` captions out of the page-edge band, and ablating this
#: rule changes nothing on either bundled PDF. It earns its place anyway because
#: ``figures.caption_entries`` reads the stripped text, so a caption that did
#: land in the band would vanish from the figure index, and those captions are
#: what ``TocNode.continued_tables`` is built from.
_CAPTION_PREFIX_RE = re.compile(r"(?i)^(figure\b|fig\.|table\b|chart\b)")


def normalize_key(text: str) -> str:
    """Collapse whitespace and mask digit runs, giving a cross-page key.

    ``002-23185 Rev. *S | 2025-11-06`` becomes ``#-# Rev. *S | #-#-#``, so a
    revision line and a page number match across pages while the letters
    still have to agree. Deliberately no fuzzy matching: a similarity
    threshold can delete a genuine one-off line that resembles its
    neighbours, and missing furniture is the safer failure.
    """
    collapsed = _WHITESPACE_RE.sub(" ", text).strip()
    return _DIGIT_RUN_RE.sub("#", collapsed)


def is_candidate(text: str) -> bool:
    """Whether a block's text is eligible to be furniture at all."""
    stripped = text.strip()
    if not stripped:
        return False
    if len(stripped) > MAX_FURNITURE_CHARS:
        return False
    return _CAPTION_PREFIX_RE.match(stripped) is None


def furniture_threshold(total_pages: int) -> int:
    """Pages a key must appear on before it counts as furniture."""
    return max(MIN_PAGES, math.ceil(PAGE_FRACTION * total_pages))


def has_lexical_evidence(key: str) -> bool:
    """Whether a normalized key carries any letters at all.

    A key with no alphabetic character is pure punctuation and masked digits
    -- ``#``, or ``#.# #.#`` -- so recurrence is the *only* thing said about
    it, and numeric table content matches it exactly. A bare page-number
    footer is an extremely common datasheet layout, and it makes ``#``
    furniture; every bare-number block in either band is then deleted
    document-wide. Reproduced on a 10-page synthetic: ``120``, ``127`` and
    ``3.3 4.3`` were removed from real table rows. No other guard helps --
    such blocks are short, in-band and carry no caption keyword.

    The cost is that a footer consisting *only* of a page number is no longer
    stripped, which is the design's stated safe direction: a miss, not a
    deletion. It costs nothing measurable -- zero furniture keys on both
    bundled PDFs and on all 16 keys of the seven-document survey, since a
    real footer carries a document title, a revision or a URL alongside its
    numbers (``# V#.# #-#-#`` still qualifies).

    ``str.isalpha`` rather than an ``[A-Za-z]`` class so a non-Latin script
    counts as evidence too.
    """
    return any(char.isalpha() for char in key)


def detect_furniture(
    page_keys: Sequence[Iterable[str]], total_pages: int
) -> frozenset[str]:
    """Return the keys that recur on enough pages to be furniture.

    ``page_keys`` is one iterable of normalized keys per page. Each key is
    counted once per page whatever the caller passes, so a header repeated
    twice on one page does not count double.

    A key with no letters is never returned, whatever its count -- see
    ``has_lexical_evidence``.
    """
    counts: dict[str, int] = {}
    for keys in page_keys:
        for key in set(keys):
            counts[key] = counts.get(key, 0) + 1
    threshold = furniture_threshold(total_pages)
    return frozenset(
        key
        for key, seen in counts.items()
        if seen >= threshold and has_lexical_evidence(key)
    )
