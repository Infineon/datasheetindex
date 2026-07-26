"""Front matter extraction: page-marked pages 1-2 text for agent orientation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    import pymupdf

from datasheetindex.core.boilerplate import count_legal_hits
from datasheetindex.core.textfile import _extract_page_text

#: Pages read from the front of the document.
DEFAULT_MAX_PAGES = 2
#: Document characters emitted. Raised from 2400 in 0.26.0: the old value cut
#: the measured 4746-character front matter of the PSoC 6 in half and kept the
#: half holding the legal footer rather than the specifications. Measured at
#: 5000: no document in the local corpus is cut (2554, 2688 and 4746 chars).
DEFAULT_MAX_CHARS = 5000

# A line opening with a bullet glyph, or with a dash that is *not* followed by
# a digit or a letter. That requirement is load-bearing: datasheets are full of
# lines like "-40 to +85" and a temperature range is not a feature bullet. End
# of line satisfies it, and must, because Infineon's extraction emits most of
# its markers as a dash alone on a line.
_BULLET_RE = re.compile(
    r"^\s*(?:[\u2022\u25aa\u25cb\u25e6\u2023\u00b7\u2219*]|[-\u2013\u2014](?:\s|$))"
)

# A heading, matched as a whole line so that "Features of the analog subsystem"
# does not count.
_FEATURES_HEADINGS = frozenset({"features", "general description"})


class _PageSignals(TypedDict):
    """Return shape of :func:`_page_signals`, typed so callers need no casts."""

    bullets: int
    legal_hits: int
    has_features_heading: bool


@dataclass(frozen=True)
class PreamblePage:
    """Per-page evidence about one front-matter page.

    The signals are reported, never acted on: the library says what it saw and
    the agent decides what to ignore. A cover letter and a real features page
    are separated by these counts, but skipping either is the caller's call.
    """

    page: int
    chars: int
    bullets: int = 0
    legal_hits: int = 0
    has_features_heading: bool = False

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "page": self.page,
            "chars": self.chars,
            "bullets": self.bullets,
            "legal_hits": self.legal_hits,
            "has_features_heading": self.has_features_heading,
        }


@dataclass(frozen=True)
class FrontMatter:
    """The preamble text plus what it cost and what it left out.

    There is deliberately no single ``truncated`` flag. The two caps are
    independent, so one boolean could not tell a caller whether to raise
    ``max_chars``, ``max_pages``, or both. ``char_truncated`` and
    ``pages_omitted`` give the two decisions separately, and the NOTE lines in
    ``text`` are rendered from these same numbers, so prose and fields cannot
    disagree.
    """

    text: str
    pages: list[PreamblePage]
    chars_shown: int
    chars_extracted: int
    pages_read: int
    total_pages: int

    @property
    def char_truncated(self) -> bool:
        """True when ``max_chars`` cut document text that was extracted."""
        return self.chars_shown < self.chars_extracted

    @property
    def pages_omitted(self) -> int:
        """Pages the document holds beyond the ones read."""
        return self.total_pages - self.pages_read


def _truncate_on_line_boundary(text: str, budget: int) -> str:
    """Keep whole lines while they fit in ``budget``. No rstrip."""
    kept: list[str] = []
    total = 0
    for line in text.splitlines(keepends=True):
        if total + len(line) > budget:
            break
        kept.append(line)
        total += len(line)
    return "".join(kept)


def _page_phrase(pages_read: int) -> str:
    """``page 1`` for one page, ``pages 1-P`` otherwise."""
    if pages_read == 1:
        return "page 1"
    return f"pages 1-{pages_read}"


def _char_note(
    *,
    max_chars: int,
    chars_shown: int,
    chars_extracted: int,
    pages_read: int,
    cut_page: int,
) -> str:
    """Name what ``max_chars`` cut.

    ``chars_extracted`` is characters on the pages *read*, never "front matter
    in the document" -- that would require knowing where front matter ends,
    which is exactly what cannot be determined.
    """
    tail = f", ending mid-page on page {cut_page}" if cut_page else ""
    return (
        f"=== NOTE: preamble truncated at {max_chars} characters; "
        f"{chars_shown} of {chars_extracted} characters from "
        f"{_page_phrase(pages_read)} shown{tail} ==="
    )


def _page_note(*, pages_read: int, total_pages: int) -> str:
    """Claim only the mechanical fact: pages exist that were not examined.

    Deliberately does not say front matter continues, or count characters it
    never extracted -- reading one page past the limit to describe what was
    skipped would defeat the limit.
    """
    return (
        f"=== NOTE: preamble covers {_page_phrase(pages_read)} of "
        f"{total_pages}; later pages were not examined ==="
    )


def _page_signals(text: str) -> _PageSignals:
    """Per-page evidence: bullet lines, legal vocabulary, a features heading.

    Reported, not acted on. ``bullets`` and ``has_features_heading`` separated
    a real datasheet's front matter from a product-change notice's cover letter
    on both measured documents (34 and 43 bullets with a features heading on
    the PSoC 6; 0 and none on the TI PCN); ``legal_hits`` scored 0 on both --
    and on every real page measured -- so it is a third opinion, not the
    discriminator there. A unit-density signal is deliberately left out because
    it is noisy in both directions (it misses "150-MHz" and "40 microamp", and
    false-positives on part numbers like "CY8C62x8/A").

    These are heuristic counts, not an API: the vocabulary and patterns change
    as more documents are measured, so a caller should compare them, not
    threshold on exact values.
    """
    lines = text.splitlines()
    return {
        "bullets": sum(1 for line in lines if _BULLET_RE.match(line)),
        "legal_hits": count_legal_hits(text),
        "has_features_heading": any(
            line.strip().rstrip(":").strip().lower() in _FEATURES_HEADINGS
            for line in lines
        ),
    }


def build_front_matter(
    doc: pymupdf.Document,
    *,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> FrontMatter:
    """Extract page-marked front matter with per-page signals. Zero LLM calls.

    ``max_chars`` bounds *document text* (``chars_shown``), not the returned
    string: the ``--- PAGE N ---`` markers and any ``=== NOTE: ... ===`` line
    are tool framing. Marker overhead for pages 1..P is exactly
    ``sum(13 + digits(n) for n in 1..P) + (2 * P - 1)`` -- 31 at the default
    ``max_pages=2``. A note line adds around 100 characters -- measured at 77
    for the page note and 124 for the character note on a 134-page document.
    Those are estimates, not bounds, since the lines embed page numbers and
    counts.

    Raises ``ValueError`` unless ``max_pages`` is an integer ``>= 1``: zero
    pages read makes ``_page_phrase`` produce the nonsensical "pages 1-0", so
    it is rejected here rather than reaching that string. ``bool`` is
    rejected explicitly -- it is an ``int`` subclass, so ``True`` would
    silently become a cap of 1. ``max_chars`` is **rejected when negative**
    rather than clamped, for the same reason: the cap is quoted verbatim in the
    truncation note, and "truncated at -50 characters" is not a fact about the
    document. ``max_chars=0`` is legal and yields the note alone.

    A page whose text does not fit at all is **left out entirely**, marker
    included. A marker with nothing after it would read as "this page holds no
    text", which is a claim about document content that an exhausted budget
    does not license. A page that is genuinely blank still gets its marker.
    """
    if not isinstance(max_pages, int) or isinstance(max_pages, bool):
        raise ValueError("max_pages must be an integer >= 1")
    if max_pages < 1:
        raise ValueError("max_pages must be an integer >= 1")
    if max_chars < 0:
        raise ValueError("max_chars must be an integer >= 0")

    total_pages = len(doc)
    pages_read = max(0, min(max_pages, total_pages))
    texts = [_extract_page_text(doc[i]) for i in range(pages_read)]

    parts: list[str] = []
    chars_shown = 0
    cut_page = 0
    budget = max_chars
    for offset, page_text in enumerate(texts):
        page_num = offset + 1
        kept = (
            page_text
            if len(page_text) <= budget
            else _truncate_on_line_boundary(page_text, budget)
        )
        if not kept and page_text:
            # Nothing of this page fits. Its marker is withheld rather than
            # emitted empty, and `cut_page` stays 0 so the note reports a cut
            # on the page boundary -- which is where it landed.
            break
        parts.append(f"--- PAGE {page_num} ---")
        parts.append(kept)
        chars_shown += len(kept)
        budget -= len(kept)
        if kept != page_text:
            cut_page = page_num
            break

    pages = []
    for offset, page_text in enumerate(texts):
        signals = _page_signals(page_text)
        pages.append(
            PreamblePage(
                page=offset + 1,
                chars=len(page_text),
                bullets=signals["bullets"],
                legal_hits=signals["legal_hits"],
                has_features_heading=signals["has_features_heading"],
            )
        )

    chars_extracted = sum(len(t) for t in texts)
    # Notes join with the parts rather than concatenating onto an assembled
    # string: at max_chars=0 there are no parts, and a `+= "\n"` would leave
    # the returned text opening on a blank line.
    notes: list[str] = []
    if chars_shown < chars_extracted:
        notes.append(
            _char_note(
                max_chars=max_chars,
                chars_shown=chars_shown,
                chars_extracted=chars_extracted,
                pages_read=pages_read,
                cut_page=cut_page,
            )
        )
    if total_pages > pages_read:
        notes.append(_page_note(pages_read=pages_read, total_pages=total_pages))
    text = "\n".join(parts + notes)

    return FrontMatter(
        text=text,
        pages=pages,
        chars_shown=chars_shown,
        chars_extracted=chars_extracted,
        pages_read=pages_read,
        total_pages=total_pages,
    )


def generate_preamble(doc: pymupdf.Document, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Return only the page-marked front-matter text.

    Retained for callers that want the string; ``build_front_matter`` carries
    the signals and the truncation counts.
    """
    return build_front_matter(doc, max_chars=max_chars).text
