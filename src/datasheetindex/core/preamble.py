"""Front matter extraction: page-marked pages 1-2 text for agent orientation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pymupdf

from datasheetindex.core.textfile import _extract_page_text

#: Pages read from the front of the document.
DEFAULT_MAX_PAGES = 2
#: Document characters emitted. Raised from 2400 in 0.25.0: the old value cut
#: the measured 4747-character front matter of the PSoC 6 in half and kept the
#: half holding the legal footer rather than the specifications.
DEFAULT_MAX_CHARS = 5000


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
    ``max_pages=2``. Note lines add roughly 120 characters each; that is an
    estimate, not a bound, since they embed page numbers and counts.
    """
    total_pages = len(doc)
    pages_read = max(0, min(max_pages, total_pages))
    texts = [_extract_page_text(doc[i]) for i in range(pages_read)]

    parts: list[str] = []
    chars_shown = 0
    cut_page = 0
    budget = max_chars
    for offset, page_text in enumerate(texts):
        page_num = offset + 1
        if budget <= 0 and page_num > 1:
            break
        kept = (
            page_text
            if len(page_text) <= budget
            else _truncate_on_line_boundary(page_text, budget)
        )
        parts.append(f"--- PAGE {page_num} ---")
        parts.append(kept)
        chars_shown += len(kept)
        budget -= len(kept)
        if kept != page_text:
            cut_page = page_num
            break

    pages = [
        PreamblePage(page=offset + 1, chars=len(page_text))
        for offset, page_text in enumerate(texts)
    ]

    chars_extracted = sum(len(t) for t in texts)
    text = "\n".join(parts)
    if chars_shown < chars_extracted:
        text += "\n" + _char_note(
            max_chars=max_chars,
            chars_shown=chars_shown,
            chars_extracted=chars_extracted,
            pages_read=pages_read,
            cut_page=cut_page,
        )
    if total_pages > pages_read:
        text += "\n" + _page_note(pages_read=pages_read, total_pages=total_pages)

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
