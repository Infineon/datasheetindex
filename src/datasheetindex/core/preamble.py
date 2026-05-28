"""Pages 1-2 raw text extraction for agent orientation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pymupdf

from datasheetindex.core.textfile import _extract_page_text


def generate_preamble(doc: pymupdf.Document, max_chars: int = 2400) -> str:
    """Extract raw text from pages 1-2 for agent orientation.

    Truncates on a line boundary at approximately ``max_chars`` characters
    (roughly 600 tokens). Zero LLM calls.
    """
    pages_to_read = min(2, len(doc))
    parts: list[str] = []
    for i in range(pages_to_read):
        parts.append(_extract_page_text(doc[i]))

    full_text = "\n".join(parts)

    if len(full_text) <= max_chars:
        return full_text

    # Truncate on a line boundary
    lines = full_text.splitlines(keepends=True)
    result: list[str] = []
    total = 0
    for line in lines:
        if total + len(line) > max_chars:
            break
        result.append(line)
        total += len(line)

    return "".join(result).rstrip()
