"""Shared text-normalization and token-matching helpers.

Extracted from ``textfile.py`` so both the page-text search ladder and the
``locate_text`` coordinate primitive share one normalization implementation
instead of reaching across modules into private names.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from functools import cache, lru_cache
from typing import NamedTuple

# Normalize Unicode hyphen/dash/minus code points to ASCII "-" so a hyphen
# query matches a datasheet that uses an en-dash, figure dash, or minus sign.
# Code points: U+2010..U+2015 (hyphen/dashes) and U+2212 (minus sign).
_DASH_TRANSLATION = str.maketrans(
    {chr(cp): "-" for cp in (0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2015, 0x2212)}
)

_TOKEN_EDGE_PUNCTUATION = ".,;:!?"

_TOKEN_RE = re.compile(r"\S+")


class _TokenSpan(NamedTuple):
    value: str
    start: int
    end: int


@lru_cache(maxsize=256)
def _translate_search_text(text: str) -> str:
    return text.translate(_DASH_TRANSLATION)


def _normalize_token(token: str, *, case_sensitive: bool) -> str:
    normalized = _translate_search_text(token).strip(_TOKEN_EDGE_PUNCTUATION)
    return normalized if case_sensitive else normalized.casefold()


def _match_query_tokens(
    page_tokens: Sequence[_TokenSpan],
    query_tokens: Sequence[str],
    start_index: int,
    *,
    max_gap_tokens: int,
) -> list[int] | None:
    if page_tokens[start_index].value != query_tokens[0]:
        return None

    @cache
    def _search(query_index: int, previous_token_index: int) -> tuple[int, ...] | None:
        if query_index >= len(query_tokens):
            return ()

        expected = query_tokens[query_index]
        search_start = previous_token_index + 1
        search_end = min(len(page_tokens), previous_token_index + max_gap_tokens + 2)
        for token_index in range(search_start, search_end):
            if page_tokens[token_index].value != expected:
                continue
            suffix = _search(query_index + 1, token_index)
            if suffix is not None:
                return (token_index, *suffix)
        return None

    suffix = _search(1, start_index)
    if suffix is None:
        return None
    return [start_index, *suffix]
