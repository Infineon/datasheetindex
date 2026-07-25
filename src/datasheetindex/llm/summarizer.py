"""Optional LLM-powered section summaries for TocNode trees."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from datasheetindex.core.textfile import extract_section_text
from datasheetindex.llm.untrusted import DATA_ONLY_INSTRUCTION, wrap_document_text
from datasheetindex.models import TocNode

if TYPE_CHECKING:
    from datasheetindex.llm.client import LlmCallable

MIN_TEXT_LENGTH = 100
MAX_SECTION_CHARS = 8000
_INTER_CALL_DELAY = 0.5  # seconds between LLM calls to avoid rate limits

SYSTEM_PROMPT = (
    "Summarize this datasheet section in 1-2 sentences. "
    "Focus on what the section covers and key specifications. " + DATA_ONLY_INSTRUCTION
)


def add_summaries(
    nodes: list[TocNode],
    text_content: str,
    llm_callable: LlmCallable,
) -> list[TocNode]:
    """Add LLM-generated summaries to each node in the tree.

    Modifies nodes in-place. Skips nodes whose text content is shorter
    than ``MIN_TEXT_LENGTH`` characters.
    """
    _summarize_recursive(nodes, text_content, llm_callable)
    return nodes


def _summarize_recursive(
    nodes: list[TocNode],
    text_content: str,
    llm_callable: LlmCallable,
    *,
    _is_first: bool = True,
) -> None:
    """Walk the tree and add summaries to each node."""
    for i, node in enumerate(nodes):
        section_text = extract_section_text(
            text_content, node.start_page, node.end_page
        )
        if len(section_text) >= MIN_TEXT_LENGTH:
            if not (_is_first and i == 0):
                time.sleep(_INTER_CALL_DELAY)
            truncated = section_text[:MAX_SECTION_CHARS]
            node.summary = llm_callable(SYSTEM_PROMPT, wrap_document_text(truncated))

        if node.nodes:
            _summarize_recursive(
                node.nodes, text_content, llm_callable, _is_first=False
            )
