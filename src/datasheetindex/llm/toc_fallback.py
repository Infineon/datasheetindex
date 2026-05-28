"""LLM-based ToC generation for PDFs with missing or poor ToC."""

from __future__ import annotations

import json
import re
import time
from typing import TYPE_CHECKING

from datasheetindex.core.structure import build_tree
from datasheetindex.models import TocNode

if TYPE_CHECKING:
    from datasheetindex.llm.client import LlmCallable

CHUNK_SIZE = 15000
INTER_CHUNK_DELAY = 1.0  # seconds between LLM calls to avoid rate limits

SYSTEM_PROMPT = (
    "You are an expert at identifying the hierarchical structure of "
    "technical datasheets. Extract sections from the provided text and "
    "return a JSON array. Each entry must have: level (int, 1=top), "
    "title (str, original wording), start_page (int, from PAGE markers)."
)

INIT_USER_PROMPT = (
    "Analyze this datasheet text and identify its hierarchical section "
    "structure. The text contains --- PAGE N --- markers indicating page "
    "boundaries.\n\n"
    "Return ONLY a JSON array, no other text:\n"
    '[{{"level": 1, "title": "Section Name", "start_page": 1}}, ...]\n\n'
    "Text:\n{text}"
)

CONTINUE_USER_PROMPT = (
    "Continue extracting the section structure from the next part of the "
    "datasheet. Here is the structure found so far:\n{previous}\n\n"
    "Return ONLY the NEW sections as a JSON array (do not repeat previous "
    "sections):\n"
    '[{{"level": 1, "title": "Section Name", "start_page": 10}}, ...]\n\n'
    "Text:\n{text}"
)


def generate_toc_from_text(
    text_content: str,
    total_pages: int,
    llm_callable: LlmCallable,
) -> list[TocNode]:
    """Generate a ToC tree from raw page-marked text using an LLM.

    Splits text into chunks on ``--- PAGE N ---`` boundaries, sends each
    chunk to the LLM, and assembles the results into a ``TocNode`` tree.
    """
    chunks = _split_into_chunks(text_content, CHUNK_SIZE)
    if not chunks:
        return []

    all_entries: list[dict] = []

    # First chunk: init prompt
    user_msg = INIT_USER_PROMPT.format(text=chunks[0])
    raw = llm_callable(SYSTEM_PROMPT, user_msg)
    entries = _parse_json_response(raw)
    all_entries.extend(entries)

    # Subsequent chunks: continue prompt with context
    for chunk in chunks[1:]:
        time.sleep(INTER_CHUNK_DELAY)
        previous_json = json.dumps(all_entries, indent=2)
        user_msg = CONTINUE_USER_PROMPT.format(previous=previous_json, text=chunk)
        raw = llm_callable(SYSTEM_PROMPT, user_msg)
        entries = _parse_json_response(raw)
        all_entries.extend(entries)

    # Convert flat entries to TocNode tree via the shared builder
    raw_toc = [[e["level"], e["title"], e["start_page"]] for e in all_entries]
    return build_tree(raw_toc, total_pages)


def _split_into_chunks(text: str, max_chars: int) -> list[str]:
    """Split text into chunks on ``--- PAGE N ---`` boundaries.

    Each chunk stays under ``max_chars`` while respecting page boundaries.
    """
    page_pattern = re.compile(r"(--- PAGE \d+ ---)")
    parts = page_pattern.split(text)

    chunks: list[str] = []
    current = ""

    for part in parts:
        if len(current) + len(part) > max_chars and current:
            chunks.append(current)
            # Start new chunk; if previous ended mid-page, include marker
            current = part
        else:
            current += part

    if current.strip():
        chunks.append(current)

    return chunks


def _parse_json_response(raw: str) -> list[dict]:
    """Extract a JSON array from the LLM response.

    Handles responses wrapped in markdown code blocks.
    """
    text = raw.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (fences)
        lines = [line for line in lines[1:] if not line.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find a JSON array in the response
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            return []

    if not isinstance(data, list):
        return []

    # Validate entries
    valid = []
    for entry in data:
        if (
            isinstance(entry, dict)
            and "level" in entry
            and "title" in entry
            and "start_page" in entry
        ):
            valid.append(
                {
                    "level": int(entry["level"]),
                    "title": str(entry["title"]),
                    "start_page": int(entry["start_page"]),
                }
            )
    return valid
