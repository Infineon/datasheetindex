"""LLM-based ToC generation for PDFs with missing or poor ToC."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from datasheetindex.core.structure import build_tree
from datasheetindex.llm.client import get_structured_output_client
from datasheetindex.models import TocNode

if TYPE_CHECKING:
    from collections.abc import Callable

    from datasheetindex.llm.client import (
        LlmCallable,
        StructuredLlmCallable,
        StructuredLlmResult,
    )

logger = logging.getLogger(__name__)

CHUNK_SIZE = 15000
INTER_CHUNK_DELAY = 1.0  # seconds between LLM calls to avoid rate limits
MAX_PREVIOUS_CONTEXT_CHARS = 4000
MAX_PREVIOUS_CONTEXT_ENTRIES = 50

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

STRUCTURED_SYSTEM_PROMPT = (
    "You are an expert at identifying the hierarchical structure of "
    "technical datasheets. Extract sections from the provided text and "
    "return a JSON object with an `entries` array. Each entry must have: "
    "level (int, 1=top), title (str, original wording), "
    "start_page (int, from PAGE markers)."
)

STRUCTURED_INIT_USER_PROMPT = (
    "Analyze this datasheet text chunk and identify its hierarchical section "
    "structure. The text contains --- PAGE N --- markers indicating page "
    "boundaries.\n\n"
    "Return only section entries whose `start_page` is supported by this chunk.\n\n"
    "Text:\n{text}"
)

STRUCTURED_CONTINUE_USER_PROMPT = (
    "Analyze this next datasheet text chunk and continue identifying the "
    "hierarchical section structure. The text contains --- PAGE N --- markers "
    "indicating page boundaries.\n\n"
    "Sections already captured from earlier chunks:\n{previous}\n\n"
    "Return only NEW section entries from this chunk.\n\n"
    "Text:\n{text}"
)

STRUCTURED_TOC_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "entries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "level": {"type": "integer", "minimum": 1},
                    "title": {"type": "string"},
                    "start_page": {"type": "integer", "minimum": 1},
                },
                "required": ["level", "title", "start_page"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["entries"],
    "additionalProperties": False,
}


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

    all_entries: list[dict[str, int | str]] = []

    structured_llm = get_structured_output_client(llm_callable)
    if structured_llm is not None:
        all_entries = _collect_entries(chunks, _structured_extractor(structured_llm))
        if not all_entries:
            # The structured path yielded nothing at all: the model or gateway
            # may not honour ``text.format=json_schema``. Retry the whole
            # document with the free-text prompt rather than give up on a ToC.
            logger.warning(
                "Structured ToC extraction yielded no entries; "
                "retrying with the free-text prompt"
            )

    if not all_entries:
        all_entries = _collect_entries(chunks, _legacy_extractor(llm_callable))

    # Convert flat entries to TocNode tree via the shared builder
    raw_toc = [[e["level"], e["title"], e["start_page"]] for e in all_entries]
    return build_tree(raw_toc, total_pages)


@dataclass(frozen=True)
class _ChunkExtractor:
    """One way of turning a text chunk into ToC entries."""

    name: str
    system_prompt: str
    init_prompt: str
    continue_prompt: str
    run: Callable[[str, str], list[dict]]


def _legacy_extractor(llm_callable: LlmCallable) -> _ChunkExtractor:
    def run(system: str, user: str) -> list[dict]:
        return _parse_json_response(llm_callable(system, user))

    return _ChunkExtractor(
        name="Free-text",
        system_prompt=SYSTEM_PROMPT,
        init_prompt=INIT_USER_PROMPT,
        continue_prompt=CONTINUE_USER_PROMPT,
        run=run,
    )


def _structured_extractor(llm_callable: StructuredLlmCallable) -> _ChunkExtractor:
    def run(system: str, user: str) -> list[dict]:
        result = llm_callable.structured_json(
            system,
            user,
            name="datasheet_toc_chunk",
            schema=STRUCTURED_TOC_SCHEMA,
        )
        return _parse_structured_chunk_response(result)

    return _ChunkExtractor(
        name="Structured",
        system_prompt=STRUCTURED_SYSTEM_PROMPT,
        init_prompt=STRUCTURED_INIT_USER_PROMPT,
        continue_prompt=STRUCTURED_CONTINUE_USER_PROMPT,
        run=run,
    )


def _collect_entries(
    chunks: list[str], extractor: _ChunkExtractor
) -> list[dict[str, int | str]]:
    """Run one extractor over every chunk, tolerating per-chunk failures.

    A chunk whose response is truncated, malformed, or refused is logged and
    skipped -- it must not discard the entries already collected from the other
    chunks. Whether the surviving entries are a good enough ToC to ship is not
    decided here: ``index._accept_llm_toc_candidate`` scores the assembled
    candidate against the original and rejects it if it came back too thin.

    A first chunk that fails aborts the run, since a path that cannot handle
    chunk 1 (unsupported schema, bad credentials) will not handle chunk 40
    either, and the caller has a cheaper path to fall back to.
    """
    all_entries: list[dict[str, int | str]] = []

    for i, chunk in enumerate(chunks):
        if i == 0:
            user_msg = extractor.init_prompt.format(text=chunk)
        else:
            time.sleep(INTER_CHUNK_DELAY)
            user_msg = extractor.continue_prompt.format(
                previous=_format_previous_entries(all_entries),
                text=chunk,
            )

        try:
            entries = extractor.run(extractor.system_prompt, user_msg)
        except Exception:
            logger.warning(
                "%s ToC extraction failed on chunk %d/%d; skipping it",
                extractor.name,
                i + 1,
                len(chunks),
                exc_info=True,
            )
            if i == 0:
                return []
            continue

        all_entries.extend(_dedupe_entries(all_entries, entries))

    return all_entries


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

    return _normalize_entries(data)


def _parse_structured_chunk_response(result: StructuredLlmResult) -> list[dict]:
    # ``status`` is optional: gateways and third-party structured callables that
    # do not report one leave it None, which says nothing about the payload.
    # Only a status that is explicitly something other than "completed" means
    # the response was truncated or aborted.
    if result.status is not None and result.status != "completed":
        detail = (
            f": {result.incomplete_details}"
            if result.incomplete_details is not None
            else ""
        )
        raise ValueError(f"Structured ToC chunk did not complete{detail}")

    try:
        data = json.loads(result.output_text)
    except json.JSONDecodeError as exc:
        raise ValueError("Structured ToC chunk returned invalid JSON") from exc

    if not isinstance(data, dict):
        raise ValueError("Structured ToC chunk returned a non-object payload")

    raw_entries = data.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError("Structured ToC chunk response missing entries")

    return _normalize_entries(raw_entries)


def _coerce_int(value: object) -> int:
    """Coerce an LLM-supplied value to an int, rejecting what cannot be one."""
    if isinstance(value, int):
        return value
    if isinstance(value, str | float):
        return int(value)
    raise TypeError(f"cannot use {value!r} as an integer")


def _normalize_entries(raw_entries: list[object]) -> list[dict]:
    valid = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            continue
        entry = cast("dict[str, object]", raw_entry)
        try:
            normalized = {
                "level": _coerce_int(entry["level"]),
                "title": str(entry["title"]),
                "start_page": _coerce_int(entry["start_page"]),
            }
        except (KeyError, TypeError, ValueError):
            # One unusable entry -- a missing key, or the null start_page the
            # model emits when it cannot find a PAGE marker -- must not take
            # the entries around it down with it.
            continue
        valid.append(normalized)
    return valid


def _dedupe_entries(
    existing: list[dict[str, int | str]],
    new_entries: list[dict],
) -> list[dict]:
    seen = {
        (int(entry["level"]), str(entry["title"]).strip(), int(entry["start_page"]))
        for entry in existing
    }
    deduped: list[dict] = []
    for entry in new_entries:
        key = (
            int(entry["level"]),
            str(entry["title"]).strip(),
            int(entry["start_page"]),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


def _format_previous_entries(entries: list[dict[str, int | str]]) -> str:
    if not entries:
        return "[]"

    tail = entries[-MAX_PREVIOUS_CONTEXT_ENTRIES:]
    while tail:
        text = json.dumps(tail, indent=2)
        if len(text) <= MAX_PREVIOUS_CONTEXT_CHARS:
            return text
        tail = tail[1:]
    return "[]"
