"""Batch processing for multiple PDF datasheets."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from datasheetindex.index import DatasheetIndex
from datasheetindex.models import DatasheetArtifacts

if TYPE_CHECKING:
    from datasheetindex.llm.client import LlmCallable

logger = logging.getLogger(__name__)


@dataclass
class BatchError:
    """Record of a single PDF processing failure."""

    pdf_path: str
    error: str


@dataclass
class BatchResult:
    """Aggregated result of a batch build run."""

    succeeded: list[DatasheetArtifacts] = field(default_factory=list)
    failed: list[BatchError] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.succeeded) + len(self.failed)

    @property
    def success_count(self) -> int:
        return len(self.succeeded)

    @property
    def failure_count(self) -> int:
        return len(self.failed)


def _allocate_output_stem(
    base_stem: str, used_output_stems: set[str], suffix_counters: dict[str, int]
) -> str:
    """Allocate a unique output stem while preserving the first filename."""
    if base_stem not in used_output_stems:
        used_output_stems.add(base_stem)
        suffix_counters[base_stem] = 1
        return base_stem

    next_suffix = suffix_counters.get(base_stem, 1) + 1
    candidate = f"{base_stem}-{next_suffix}"
    while candidate in used_output_stems:
        next_suffix += 1
        candidate = f"{base_stem}-{next_suffix}"

    suffix_counters[base_stem] = next_suffix
    used_output_stems.add(candidate)
    return candidate


def build_batch(
    pdf_paths: list[str],
    output_dir: str = "output",
    include_summaries: bool = False,
    llm_callable: LlmCallable | None = None,
) -> BatchResult:
    """Process multiple PDFs, collecting successes and failures.

    Each PDF is processed independently; a failure in one does not
    stop the rest of the batch. Output filenames are made unique when
    multiple inputs would otherwise resolve to the same stem.
    """
    result = BatchResult()
    used_output_stems: set[str] = set()
    suffix_counters: dict[str, int] = {}

    for path in pdf_paths:
        idx = DatasheetIndex(path)
        output_stem = _allocate_output_stem(
            idx._output_stem(),
            used_output_stems,
            suffix_counters,
        )
        try:
            artifacts = idx.build(
                output_dir=output_dir,
                include_summaries=include_summaries,
                llm_callable=llm_callable,
                output_stem=output_stem,
            )
            result.succeeded.append(artifacts)
        except Exception as exc:
            logger.warning("Failed to process %s: %s", path, exc)
            result.failed.append(BatchError(pdf_path=path, error=str(exc)))
        finally:
            idx.close()

    return result
