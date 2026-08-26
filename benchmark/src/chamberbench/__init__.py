"""Chamber-grounded benchmark: the grading surface, and the archive it graded.

This package is the *scoring* half of the benchmark described in the
accompanying paper. It contains everything needed to re-derive a published
number from archived model output, and nothing that calls a model to produce
new output (with one exception, ``classifier``, whose auto-labelling pass is
LLM-backed).

Two axes are graded, and keeping them apart is the point of the design:

``grading``
    *Fidelity* -- did the agent report what the datasheet says? A pure
    function of the extracted value and the claim's expectations.

``reproducibility``
    *Reproducibility* -- is the claimed value physically true, as measured in
    the chamber? ``verdict()`` takes a ``ClaimSpec`` and a
    ``ChamberMeasurement`` and never sees agent output at all. That
    independence is structural, not a convention.

``silent_failure``
    The two dispatch-level detector rules. Both are predicates over the
    ``datasheetindex`` tool surface, which is why the benchmark lives in this
    repository.

The package is not shipped in the ``datasheetindex`` wheel; it is a
repository-only artifact installed from ``benchmark/``.
"""

from __future__ import annotations

__all__ = ["ARCHIVE_DIR", "BENCHMARK_ROOT", "DATA_DIR", "__version__"]

__version__ = "1.0.0"

from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = BENCHMARK_ROOT / "data"
ARCHIVE_DIR = BENCHMARK_ROOT / "archive"
