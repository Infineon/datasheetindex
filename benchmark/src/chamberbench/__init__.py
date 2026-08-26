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

from chamberbench.claimsio import (
    archive_dir,
    data_dir,
    load_archive,
    load_claim,
    load_claims,
)
from chamberbench.grading import evaluate_case
from chamberbench.reproducibility import verdict
from chamberbench.silent_failure import detect_silent_failure

__version__ = "1.0.0"

# The public API is FUNCTIONS, not path constants. It used to export
# BENCHMARK_ROOT / DATA_DIR / ARCHIVE_DIR, computed once at import -- which
# silently ignored CHAMBERBENCH_DATA_DIR and CHAMBERBENCH_ARCHIVE_DIR, the
# overrides the documentation tells a reproduction to set. Someone following
# `__all__` would have got a path that quietly disregarded their override: a
# wrong answer rather than an error.
__all__ = [
    "__version__",
    "archive_dir",
    "data_dir",
    "detect_silent_failure",
    "evaluate_case",
    "load_archive",
    "load_claim",
    "load_claims",
    "verdict",
]
