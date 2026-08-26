"""Loading claim sets and locating the shipped data and archive.

Every consumer previously reached for the claim file with its own
``PROJECT_ROOT / "eval" / "chamber" / "claims.yaml"`` expression, computed by
counting ``.parents[n]`` from wherever that consumer happened to live. Three
copies had already drifted apart on which claim file they meant. Centralising
it here means a script's location no longer encodes a data path.

``CHAMBERBENCH_DATA_DIR`` and ``CHAMBERBENCH_ARCHIVE_DIR`` override the
defaults, which is what lets a reproduction point the same code at a
re-derived claim set or a freshly produced archive without editing it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from chamberbench.claims import ClaimSpec

BENCHMARK_ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    """Directory holding the claim sets and annotation files."""
    return Path(os.environ.get("CHAMBERBENCH_DATA_DIR", BENCHMARK_ROOT / "data"))


def archive_dir() -> Path:
    """Directory holding the archived model outputs the paper reports."""
    return Path(os.environ.get("CHAMBERBENCH_ARCHIVE_DIR", BENCHMARK_ROOT / "archive"))


def corpus_dir() -> Path:
    """Where the datasheet PDFs live.

    Not committed: the corpus is third-party vendor documentation. See
    ``docs/reproducing.md`` for the fetch step and checksums.
    """
    return Path(os.environ.get("CHAMBERBENCH_CORPUS_DIR", BENCHMARK_ROOT / "corpus"))


CLAIMS_FILENAME = "claims.yaml"
A4988_CLAIMS_FILENAME = "claims_a4988.yaml"


def claims_path(filename: str = CLAIMS_FILENAME) -> Path:
    return data_dir() / filename


def load_claims(filename: str = CLAIMS_FILENAME) -> list[ClaimSpec]:
    """Parse a claim file into validated ``ClaimSpec`` instances.

    Validation is not incidental: a claim carries the substrings and
    confidence floor that decide fidelity, and a malformed one would
    otherwise fail much later, inside a comparison, as a wrong verdict
    rather than an error.
    """
    raw = yaml.safe_load(claims_path(filename).read_text(encoding="utf-8"))
    return [ClaimSpec.model_validate(c) for c in raw["claims"]]


def load_claim(claim_id: str, filename: str = CLAIMS_FILENAME) -> ClaimSpec:
    """Return one claim by id, or raise ``KeyError``."""
    for claim in load_claims(filename):
        if claim.id == claim_id:
            return claim
    raise KeyError(f"claim not found: {claim_id}")


def load_archive(name: str) -> Any:
    """Read one archived result file by name, e.g. ``baseline_chamber.json``."""
    import json

    return json.loads((archive_dir() / name).read_text(encoding="utf-8"))


def short_path(p: Path) -> str:
    """Path relative to the benchmark root when possible, absolute otherwise.

    `CHAMBERBENCH_ARCHIVE_DIR` / `CHAMBERBENCH_DATA_DIR` are documented as the
    way to point this code at an external archive or claim set, so paths
    derived from them routinely sit outside the benchmark tree. A bare
    `Path.relative_to` raises `ValueError` on exactly that case -- and did so
    in five reporting scripts, AFTER the work was done, turning a completed
    run into a non-zero exit.
    """
    try:
        return str(p.relative_to(BENCHMARK_ROOT))
    except ValueError:
        return str(p)
