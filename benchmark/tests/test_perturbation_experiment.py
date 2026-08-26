"""Unit tests for the reproducibility-perturbation sweep (offline, no LLM).

One test at the bottom of this file, `test_build_only_writes_verified_pdf`,
covers `scripts/perturbation.py --build-only` instead and is marked
`network`: building the perturbed PDF needs to download the source datasheet
on a machine that has never fetched it before. It is excluded from the
default run by the `-m "not network"` addopts in pyproject.toml, so
`uv run pytest -q` stays offline; run it explicitly with `-m network` (and
network access) to exercise it.
"""

from __future__ import annotations

import pytest

# `pymupdf` arrives with `datasheetindex`, and `requests` with the harness
# extra; neither is a Tier 1 dependency. Skip rather than raise a collection
# error, so a Tier-1-only install still runs the suite. The pure sweep under
# test lives in `chamberbench.perturbation` and needs none of this -- it is the
# PDF-building half of `scripts/perturbation.py` that does.
pymupdf = pytest.importorskip(
    "pymupdf",
    reason="needs the harness extra: uv pip install -e '.[harness]'",
)
requests = pytest.importorskip(
    "requests",
    reason="needs the harness extra: uv pip install -e '.[harness]'",
)

# pyproject sets pythonpath = ["src", "scripts"]; no sys.path surgery needed.
import perturbation
from chamberbench.claims import ChamberMeasurement, ClaimSpec
from chamberbench.perturbation import sweep_claimed_max


def _base_claim() -> ClaimSpec:
    return ClaimSpec(
        id="dps310-operating-range-test",
        pdf_source="local",
        parameter="Operating pressure range",
        expected_unit="hPa",
        claim_kind="range",
        claimed_min=300.0,
        claimed_max=1200.0,
        tolerance_kind="absolute",
        tolerance_value=0.0,
        chamber_protocol="chamberbench.protocols.barometer_dc_accuracy",
        primary_chamber_variable="pressure_intake",
    )


def _measurement() -> ChamberMeasurement:
    return ChamberMeasurement(
        claim_id="dps310-operating-range-test",
        measured_value=945.285,
        measured_unit="hPa",
        measured_sigma=0.0801,
        measured_sigma_basis="cross_sensor",
    )


def test_sweep_transitions_pass_inconclusive_fail():
    rows = sweep_claimed_max(
        _base_claim(), _measurement(), [945.6, 945.285, 945.25, 945.20, 945.0]
    )
    assert [r["verdict"] for r in rows] == [
        "pass",
        "pass",
        "inconclusive",
        "fail",
        "fail",
    ]


def test_transition_edge_is_combined_uncertainty():
    rows = sweep_claimed_max(_base_claim(), _measurement(), [945.25, 945.20])
    inconclusive, fail = rows[0], rows[1]
    assert inconclusive["boundary_distance"] <= inconclusive["combined_uncertainty"]
    assert fail["boundary_distance"] > fail["combined_uncertainty"]
    # combined is purely the measurement sigma (spec_tol == 0)
    assert abs(fail["combined_uncertainty"] - 0.0801) < 1e-6


@pytest.mark.network
def test_build_only_writes_verified_pdf(monkeypatch):
    """`--build-only` builds and verifies the perturbed PDF without any
    credentials and without ever reaching a model.

    No API key is set, so a build that somehow needed one would raise
    (`setup_credentials()` is never called on this path) instead of silently
    picking up an ambient key. Skipped, not failed, if there is no network
    route to the source datasheet -- that is an environment limit, not a
    defect in `--build-only` itself.
    """
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "LITELLM_MASTER_KEY"):
        monkeypatch.delenv(var, raising=False)

    try:
        path = perturbation.build_only()
    except requests.exceptions.RequestException as exc:
        pytest.skip("no network route to the source datasheet: " + str(exc))

    assert path == perturbation.PERTURBED_PDF
    assert path.exists()

    doc = pymupdf.open(path)
    try:
        texts = [page.get_text() for page in doc]
    finally:
        doc.close()
    assert all("1200" not in t for t in texts)
    assert "945" in texts[0]
