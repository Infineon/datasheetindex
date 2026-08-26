"""Unit tests for the reproducibility-perturbation sweep (offline, no LLM)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

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
