"""Controlled perturbation of a claimed bound against a fixed measurement.

The paper's perturbation figure sweeps one claim's `claimed_max` from solidly
passing to solidly failing while holding the physical measurement constant,
and shows the reproducibility verdict flipping at the measurement uncertainty
rather than at the nominal bound.

Extracted from the experiment driver, whose remaining half re-runs the agent
against a perturbed PDF and therefore belongs with the harness. The sweep
itself consumes only a `ClaimSpec` and a `ChamberMeasurement`, which is the
same independence `reproducibility.verdict` relies on: nothing here sees agent
output.
"""

from __future__ import annotations

from typing import Any

from chamberbench.claims import ChamberMeasurement, ClaimSpec
from chamberbench.reproducibility import verdict

TARGET_CLAIM_ID = "dps310-operating-range"


def perturb_claimed_max(claim: ClaimSpec, new_max: float) -> ClaimSpec:
    """Clone `claim` with `claimed_max` overridden.

    The physical measurement is independent of the claimed bound, so callers
    reuse one measurement across the whole sweep.
    """
    return claim.model_copy(update={"claimed_max": float(new_max)})


def _boundary_distance(
    measured: float, claimed_min: float, claimed_max: float
) -> float:
    """How far `measured` sits outside [claimed_min, claimed_max] (0 if inside)."""
    if measured < claimed_min:
        return claimed_min - measured
    if measured > claimed_max:
        return measured - claimed_max
    return 0.0


def sweep_claimed_max(
    base_claim: ClaimSpec,
    measurement: ChamberMeasurement,
    claimed_maxes: list[float],
) -> list[dict[str, Any]]:
    """Verdict for each perturbed `claimed_max` against the fixed measurement."""
    rows: list[dict[str, Any]] = []
    m = measurement.measured_value
    # divergence > 0 means the measurement sits above claimed_max (outside the range)
    for hi in claimed_maxes:
        pc = perturb_claimed_max(base_claim, hi)
        v = verdict(pc, measurement)
        rows.append(
            {
                "claimed_max": float(hi),
                "divergence": m - float(hi),
                "boundary_distance": _boundary_distance(
                    m, pc.claimed_min, pc.claimed_max
                ),
                "verdict": v.verdict,
                "combined_uncertainty": v.combined_uncertainty,
            }
        )
    return rows
