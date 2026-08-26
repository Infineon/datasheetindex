"""Reproducibility protocol for DPS310 DC-accuracy claims.

Strategy
--------
The chamber has four DPS310 barometers at different positions. We do NOT have
a NIST-traceable reference. What we can test cleanly from the public datasets:

* **Relative accuracy** -- spread among the four units at the same physical
  pressure. Compare the primary sensor's reading to the mean of the others
  in a steady-state window.
* **Absolute accuracy (within scope)** -- if the spec says +/-X hPa, we report
  the maximum primary-vs-mean offset over the realizable pressure window. A
  passing measurement is necessary-but-not-sufficient for the absolute claim
  (the four units could share a common bias against true pressure); the
  verdict module flags this honestly via `unmatched_conditions`.

Either way, the chamber-side sigma is `cross_sensor_sigma`, and the rationale
is recorded in `ChamberMeasurement.measured_sigma_basis = "cross_sensor"`.

Run as a script for smoke testing:
    uv run python -m eval.chamber.protocols.barometer_dc_accuracy
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..uncertainty import (
    cross_sensor_sigma,
    select_steady_state_window,
)
from ._common import (
    WT_ACTUATORS,
    make_stub_measurement,
    match_conditions,
    resolve_cache_root,
    unmatched_load_bearing,
)

if TYPE_CHECKING:
    from chamberbench.claims import ChamberMeasurement, ClaimSpec


# Conventional steady-state experiments by claim kind. Used when
# claim.chamber_experiment_hint is empty.
DEFAULT_EXPERIMENTS: dict[str, str] = {
    "dc_accuracy": "validate_load_out_pressure_intake",
    "range": "validate_load_out_pressure_intake",
    "typical": "validate_load_out_pressure_intake",
}


def run(claim: ClaimSpec) -> ChamberMeasurement:
    """Compute a `ChamberMeasurement` for a DPS310 DC-accuracy claim."""
    # Lazy import for the optional chamber dependency.
    from causalchamber.datasets import Dataset

    from chamberbench.claims import ChamberMeasurement

    # Short-circuit when any load-bearing operating condition has no
    # chamber_variable mapping. The chamber cannot exercise the stated
    # regime, so we return a stub measurement; verdict() will see
    # sigma_basis="stub" and emit inconclusive. This avoids running
    # numerical code (and unit conversions for units like uA/Hz/V/ms
    # that have no chamber-side meaning) when the result wouldn't be
    # consumed anyway.
    unmatched_lb = unmatched_load_bearing(claim)
    if unmatched_lb:
        return make_stub_measurement(claim, unmatched_load_bearing_names=unmatched_lb)

    cache_root = resolve_cache_root()
    ds = Dataset(claim.chamber_dataset, root=str(cache_root), download=True)

    experiment_id = claim.chamber_experiment_hint or DEFAULT_EXPERIMENTS.get(
        claim.claim_kind, "validate_load_out_pressure_intake"
    )
    df = ds.get_experiment(experiment_id).as_pandas_dataframe()

    # Restrict to a steady-state window so the cross-sensor sigma reflects
    # readout disagreement, not transient physical differences across the
    # tunnel (upwind/downwind only equalise when fans + hatch are stable).
    df_steady = select_steady_state_window(df, WT_ACTUATORS, min_samples=50)

    primary = claim.primary_chamber_variable
    redundants = list(claim.cross_check_variables)

    primary_arr = df_steady[primary].to_numpy(dtype=float)
    redundants_arr = df_steady[redundants].to_numpy(dtype=float)
    others_mean = redundants_arr.mean(axis=1)
    diff = primary_arr - others_mean

    # Convert from Pa (chamber units) to the claim's expected unit if needed.
    chamber_unit = "Pa"
    scale = _unit_scale(chamber_unit, claim.expected_unit)

    # Choice of `measured_value` depends on claim kind:
    # - dc_accuracy:    offset between primary and inter-unit mean (tests
    #                   cross-unit agreement / the relative claim).
    # - range:          absolute steady-state pressure read by the primary
    #                   sensor (tests the part is operating inside the
    #                   stated envelope).
    # - typical:        per-sensor standard deviation in steady state. Used
    #                   for precision-style claims (e.g. DPS310 Ap_prc).
    # - max | min:      offset (same as dc_accuracy); the verdict logic
    #                   handles single-bound comparison.
    if claim.claim_kind == "range":
        measured_value = float(primary_arr.mean()) * scale
    elif claim.claim_kind == "typical":
        measured_value = (
            float(primary_arr.std(ddof=1)) * scale if primary_arr.size > 1 else 0.0
        )
    else:
        measured_value = float(diff.mean()) * scale

    chamber_sigma = cross_sensor_sigma(df_steady, primary, redundants) * scale

    matched, unmatched = match_conditions(claim, df_steady)

    return ChamberMeasurement(
        claim_id=claim.id,
        experiment_ids=[experiment_id],
        measured_value=measured_value,
        measured_unit=claim.expected_unit,
        measured_sigma=chamber_sigma,
        measured_sigma_basis="cross_sensor",
        matched_conditions=matched,
        unmatched_conditions=unmatched,
        sample_n=len(df_steady),
        cross_sensor_spread=float(redundants_arr.std(axis=1, ddof=1).mean()) * scale,
        notes=(
            "Offset is primary - mean(redundants) over the steady-state window. "
            "Cross-sensor sigma bounds inter-unit relative agreement, not absolute "
            "calibration; a common bias across all four units would not be caught."
        ),
    )


def _unit_scale(chamber_unit: str, claim_unit: str) -> float:
    """Convert from chamber readout unit to claim unit.

    Chamber pressures are reported in Pa. Datasheet pressure claims are
    typically in hPa or kPa. Add cases as new claim units appear.
    """
    if chamber_unit == claim_unit:
        return 1.0
    table = {
        ("Pa", "hPa"): 1e-2,
        ("Pa", "kPa"): 1e-3,
        ("hPa", "Pa"): 1e2,
        ("kPa", "Pa"): 1e3,
        ("hPa", "kPa"): 1e-1,
        ("kPa", "hPa"): 10.0,
    }
    try:
        return table[(chamber_unit, claim_unit)]
    except KeyError as exc:
        raise ValueError(
            f"No unit conversion defined: {chamber_unit} -> {claim_unit}"
        ) from exc


if __name__ == "__main__":
    # Smoke test using a synthetic ClaimSpec, no LLM involvement.
    from chamberbench.claims import ClaimSpec, OperatingCondition

    claim = ClaimSpec(
        id="dps310-relative-accuracy",
        pdf_source="eval/chamber/datasheets/barometer.pdf",
        parameter="Relative pressure accuracy",
        expected_unit="hPa",
        claim_kind="dc_accuracy",
        claimed_min=-0.06,
        claimed_max=0.06,
        tolerance_value=0.06,
        tolerance_kind="absolute",
        operating_conditions=[
            OperatingCondition(
                name="temperature_C",
                value=25.0,
                unit="C",
                chamber_variable="res_in",
                load_bearing=False,
            ),
        ],
        chamber_dataset="wt_validate_v1",
        chamber_protocol="chamberbench.protocols.barometer_dc_accuracy",
        primary_chamber_variable="pressure_intake",
        cross_check_variables=[
            "pressure_ambient",
            "pressure_downwind",
            "pressure_upwind",
        ],
    )
    measurement = run(claim)
    print(measurement.model_dump_json(indent=2))
