"""Shared helpers for chamber-side reproducibility protocols.

Centralises the actuator-column tuples and condition-matching that all
protocol modules use, so adding a new component (e.g. Si115x in the light
tunnel) doesn't require copying code from `barometer_dc_accuracy.py`.

`select_steady_state_window`, `cross_sensor_sigma`, and the rest of the
chamber-data math live in `chamber_eval/uncertainty.py`; this module
imports from there but does not duplicate it. The reason this isn't all
in one file: the actuator constants and condition-matching are
protocol-shaped (they reference `ClaimSpec`), while the uncertainty
helpers are protocol-agnostic and useful from any code path that touches
chamber dataframes.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from chamberbench.claims import (
        ChamberMeasurement,
        ClaimSpec,
        OperatingCondition,
    )


# Wind-tunnel actuator columns (set by the user / experiment config). When
# all of these are constant the chamber is in actuator equilibrium, which
# is a necessary condition for sensor equilibrium across the redundant
# DPS310 barometers.
WT_ACTUATORS: tuple[str, ...] = (
    "hatch",
    "load_in",
    "load_out",
    "v_in",
    "v_out",
    "v_mic",
)

# Light-tunnel actuator columns. Same idea: when these are constant the
# light tunnel is in actuator equilibrium and the redundant Si115x units
# should be reading the same illumination. Pulled from `lt_validate_v1`'s
# column set (`red`/`green`/`blue` are the LED PWM inputs; `pol_*` are
# the polariser angle setpoints; `osr_*` and `v_*` are configuration
# voltages and oversampling-rate selects).
LT_ACTUATORS: tuple[str, ...] = (
    "red",
    "green",
    "blue",
    "v_c",
    "osr_c",
    "pol_1",
    "pol_2",
    "v_angle_1",
    "v_angle_2",
    "osr_angle_1",
    "osr_angle_2",
)


# Float comparison tolerance for value-matching against recorded chamber
# data. Recorded actuator values can have tiny float jitter (e.g.
# 25.0000001 for a nominally-25.0 setpoint). Relative tolerance dominates
# at large magnitudes; absolute tolerance dominates near zero.
_VALUE_MATCH_RTOL = 1e-3
_VALUE_MATCH_ATOL = 1e-6


def match_conditions(
    claim: ClaimSpec,
    df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Decide which of `claim.operating_conditions` the dataframe satisfies.

    A condition is *matched* iff:
      1. It names a `chamber_variable` that exists as a column in `df`.
      2. Either the claim places no value/range on that variable, OR the
         column's recorded values lie inside the claim's stated value /
         min_value / max_value window for the entire frame.

    Why this is tighter than the original implementation in
    `barometer_dc_accuracy._match_conditions` (which only checked column
    existence): once we annotate temperature-conditioned claims (Si115x
    has several), accepting a column whose values actually lie outside the
    stated window would silently report `matched_conditions` while the
    chamber is in the wrong regime. That leads `verdict()` to compute
    pass/fail on a misaligned measurement.

    The behaviour change for the existing 11 DPS310 claims is contained:
    only non-load-bearing temperature conditions move from "matched" to
    "unmatched" (the chamber doesn't actively control temperature, so
    `res_in` doesn't sit at exactly 25 °C). Verdicts are unchanged because
    `verdict()` keys on load-bearing conditions only.

    Returns (matched, unmatched). Order matches `claim.operating_conditions`.
    """
    matched: list[str] = []
    unmatched: list[str] = []
    for cond in claim.operating_conditions:
        if not cond.chamber_variable:
            unmatched.append(cond.name)
            continue
        if cond.chamber_variable not in df.columns:
            unmatched.append(cond.name)
            continue
        if not _column_in_window(df[cond.chamber_variable], cond):
            unmatched.append(cond.name)
            continue
        matched.append(cond.name)
    return matched, unmatched


def make_stub_measurement(
    claim: ClaimSpec,
    unmatched_load_bearing_names: list[str] | None = None,
    *,
    reason: str = "",
) -> ChamberMeasurement:
    """Build the NaN-flagged stub measurement that protocol modules return
    when they cannot compute a meaningful value.

    Two stub kinds are supported via the two keyword arguments:

    * ``unmatched_load_bearing_names`` -- the original case: at least one
      load-bearing operating condition has no `chamber_variable` mapping,
      so the chamber cannot exercise the stated regime. Notes string is
      built from the names list.
    * ``reason`` -- an explicit free-text reason for protocols that
      short-circuit for non-condition reasons (e.g. claim_kind not yet
      implemented, degenerate input, missing required column). Notes
      string is the raw reason.

    Exactly one of the two must be provided. The lists
    `matched_conditions` / `unmatched_conditions` are always the
    structural partition over `claim.operating_conditions` -- we
    deliberately do not load the chamber dataset because the verdict is
    already determined.

    `measured_sigma_basis="stub"` is the marker that
    `quality_gates.H5` and `verdict()` both key on. H5 uses it to
    distinguish expected-NaN from a NaN bug in the full-run path;
    `verdict()` short-circuits to `inconclusive` regardless of the
    downstream comparison logic.
    """
    from chamberbench.claims import ChamberMeasurement

    if (unmatched_load_bearing_names is None) == (not reason):
        raise ValueError(
            "make_stub_measurement requires exactly one of unmatched_load_bearing_names or reason"
        )

    matched = [c.name for c in claim.operating_conditions if c.chamber_variable]
    unmatched = [c.name for c in claim.operating_conditions if not c.chamber_variable]
    if unmatched_load_bearing_names is not None:
        notes = (
            "Short-circuit: load-bearing condition(s) "
            + ", ".join(unmatched_load_bearing_names)
            + " unmatchable by the chamber. measured_value=NaN by design; "
            "verdict() emits inconclusive without consuming the value."
        )
    else:
        notes = f"Short-circuit: {reason}. measured_value=NaN by design."

    return ChamberMeasurement(
        claim_id=claim.id,
        experiment_ids=[],
        measured_value=float("nan"),
        measured_unit=claim.expected_unit,
        measured_sigma=float("nan"),
        measured_sigma_basis="stub",
        matched_conditions=matched,
        unmatched_conditions=unmatched,
        sample_n=0,
        cross_sensor_spread=float("nan"),
        notes=notes,
    )


def resolve_cache_root() -> Path:
    """Default chamber-data download location; CHAMBER_CACHE_ROOT env overrides.

    Lifted from per-protocol modules so a third protocol on Day 11+ does
    not introduce yet another copy.
    """
    import os

    return Path(os.environ.get("CHAMBER_CACHE_ROOT", "/tmp/cc_data"))


def unmatched_load_bearing(claim: ClaimSpec) -> list[str]:
    """Return load-bearing condition names that have no chamber_variable.

    A non-empty result means the protocol's `run()` should short-circuit
    via `make_stub_measurement` -- the chamber cannot exercise the
    stated regime regardless of which experiment is loaded.
    """
    return [
        c.name
        for c in claim.operating_conditions
        if c.load_bearing and not c.chamber_variable
    ]


def _column_in_window(column: pd.Series, cond: OperatingCondition) -> bool:
    """True iff every recorded value lies in the condition's stated window.

    If the condition has no value/range constraints (chamber records the
    variable but the claim doesn't pin it to a value), the column matches
    trivially -- structural existence is enough, even on an empty frame.
    Otherwise an empty frame is unmatched: we cannot verify a constraint
    against zero rows.
    """
    has_constraint = cond.has_value() or cond.has_min() or cond.has_max()

    if not has_constraint:
        # Structural-existence check passed by the time we got here
        # (the column exists). No further verification possible or
        # needed. An empty frame still satisfies structural existence.
        return True

    try:
        arr = column.to_numpy(dtype=float)
    except (TypeError, ValueError):
        # Non-numeric column (string flags, categorical config) with a
        # numeric constraint: we cannot verify, so unmatched.
        return False

    if arr.size == 0:
        return False

    if cond.has_value():
        target = float(cond.value)
        tol = max(_VALUE_MATCH_ATOL, abs(target) * _VALUE_MATCH_RTOL)
        if not bool(((arr >= target - tol) & (arr <= target + tol)).all()):
            return False
    # Range bounds use atol only: min/max are stated operating ranges (not
    # setpoints), so recorded values should sit strictly inside the range,
    # and there is no large-magnitude actuator jitter at the boundary that
    # rtol would need to absorb. The atol just defends against numerical
    # noise when a recorded value sits exactly at the bound.
    if cond.has_min() and not bool(
        (arr >= float(cond.min_value) - _VALUE_MATCH_ATOL).all()
    ):
        return False
    return not (
        cond.has_max()
        and not bool((arr <= float(cond.max_value) + _VALUE_MATCH_ATOL).all())
    )
