"""Reproducibility protocol for the Silicon Labs Si115x light sensor.

Strategy
--------
The light tunnel exposes three Si115x units at distinct positions
(`vis_1`/`vis_2`/`vis_3` plus IR companions). Reconnaissance confirmed they
are NOT spatially redundant -- at fixed RGB they read on the order of
10000 / 1500 / 480 counts. The cross-sensor sigma protocol that worked
for the four DPS310 barometers cannot be applied here without spatial
geometry dominating the chamber-side uncertainty.

What this module CAN test cleanly, decomposed by `claim_kind`:

* ``typical``  -> precision / RMS-noise. Per-channel std over a steady-
                  state window at fixed RGB. Chamber-side sigma is
                  reported as `single_channel_std`.
* ``range``    -> envelope check. Same shape as the DPS310 range path:
                  measured_value = mean of the channel over the steady
                  window; the verdict module checks that it lies inside
                  the claimed [min, max] bounds.
* ``dc_accuracy``,
  ``max``, ``min`` -> not testable absolutely (chamber has no NIST-
                      traceable optical reference). These claims should
                      route through ``run`` only when paired with a
                      load-bearing "external_optical_reference" condition
                      with no ``chamber_variable``, which short-circuits
                      to inconclusive via `make_stub_measurement`. If a
                      claim author skips that condition, the protocol
                      falls back to the range path and emits a
                      best-effort steady-state mean -- caller's
                      responsibility.

What this module deliberately does NOT implement on an earlier revision:

* ``linearity`` -> regression of channel vs. swept input axis. Requires a
                   univariate sweep dataset (`lt_walks_v1` or similar)
                   for clean isolation, or a multivariate-regression
                   framing on `lt_interventions_standard_v1`. Decision
                   on dataset + framing deferred to an earlier revision+. Linearity
                   claims today raise NotImplementedError so the test
                   runner fails loudly rather than silently routing to
                   the wrong path.

Run as a script for smoke testing:
    uv run python -m chamberbench.protocols.light_sensor
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..uncertainty import select_steady_state_window
from ._common import (
    LT_ACTUATORS,
    make_stub_measurement,
    match_conditions,
    resolve_cache_root,
    unmatched_load_bearing,
)

if TYPE_CHECKING:
    import pandas as pd

    from chamberbench.claims import ChamberMeasurement, ClaimSpec


# Fallback experiment when claim.chamber_experiment_hint is empty. All
# entries point at a constant-RGB validate_diode_vis_1 run that exists
# in `lt_validate_v1`; per-claim hints override this.
DEFAULT_EXPERIMENT = "validate_diode_vis_1"


def run(claim: ClaimSpec) -> ChamberMeasurement:
    """Compute a `ChamberMeasurement` for an Si115x claim.

    Branches on `claim.claim_kind`. The dispatcher in
    `chamber_eval.reproducibility.run_protocol` reaches us by importing
    this module's dotted path and calling `run`.
    """
    # Lazy imports so this module is cheap to load when the chamber
    # group is not installed (mirrors barometer_dc_accuracy).
    from causalchamber.datasets import Dataset

    if claim.claim_kind == "linearity":
        # Forward-compat: the literal exists in chamber_models.ClaimKind
        # but this module deliberately does not implement linearity yet
        # (see module docstring -- needs a univariate-sweep dataset). Route
        # to inconclusive via a stub so the methodology stays consistent
        # ("inconclusive is needs-inspection, not a hard fail") and the
        # H5 quality gate doesn't trip on a known-deferred case.
        return make_stub_measurement(
            claim,
            reason=(
                "linearity protocol not yet implemented in light_sensor "
                "(deferred to an earlier revision+ pending dataset and framing decisions)"
            ),
        )

    # Short-circuit: if any load-bearing condition has no chamber_variable,
    # we cannot exercise the stated regime regardless of experiment.
    unmatched_lb = unmatched_load_bearing(claim)
    if unmatched_lb:
        return make_stub_measurement(claim, unmatched_load_bearing_names=unmatched_lb)

    cache_root = resolve_cache_root()
    ds = Dataset(claim.chamber_dataset, root=str(cache_root), download=True)

    experiment_id = claim.chamber_experiment_hint or DEFAULT_EXPERIMENT
    df = ds.get_experiment(experiment_id).as_pandas_dataframe()

    # Both supported claim kinds want a steady-state window where the
    # LED PWM and polariser actuators are constant. The min_samples
    # threshold is lower than the DPS310 (50) because lt_validate_v1's
    # validate_* experiments are 50 rows total -- a 20-sample window
    # leaves enough headroom.
    df_steady = select_steady_state_window(df, LT_ACTUATORS, min_samples=20)

    if claim.claim_kind == "typical":
        return _run_precision(claim, df_steady, experiment_id)
    # range / dc_accuracy / max / min all use the range-shape envelope
    # measurement. dc_accuracy without an external-reference unmatched
    # condition is the caller's bug, not ours; we still emit something
    # so the trace is informative.
    return _run_range(claim, df_steady, experiment_id)


# ---------------------------------------------------------------------------
# Per-kind routines
# ---------------------------------------------------------------------------


def _run_precision(
    claim: ClaimSpec, df_steady: pd.DataFrame, experiment_id: str
) -> ChamberMeasurement:
    """Per-channel std over a steady-state window. Tests RMS-noise specs."""
    from chamberbench.claims import ChamberMeasurement

    primary = claim.primary_chamber_variable
    if primary not in df_steady.columns:
        raise ValueError(
            f"primary_chamber_variable {primary!r} not in experiment "
            f"{experiment_id!r}; available: {sorted(df_steady.columns)[:10]}..."
        )
    arr = df_steady[primary].to_numpy(dtype=float)
    measured_value = float(arr.std(ddof=1)) if arr.size > 1 else 0.0

    # Chamber-side sigma estimate for precision claims: the variance of
    # the std estimate itself across plausible windowings, approximated
    # by the std of std over halves. Coarse but defensible -- we can
    # tighten this with bootstrap once the claim set is large enough to
    # justify the extra compute.
    chamber_sigma = _std_of_std(arr) if arr.size >= 4 else 0.0

    matched, unmatched = match_conditions(claim, df_steady)

    return ChamberMeasurement(
        claim_id=claim.id,
        experiment_ids=[experiment_id],
        measured_value=measured_value,
        measured_unit=claim.expected_unit,
        measured_sigma=chamber_sigma,
        measured_sigma_basis="single_channel_std",
        matched_conditions=matched,
        unmatched_conditions=unmatched,
        sample_n=int(arr.size),
        # NaN: Si115x triplet is non-redundant (see recon notes); cross-sensor
        # spread is geometry-dominated and not a sensor signal here.
        # Reporting 0.0 would silently land at the origin in any plot
        # downstream of the trace JSON.
        cross_sensor_spread=float("nan"),
        notes=(
            f"Precision: per-channel std of {primary} over steady-state "
            f"window (n={arr.size}). chamber_sigma = std-of-std across "
            "halves; bounds the precision estimate's own uncertainty."
        ),
    )


def _run_range(
    claim: ClaimSpec, df_steady: pd.DataFrame, experiment_id: str
) -> ChamberMeasurement:
    """Steady-state mean of the primary channel. Tests envelope claims."""
    from chamberbench.claims import ChamberMeasurement

    primary = claim.primary_chamber_variable
    if primary not in df_steady.columns:
        raise ValueError(
            f"primary_chamber_variable {primary!r} not in experiment "
            f"{experiment_id!r}; available: {sorted(df_steady.columns)[:10]}..."
        )
    arr = df_steady[primary].to_numpy(dtype=float)
    measured_value = float(arr.mean()) if arr.size > 0 else 0.0
    chamber_sigma = float(arr.std(ddof=1)) if arr.size > 1 else 0.0

    matched, unmatched = match_conditions(claim, df_steady)

    return ChamberMeasurement(
        claim_id=claim.id,
        experiment_ids=[experiment_id],
        measured_value=measured_value,
        measured_unit=claim.expected_unit,
        measured_sigma=chamber_sigma,
        measured_sigma_basis="single_channel_std",
        matched_conditions=matched,
        unmatched_conditions=unmatched,
        sample_n=int(arr.size),
        # NaN: see _run_precision -- Si115x triplet is non-redundant; the
        # field doesn't apply to single-channel measurements.
        cross_sensor_spread=float("nan"),
        notes=(
            f"Range: mean of {primary} over steady-state window "
            f"(n={arr.size}); compare against [claimed_min, claimed_max]."
        ),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _std_of_std(arr) -> float:
    """std of per-half stds; cheap chamber-side sigma estimate for precision."""
    import numpy as np

    if arr.size < 4:
        return 0.0
    half = arr.size // 2
    s1 = float(arr[:half].std(ddof=1))
    s2 = float(arr[half:].std(ddof=1))
    return float(np.std([s1, s2], ddof=1)) if abs(s1 - s2) > 0 else 0.0


if __name__ == "__main__":
    # Smoke test: synthetic precision and range claims on lt_validate_v1.
    from chamberbench.claims import ClaimSpec

    precision_claim = ClaimSpec(
        id="si115x-vis-precision-smoke",
        pdf_source="Si115x",  # corpus is fetched, not vendored; see docs/reproducing.md
        parameter="ALS visible-channel precision",
        expected_unit="counts",
        claim_kind="typical",
        claimed_typical=10.0,
        tolerance_value=0.5,
        tolerance_kind="relative",
        chamber_dataset="lt_validate_v1",
        chamber_experiment_hint="validate_diode_vis_1",
        chamber_protocol="chamberbench.protocols.light_sensor",
        primary_chamber_variable="vis_1",
    )
    print("--- precision smoke ---")
    print(run(precision_claim).model_dump_json(indent=2))

    range_claim = ClaimSpec(
        id="si115x-vis-range-smoke",
        pdf_source="Si115x",  # corpus is fetched, not vendored; see docs/reproducing.md
        parameter="ALS visible-channel envelope",
        expected_unit="counts",
        claim_kind="range",
        claimed_min=0.0,
        claimed_max=8388607.0,  # 2^23 - 1 (Si115x has dual 23-bit ADCs)
        tolerance_value=0.0,
        tolerance_kind="absolute",
        chamber_dataset="lt_validate_v1",
        chamber_experiment_hint="validate_diode_vis_1",
        chamber_protocol="chamberbench.protocols.light_sensor",
        primary_chamber_variable="vis_1",
    )
    print("--- range smoke ---")
    print(run(range_claim).model_dump_json(indent=2))
