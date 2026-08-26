"""Chamber-side uncertainty estimation.

Honest about what it measures: cross-sensor agreement, not NIST-traceable
absolute accuracy. The chamber's four pressure sensors (`pressure_upwind`,
`pressure_downwind`, `pressure_ambient`, `pressure_intake`) sit at different
physical positions and only read the same value when the chamber is in
pressure equilibrium (e.g. fans off, hatch closed). When that holds, the
spread among them bounds the *relative* readout disagreement -- a useful
lower bound on chamber-side uncertainty for benchmark purposes, but not a
calibrated absolute reference.

This module is intentionally pure (no LLM, no agent dependency).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd


def cross_sensor_sigma(
    df: pd.DataFrame,
    primary: str,
    redundants: Sequence[str],
) -> float:
    """Bound on chamber-side readout uncertainty for `primary`.

    Defined as the standard deviation of (primary - mean(redundants)) across
    all rows in the supplied dataframe. The caller is responsible for choosing
    a dataframe slice where all sensors *should* be reading the same physical
    quantity (steady-state equilibrium). Otherwise this overestimates the
    true sensor-side noise because it includes physical position differences.

    Returns 0.0 if no redundants are available.
    """
    if not redundants:
        return 0.0
    primary_arr = df[primary].to_numpy(dtype=float)
    others = df[list(redundants)].to_numpy(dtype=float)
    others_mean = others.mean(axis=1)
    diff = primary_arr - others_mean
    return float(diff.std(ddof=1)) if len(diff) > 1 else 0.0


def pairwise_spread(df: pd.DataFrame, sensors: Sequence[str]) -> float:
    """Mean row-wise standard deviation across all listed sensors.

    Different metric from `cross_sensor_sigma`: this is the average of
    per-row stdev across sensors. Higher than cross_sensor_sigma when the
    sensors disagree systematically; useful as a sanity check.
    """
    arr = df[list(sensors)].to_numpy(dtype=float)
    if arr.shape[1] < 2:
        return 0.0
    return float(arr.std(axis=1, ddof=1).mean())


def combined_uncertainty(
    chamber_sigma: float,
    spec_tolerance: float,
    agent_sigma: float = 0.0,
) -> float:
    """Quadrature combination for the verdict logic.

    The reproducibility verdict declares "inconclusive" when the measured
    delta is larger than `spec_tolerance` but smaller than this combined
    bound. Using quadrature (rather than linear sum) reflects independence
    of the three error sources; in practice they may be correlated, so we
    treat this as a reasonable best-effort bound, not a strict CI.
    """
    return math.sqrt(chamber_sigma**2 + spec_tolerance**2 + agent_sigma**2)


def select_steady_state_window(
    df: pd.DataFrame,
    actuator_cols: Sequence[str],
    min_samples: int = 50,
) -> pd.DataFrame:
    """Return rows where all listed actuators are constant within their experiment.

    Useful for finding equilibrium windows in datasets that include both
    transient and steady-state segments. Returns the input unchanged if the
    actuators are already constant or no qualifying window exists.
    """
    if not actuator_cols or len(df) < min_samples:
        return df
    # An actuator is "constant enough" if its std is below a small fraction
    # of its dynamic range over the whole experiment.
    is_steady = np.ones(len(df), dtype=bool)
    for col in actuator_cols:
        arr = df[col].to_numpy(dtype=float)
        rng = arr.max() - arr.min()
        if rng <= 0:
            continue
        # Mark rows where the local 5-sample window has small relative variation.
        local = pd.Series(arr).rolling(window=5, min_periods=1).std().to_numpy()
        threshold = max(rng * 0.01, 1e-6)
        is_steady &= local < threshold
    if is_steady.sum() < min_samples:
        return df
    return df.loc[is_steady]


if __name__ == "__main__":
    # Standalone sanity check: run against wt_validate_v1 and report the
    # chamber-side pressure sigma for a few candidate experiments.
    from causalchamber.datasets import Dataset

    PRIMARY = "pressure_intake"
    REDUNDANTS = ("pressure_ambient", "pressure_downwind", "pressure_upwind")
    ds = Dataset("wt_validate_v1", root="/tmp/cc_data", download=True)
    print(f"\nChamber-side sigma estimates ({PRIMARY} vs mean of {REDUNDANTS}):")
    print(f"{'experiment':<40s}  {'rows':>6s}  {'sigma_Pa':>9s}  {'spread_Pa':>10s}")
    for name in ds.available_experiments():
        df = ds.get_experiment(name).as_pandas_dataframe()
        if PRIMARY not in df.columns:
            continue
        sigma = cross_sensor_sigma(df, PRIMARY, REDUNDANTS)
        spread = pairwise_spread(df, [PRIMARY, *REDUNDANTS])
        print(f"{name:<40s}  {len(df):>6d}  {sigma:>9.2f}  {spread:>10.2f}")
