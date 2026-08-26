"""Chamber-side tools exposed to the agent during the chamber benchmark.

Closure-bound `@beta_async_tool` factory mirroring `_make_large_pdf_tools`
in `datasheet_tools.py`: a single chamber dataset (and optional simulator
configuration) is bound at construction time, and the agent can call any
of the resulting tools to query it.

All tool outputs are short JSON or compact tables so the agent's input
budget stays small even when many calls happen in one turn.

The `causalchamber` package is sync; we bridge with `asyncio.to_thread`
exactly like the datasheetindex side.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from anthropic import beta_async_tool

logger = logging.getLogger(__name__)


# Wind-tunnel simulator default parameters. Pulled from the published
# Causal Chamber paper (Gamella et al. 2025) appendix IV / V; these are
# rough working defaults so the agent can call run_simulator without
# specifying calibration constants. Users with stricter needs can pass
# `parameters` to override.
_WT_SIMULATOR_DEFAULTS: dict[str, dict[str, float]] = {
    # ModelA1: steady-state fan speed from load. omega_max in rpm; L_min
    # is the dead-zone load below which the fan does not spin.
    "ModelA1": {"L_min": 0.10, "omega_max": 4500.0},
    # ModelB1: fan steady-state current from load.
    "ModelB1": {"L_min": 0.10, "C_max": 0.450},
}

WT_ACTUATOR_COLS = (
    "hatch",
    "load_in",
    "load_out",
    "v_in",
    "v_out",
    "v_mic",
    "v_1",
    "v_2",
    "pot_1",
    "pot_2",
    "osr_in",
    "osr_out",
    "osr_mic",
    "osr_upwind",
    "osr_downwind",
    "osr_ambient",
    "osr_intake",
    "osr_1",
    "osr_2",
)


def _resolve_cache_root() -> Path:
    return Path(os.environ.get("CHAMBER_CACHE_ROOT", "/tmp/cc_data"))


def _make_chamber_tools(
    chamber: Literal["wt", "lt"] = "wt",
    config: str = "standard",
    dataset_name: str = "wt_validate_v1",
    cache_root: Path | None = None,
) -> tuple[list, Callable[[], None]]:
    """Build the chamber-side tool list bound to one dataset configuration.

    Returns (tools_list, cleanup_fn). Cleanup is a no-op today (datasets are
    file-backed and cheap to drop) but the signature matches
    `_make_large_pdf_tools` for symmetry; the manual agentic loop calls
    both cleanups in its `finally` block.

    Tool surface (six tools):
      - list_experiments  -> compact table of all experiments in the dataset
      - get_experiment_metadata(experiment_id) -> per-experiment summary
      - query_dataset(experiment_id, variables, statistic, time_window_s) ->
            JSON statistic per variable
      - cross_sensor_check(experiment_id, variables, time_window_s) ->
            inter-channel offset and pairwise spread
      - run_simulator(model, inputs, parameters?) -> simulator outputs
      - get_ground_truth_graph -> chamber DAG as adjacency list
    """
    cache_root = cache_root or _resolve_cache_root()
    state: dict[str, Any] = {"dataset": None}

    def _get_dataset():
        if state["dataset"] is None:
            from causalchamber.datasets import Dataset

            cache_root.mkdir(parents=True, exist_ok=True)
            state["dataset"] = Dataset(
                dataset_name, root=str(cache_root), download=True
            )
        return state["dataset"]

    def _get_experiment_df(experiment_id: str):
        ds = _get_dataset()
        return ds.get_experiment(experiment_id).as_pandas_dataframe()

    @beta_async_tool
    async def list_experiments() -> str:
        """List all experiments in the bound chamber dataset.

        Use this first to discover which experiments may be relevant to a
        claim. The output table includes the experiment id, row count, and
        which sensors and actuators it exercises (limited to commonly used
        variables for compactness).

        Returns a small markdown-style table.
        """
        ds = await asyncio.to_thread(_get_dataset)
        names = ds.available_experiments()

        def _summarize_one(name: str) -> dict[str, Any]:
            df = ds.get_experiment(name).as_pandas_dataframe()
            actuator_cols = [c for c in df.columns if c in WT_ACTUATOR_COLS]
            varied = [c for c in actuator_cols if df[c].nunique() > 1]
            return {
                "id": name,
                "rows": len(df),
                "varied_actuators": ",".join(varied),
            }

        rows = await asyncio.to_thread(lambda: [_summarize_one(n) for n in names])
        lines = [
            f"Dataset: {dataset_name} (chamber={chamber}, config={config})",
            f"{len(rows)} experiments:",
            f"{'id':<40s} {'rows':>6s}  varied_actuators",
        ]
        for r in rows:
            lines.append(f"{r['id']:<40s} {r['rows']:>6d}  {r['varied_actuators']}")
        return "\n".join(lines)

    @beta_async_tool
    async def get_experiment_metadata(experiment_id: str) -> str:
        """Return per-variable summary stats for one experiment.

        Args:
            experiment_id: The experiment id (see list_experiments).

        Returns JSON: per-variable {mean, std, min, max, n_samples}.
        """
        df = await asyncio.to_thread(_get_experiment_df, experiment_id)
        out: dict[str, Any] = {
            "experiment_id": experiment_id,
            "n_rows": len(df),
            "columns": list(df.columns),
            "stats": {},
        }
        for col in df.columns:
            try:
                arr = df[col].to_numpy(dtype=float)
            except (TypeError, ValueError):
                continue
            if arr.size == 0:
                continue
            out["stats"][col] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                "min": float(arr.min()),
                "max": float(arr.max()),
                "n_samples": int(arr.size),
            }
        return json.dumps(out)

    @beta_async_tool
    async def query_dataset(
        experiment_id: str,
        variables: list[str],
        statistic: Literal[
            "mean", "median", "std", "p05", "p95", "min", "max", "n"
        ] = "mean",
        time_window_s: list[float] | None = None,
    ) -> str:
        """Compute a summary statistic of variables in one experiment.

        Args:
            experiment_id: The experiment id (see list_experiments).
            variables: Column names to query.
            statistic: Aggregation per variable.
            time_window_s: Optional [t_lo, t_hi] window in seconds against
                the experiment's `timestamp` column. If None, use the full
                experiment.

        Returns JSON: {variable: value} plus n_samples and sample_std for
        each variable.
        """
        df = await asyncio.to_thread(_get_experiment_df, experiment_id)
        df = _apply_time_window(df, time_window_s)
        result: dict[str, Any] = {
            "experiment_id": experiment_id,
            "statistic": statistic,
            "n_samples": len(df),
            "values": {},
        }
        for var in variables:
            if var not in df.columns:
                raise ValueError(
                    f"unknown variable {var!r}; available: {sorted(df.columns)[:8]}..."
                )
            try:
                arr = df[var].to_numpy(dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"non-numeric variable {var!r}: {exc}") from exc
            value = _compute_statistic(arr, statistic)
            sample_std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
            result["values"][var] = {
                "value": value,
                "sample_std": sample_std,
                "n": int(arr.size),
            }
        return json.dumps(result)

    @beta_async_tool
    async def cross_sensor_check(
        experiment_id: str,
        variables: list[str],
        time_window_s: list[float] | None = None,
    ) -> str:
        """Compute inter-channel agreement among redundant sensors.

        Use this when the claim is about inter-unit consistency or relative
        accuracy. Returns the mean of each variable, the pairwise standard
        deviation across the listed channels (per-row), and the std of
        (variable[0] - mean(variable[1:])) which the benchmark uses as a
        chamber-side sigma estimate.

        Args:
            experiment_id: The experiment id (see list_experiments).
            variables: Two or more column names of redundant sensors.
            time_window_s: Optional [t_lo, t_hi] in seconds.

        Returns JSON.
        """
        if len(variables) < 2:
            raise ValueError("cross_sensor_check requires at least 2 variables")

        df = await asyncio.to_thread(_get_experiment_df, experiment_id)
        df = _apply_time_window(df, time_window_s)
        for v in variables:
            if v not in df.columns:
                raise ValueError(
                    f"unknown variable {v!r}; available: {sorted(df.columns)[:8]}..."
                )

        import numpy as np

        arrs = [df[v].to_numpy(dtype=float) for v in variables]
        primary = arrs[0]
        others = np.column_stack(arrs[1:])
        others_mean = others.mean(axis=1)
        diff = primary - others_mean
        pair_spread = (
            float(np.column_stack(arrs).std(axis=1, ddof=1).mean())
            if all(len(a) > 1 for a in arrs)
            else 0.0
        )
        return json.dumps(
            {
                "experiment_id": experiment_id,
                "n_samples": len(primary),
                "means": {v: float(arrs[i].mean()) for i, v in enumerate(variables)},
                "primary": variables[0],
                "redundants": variables[1:],
                "primary_minus_mean": {
                    "mean": float(diff.mean()),
                    "std": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
                },
                "pairwise_row_spread_mean": pair_spread,
            }
        )

    @beta_async_tool
    async def run_simulator(
        model: str,
        inputs: dict[str, float],
        parameters: dict[str, float] | None = None,
    ) -> str:
        """Run a wind-tunnel simulator from the published causal chamber package.

        Args:
            model: Simulator class name. Supported: "ModelA1" (fan speed
                from load), "ModelB1" (fan current from load).
            inputs: Dict of input variables required by the model. See
                ModelA1.inputs_names = ["load"], ModelB1.inputs_names =
                ["load"].
            parameters: Optional override for the simulator's calibration
                constants. Default values are baked in.

        Returns JSON {output_name: value}.
        """
        from causalchamber.simulators import wt as wt_sim

        params = dict(_WT_SIMULATOR_DEFAULTS.get(model, {}))
        if parameters:
            params.update(parameters)

        cls = getattr(wt_sim, model, None)
        if cls is None:
            raise ValueError(
                f"unknown simulator model {model!r}; supported: ModelA1, ModelB1"
            )

        try:
            sim = cls(**params)
        except TypeError as exc:
            raise ValueError(f"missing or extra parameter for {model}: {exc}") from exc

        import pandas as pd

        df = pd.DataFrame({k: [v] for k, v in inputs.items()})
        outputs = await asyncio.to_thread(sim.simulate_from_inputs, df)

        # Outputs may be a numpy array or scalar. Pair with outputs_names.
        names = list(getattr(cls, "outputs_names", []))
        if hasattr(outputs, "__iter__") and not isinstance(outputs, (str, bytes)):
            try:
                values = [float(x) for x in outputs]
            except TypeError:
                values = [float(outputs)]
        else:
            values = [float(outputs)]
        result = {
            n: values[i] if i < len(values) else None for i, n in enumerate(names)
        }
        return json.dumps({"model": model, "outputs": result, "parameters": params})

    @beta_async_tool
    async def get_ground_truth_graph() -> str:
        """Return the chamber's published causal graph for this configuration.

        Use this to reason about which actuators causally drive which
        sensors before designing a query. Output is a compact adjacency
        list `from -> [to1, to2, ...]`.
        """
        # causalchamber 0.2.x ships these inside the `.main` submodule; the
        # package's `ground_truth/__init__.py` doesn't re-export them, so
        # the bare `from causalchamber.ground_truth import edges, variables`
        # raises ImportError. Importing from `.main` directly is the
        # stable path across the 0.2.x line and matches how `graph` is
        # imported earlier in this module.
        from causalchamber.ground_truth.main import edges, variables

        nodes = await asyncio.to_thread(variables, chamber, config)
        eds = await asyncio.to_thread(edges, chamber, config)
        adj: dict[str, list[str]] = {n: [] for n in nodes}
        for fro, to in eds:
            adj.setdefault(fro, []).append(to)
        lines = [
            f"Causal graph ({chamber}/{config}): {len(nodes)} nodes, {len(eds)} edges"
        ]
        for fro in sorted(adj):
            tos = adj[fro]
            if tos:
                lines.append(f"  {fro} -> {','.join(sorted(tos))}")
        return "\n".join(lines)

    tools_list = [
        list_experiments,
        get_experiment_metadata,
        query_dataset,
        cross_sensor_check,
        run_simulator,
        get_ground_truth_graph,
    ]

    def _cleanup() -> None:
        # Datasets are file-backed CSVs; nothing to release. Kept for
        # symmetry with _make_large_pdf_tools.
        state["dataset"] = None

    return tools_list, _cleanup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apply_time_window(df, time_window_s: Sequence[float] | None):
    if not time_window_s or "timestamp" not in df.columns:
        return df
    if len(time_window_s) != 2:
        return df
    lo, hi = float(time_window_s[0]), float(time_window_s[1])
    mask = (df["timestamp"] >= lo) & (df["timestamp"] <= hi)
    return df.loc[mask]


def _compute_statistic(arr, statistic: str) -> float:
    import numpy as np

    if statistic == "mean":
        return float(arr.mean())
    if statistic == "median":
        return float(np.median(arr))
    if statistic == "std":
        return float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    if statistic == "p05":
        return float(np.percentile(arr, 5))
    if statistic == "p95":
        return float(np.percentile(arr, 95))
    if statistic == "min":
        return float(arr.min())
    if statistic == "max":
        return float(arr.max())
    if statistic == "n":
        return float(arr.size)
    raise ValueError(f"unknown statistic: {statistic}")
