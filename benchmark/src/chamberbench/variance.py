"""Offline aggregation of repeated chamber runs.

Extracted verbatim from the live-run driver, which ships here as
`scripts/variance.py`. The aggregation itself is pure -- it reduces archived
per-(model, repeat, claim) cells to the per-model mean +/- std that the
paper's model-comparison table reports -- so it lives in the package, where it
is importable without the harness extra the live driver needs.

`aggregate_variance` is the entry point; the rest are its helpers.
"""

from __future__ import annotations

import statistics
from typing import Any

_ZERO_USAGE = {
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_read_tokens": 0,
    "cache_creation_tokens": 0,
}


# ---------------------------------------------------------------------------
# Aggregation (pure)
# ---------------------------------------------------------------------------


def _cell_passed(cell: dict[str, Any]) -> bool:
    """Effective fidelity verdict of a variance cell.

    An engine error is a fail regardless of the cell's fidelity dict --
    an imported cell may carry a stale `overall_pass` next to a populated
    `engine_error`.
    """
    if cell.get("engine_error"):
        return False
    return bool(cell.get("fidelity", {}).get("overall_pass"))


def _stats(per_run: list[float | None]) -> dict[str, Any]:
    """mean / sample-std over the per-run values, skipping None.

    `std` is None when fewer than two repeats have a usable value
    (statistics.stdev needs n>=2); `mean` is None when none do.
    """
    usable = [v for v in per_run if v is not None]
    return {
        "per_run": per_run,
        "mean": statistics.mean(usable) if usable else None,
        "std": statistics.stdev(usable) if len(usable) >= 2 else None,
    }


def _claim_stability(repeats: list[dict[str, Any]]) -> dict[str, Any]:
    """Per claim, did the fidelity verdict agree across every repeat."""
    claim_ids: list[str] = []
    seen: set[str] = set()
    for rep in repeats:
        for cid in rep.get("cells", {}):
            if cid not in seen:
                seen.add(cid)
                claim_ids.append(cid)

    stable = 0
    flipped_claims: list[dict[str, Any]] = []
    for cid in claim_ids:
        pattern = [
            _cell_passed(rep["cells"][cid])
            for rep in repeats
            if cid in rep.get("cells", {})
        ]
        if len(set(pattern)) <= 1:
            stable += 1
        else:
            flipped_claims.append({"id": cid, "pattern": pattern})
    return {
        "stable": stable,
        "flipped": len(flipped_claims),
        "flipped_claims": flipped_claims,
    }


def aggregate_variance(
    runs: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Reduce per-(model, repeat, claim) cells to per-model mean +/- std.

    `runs` maps model -> list of repeat dicts, each
    `{"repeat", "source", "started", "cells": {claim_id: cell}}`.
    """
    agg: dict[str, dict[str, Any]] = {}
    for model, repeats in runs.items():
        fid_per_run: list[float | None] = []
        conf_per_run: list[float | None] = []
        lat_per_run: list[float | None] = []
        err_per_run: list[int] = []
        for rep in repeats:
            cells = rep.get("cells", {})
            fid_per_run.append(sum(1 for c in cells.values() if _cell_passed(c)))
            ok = [c for c in cells.values() if not c.get("engine_error")]
            conf_per_run.append(
                statistics.mean(c["fidelity"]["confidence"] for c in ok) if ok else None
            )
            lat_per_run.append(
                statistics.mean(c["latency_s"] for c in ok) if ok else None
            )
            err_per_run.append(sum(1 for c in cells.values() if c.get("engine_error")))
        agg[model] = {
            "fidelity": _stats(fid_per_run),
            "confidence": _stats(conf_per_run),
            "latency_s": _stats(lat_per_run),
            "engine_errors": {"per_run": err_per_run, "total": sum(err_per_run)},
            "claim_stability": _claim_stability(repeats),
        }
    return agg


# ---------------------------------------------------------------------------
# Repeat-1 import (pure)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Importing the first repeat from the baseline run
# ---------------------------------------------------------------------------


def _failed_cell(engine_error: str) -> dict[str, Any]:
    """A variance cell standing in for a run that produced no result."""
    return {
        "fidelity": {
            "found_expected": True,
            "found_actual": False,
            "found_correct": False,
            "value_pass": False,
            "confidence": 0.0,
            "failure_reason": engine_error,
            "overall_pass": False,
            # Shape parity with the pytest harness's engine-error fidelity
            # dict (test_chamber.py); the cell-level engine_error string
            # is the field consumers actually read.
            "engine_error": True,
        },
        "latency_s": 0.0,
        "usage": dict(_ZERO_USAGE),
        "engine_error": engine_error,
        "n_tool_calls_by_tool": {},
        "n_steps": 0,
    }


def _project_baseline_cell(
    raw: dict[str, Any] | None, claim_id: str, model: str
) -> dict[str, Any]:
    """Project a baseline_chamber.json agentic cell into the variance shape.

    A missing cell, a non-`ok` consolidation status, or a `pending_rerun`
    stub (which carries no `fidelity`) becomes a failed repeat. An `ok`
    cell that carries a populated `engine_error` -- the Qwen
    dps310-meas-time-standard case -- is carried through unchanged, so it
    still counts as a failed repeat downstream.
    """
    if raw is None:
        return _failed_cell(
            f"(claim, model)=({claim_id}, {model}) missing from baseline_chamber.json"
        )
    status = raw.get("status", "ok")
    if status != "ok" or "fidelity" not in raw:
        reason = (
            raw.get("not_applicable_reason") or raw.get("reason") or f"status={status}"
        )
        return _failed_cell(f"imported repeat 1 not runnable: {reason}")
    return {
        "fidelity": raw["fidelity"],
        "latency_s": raw.get("latency_s", 0.0),
        "usage": raw.get("usage") or dict(_ZERO_USAGE),
        "engine_error": raw.get("engine_error", ""),
        "n_tool_calls_by_tool": raw.get("n_tool_calls_by_tool") or {},
        "n_steps": raw.get("n_steps", 0),
    }


def import_repeat_one(
    baseline: dict[str, Any],
    models: list[str] | tuple[str, ...],
    claim_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Build repeat 1 for each model from the existing baseline run.

    Returns `{model: repeat_dict}`; each repeat_dict is the
    `{"repeat", "source", "started", "cells"}` shape the runner appends
    fresh repeats to.
    """
    timestamp = baseline.get("timestamp", "")
    results = baseline.get("results", {})
    out: dict[str, dict[str, Any]] = {}
    for model in models:
        cells = {
            cid: _project_baseline_cell(
                results.get(cid, {}).get("agentic", {}).get(model), cid, model
            )
            for cid in claim_ids
        }
        out[model] = {
            "repeat": 1,
            "source": "imported:baseline_chamber.json",
            "started": timestamp,
            "cells": cells,
        }
    return out
