"""Unit tests for the variance-experiment pure functions.

Fast and non-LLM: exercises aggregate_variance and import_repeat_one
(scripts/variance_experiment.py) on synthetic inputs. The live agentic
path is integration-only, gated behind the 1-cell smoke in the script.
"""

from __future__ import annotations

import statistics

# pyproject sets pythonpath = ["src", "scripts"]; no sys.path surgery needed.
import chamberbench.variance as ve

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _cell(
    *,
    overall_pass: bool = True,
    confidence: float = 0.9,
    latency_s: float = 100.0,
    engine_error: str = "",
) -> dict:
    """A variance cell (the shape stored in variance_chamber.json runs)."""
    return {
        "fidelity": {"overall_pass": overall_pass, "confidence": confidence},
        "latency_s": latency_s,
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
        },
        "engine_error": engine_error,
        "n_tool_calls_by_tool": {},
        "n_steps": 0,
    }


def _repeat(repeat: int, cells: dict, source: str = "live") -> dict:
    return {
        "repeat": repeat,
        "source": source,
        "started": "2026-05-22T00:00:00Z",
        "cells": cells,
    }


def _baseline_cell(
    *,
    status: str = "ok",
    overall_pass: bool = True,
    confidence: float = 0.9,
    latency_s: float = 100.0,
    engine_error: str = "",
    n_steps: int = 5,
) -> dict:
    """An agentic cell as it appears in baseline_chamber.json."""
    return {
        "claim_id": "c",
        "engine": "agentic",
        "status": status,
        "fidelity": {
            "found_expected": True,
            "found_actual": overall_pass,
            "found_correct": overall_pass,
            "value_pass": overall_pass,
            "confidence": confidence,
            "failure_reason": None,
            "overall_pass": overall_pass,
        },
        "latency_s": latency_s,
        "usage": {
            "input_tokens": 111,
            "output_tokens": 22,
            "cache_read_tokens": 3,
            "cache_creation_tokens": 4,
        },
        "engine_error": engine_error,
        "n_tool_calls_by_tool": {"search_text": 2},
        "n_steps": n_steps,
        # Fields the projection must drop:
        "claim_result": {"extracted": {"confidence": confidence}},
        "reproducibility": {"verdict": "inconclusive"},
        "measurement": {},
    }


# ---------------------------------------------------------------------------
# aggregate_variance
# ---------------------------------------------------------------------------


def test_fidelity_per_run_counts_passes():
    runs = {
        "m": [
            _repeat(1, {"a": _cell(overall_pass=True), "b": _cell(overall_pass=True)}),
            _repeat(2, {"a": _cell(overall_pass=True), "b": _cell(overall_pass=False)}),
            _repeat(3, {"a": _cell(overall_pass=True), "b": _cell(overall_pass=True)}),
        ]
    }
    agg = ve.aggregate_variance(runs)
    assert agg["m"]["fidelity"]["per_run"] == [2, 1, 2]
    assert agg["m"]["fidelity"]["mean"] == statistics.mean([2, 1, 2])
    assert agg["m"]["fidelity"]["std"] == statistics.stdev([2, 1, 2])


def test_engine_error_cell_counts_as_fidelity_fail():
    # A stale fidelity dict still says overall_pass=True; the engine error
    # must override it to a fail.
    runs = {
        "m": [
            _repeat(
                1,
                {
                    "a": _cell(overall_pass=True),
                    "b": _cell(overall_pass=True, engine_error="timeout after 360s"),
                },
            )
        ]
    }
    agg = ve.aggregate_variance(runs)
    assert agg["m"]["fidelity"]["per_run"] == [1]
    assert agg["m"]["engine_errors"]["per_run"] == [1]
    assert agg["m"]["engine_errors"]["total"] == 1


def test_confidence_and_latency_skip_engine_error_cells():
    runs = {
        "m": [
            _repeat(
                1,
                {
                    "a": _cell(confidence=0.9, latency_s=100.0),
                    "b": _cell(confidence=0.5, latency_s=999.0, engine_error="boom"),
                },
            )
        ]
    }
    agg = ve.aggregate_variance(runs)
    assert agg["m"]["confidence"]["per_run"] == [0.9]
    assert agg["m"]["latency_s"]["per_run"] == [100.0]


def test_std_is_null_with_fewer_than_two_repeats():
    runs = {"m": [_repeat(1, {"a": _cell()})]}
    agg = ve.aggregate_variance(runs)
    assert agg["m"]["fidelity"]["std"] is None
    assert agg["m"]["confidence"]["std"] is None


def test_claim_stability_separates_stable_and_flipped():
    runs = {
        "m": [
            _repeat(1, {"a": _cell(overall_pass=True), "b": _cell(overall_pass=True)}),
            _repeat(2, {"a": _cell(overall_pass=True), "b": _cell(overall_pass=True)}),
            _repeat(3, {"a": _cell(overall_pass=True), "b": _cell(overall_pass=False)}),
        ]
    }
    stab = ve.aggregate_variance(runs)["m"]["claim_stability"]
    assert stab["stable"] == 1
    assert stab["flipped"] == 1
    assert stab["flipped_claims"] == [{"id": "b", "pattern": [True, True, False]}]


def test_claim_stability_flip_from_engine_error():
    # An engine error in one repeat is a verdict flip even though the
    # cell's stale fidelity dict says pass.
    runs = {
        "m": [
            _repeat(1, {"a": _cell(overall_pass=True)}),
            _repeat(2, {"a": _cell(overall_pass=True)}),
            _repeat(3, {"a": _cell(overall_pass=True, engine_error="boom")}),
        ]
    }
    stab = ve.aggregate_variance(runs)["m"]["claim_stability"]
    assert stab["flipped"] == 1
    assert stab["flipped_claims"] == [{"id": "a", "pattern": [True, True, False]}]


# ---------------------------------------------------------------------------
# import_repeat_one
# ---------------------------------------------------------------------------


def _baseline(results: dict, timestamp: str = "2026-05-20T00:00:00Z") -> dict:
    return {"timestamp": timestamp, "results": results}


def test_import_projects_ok_cell_to_variance_shape():
    baseline = _baseline(
        {"c1": {"agentic": {"m": _baseline_cell(confidence=0.93, latency_s=88.0)}}}
    )
    out = ve.import_repeat_one(baseline, ["m"], ["c1"])
    rep = out["m"]
    assert rep["repeat"] == 1
    assert rep["source"] == "imported:baseline_chamber.json"
    cell = rep["cells"]["c1"]
    assert set(cell) == {
        "fidelity",
        "latency_s",
        "usage",
        "engine_error",
        "n_tool_calls_by_tool",
        "n_steps",
    }
    assert cell["fidelity"]["confidence"] == 0.93
    assert cell["latency_s"] == 88.0
    assert cell["engine_error"] == ""
    assert cell["usage"]["input_tokens"] == 111


def test_import_carries_through_ok_cell_with_engine_error():
    # The Qwen dps310-meas-time-standard trap: status "ok" but a populated
    # engine_error -- must survive the import as a failed repeat.
    baseline = _baseline(
        {
            "c1": {
                "agentic": {
                    "m": _baseline_cell(
                        status="ok",
                        overall_pass=False,
                        engine_error="turn ended without submit_claim_result",
                    )
                }
            }
        }
    )
    cell = ve.import_repeat_one(baseline, ["m"], ["c1"])["m"]["cells"]["c1"]
    assert cell["engine_error"] == "turn ended without submit_claim_result"


def test_import_missing_cell_becomes_failed_repeat():
    baseline = _baseline({})  # no results at all
    cell = ve.import_repeat_one(baseline, ["m"], ["c1"])["m"]["cells"]["c1"]
    assert cell["engine_error"] != ""
    assert cell["fidelity"]["overall_pass"] is False


def test_import_non_ok_status_becomes_failed_repeat():
    # A pending_rerun stub has no fidelity key; the import must not KeyError.
    baseline = _baseline(
        {
            "c1": {
                "agentic": {
                    "m": {
                        "claim_id": "c1",
                        "engine": "agentic",
                        "status": "pending_rerun",
                        "reason": "gateway under maintenance",
                    }
                }
            }
        }
    )
    cell = ve.import_repeat_one(baseline, ["m"], ["c1"])["m"]["cells"]["c1"]
    assert cell["engine_error"] != ""
    assert cell["fidelity"]["overall_pass"] is False


# ---------------------------------------------------------------------------
# _resolve_concurrency
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _carried_forward_runs (per-model resume / merge)
# ---------------------------------------------------------------------------


# Six tests were removed when this suite was extracted: they covered
# `_resolve_concurrency` and `_carried_forward_runs`, which schedule and
# resume a LIVE repeated run. Both live with the agent harness, not with the
# offline aggregation in `chamberbench.variance` that the rest of this file
# exercises. Nothing they tested is reachable from this package.
