"""Unit tests for the chamber cross-contamination audit.

Synthetic JSONL fixtures cover the three behaviours that matter:
- a clean cell (datasheet tools only, then submit) → not contaminated
- a contaminated cell (chamber tool fires before submit) → counted
- an unfinished cell (no submit event) → excluded from the rate
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chamberbench.contamination_audit import (
    CHAMBER_TOOL_NAMES,
    analyze_traces,
)


def _write_trace(path: Path, events: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(ev) for ev in events) + "\n",
        encoding="utf-8",
    )


def _ev(
    run_id: str,
    claim_id: str,
    step: int,
    tool: str | None,
    *,
    engine: str = "agentic",
    kind: str = "tool_call",
    phase: str | None = None,
) -> dict:
    out = {
        "run_id": run_id,
        "claim_id": claim_id,
        "engine": engine,
        "step": step,
        "kind": kind,
    }
    if tool:
        out["tool_name"] = tool
    if phase is not None:
        out["phase"] = phase
    return out


class TestContaminationDetection:
    def test_chamber_tools_constant_matches_protocol(self) -> None:
        # Pin the named set so a future tool rename doesn't silently
        # widen or narrow the audit. If a new chamber tool is added,
        # update both `chamber_tools.py` and this set together.
        assert CHAMBER_TOOL_NAMES == frozenset(
            {
                "list_experiments",
                "get_experiment_metadata",
                "query_dataset",
                "cross_sensor_check",
                "run_simulator",
                "get_ground_truth_graph",
            }
        )

    def test_clean_cell_not_counted(self, tmp_path: Path) -> None:
        # Datasheet-only path, then submit.
        events = [
            _ev("r1", "c1", 0, "build_datasheet"),
            _ev("r1", "c1", 1, "search_text"),
            _ev("r1", "c1", 2, "extract_table_markdown"),
            _ev("r1", "c1", 3, "submit_claim_result"),
        ]
        path = tmp_path / "latest_traces.x.jsonl"
        _write_trace(path, events)
        summary = analyze_traces(path)
        assert summary["n_cells_total"] == 1
        assert summary["n_cells_with_submit"] == 1
        assert summary["n_cells_contaminated"] == 0
        assert summary["contamination_rate"] == 0.0
        cell = summary["per_cell"][0]
        assert cell["n_chamber_calls_before_submit"] == 0
        assert cell["first_chamber_tool_step"] is None
        assert cell["chamber_tools_used"] == []

    def test_chamber_before_submit_counted(self, tmp_path: Path) -> None:
        events = [
            _ev("r1", "c1", 0, "build_datasheet"),
            _ev("r1", "c1", 1, "list_experiments"),
            _ev("r1", "c1", 2, "query_dataset"),
            _ev("r1", "c1", 3, "search_text"),
            _ev("r1", "c1", 4, "submit_claim_result"),
        ]
        path = tmp_path / "latest_traces.x.jsonl"
        _write_trace(path, events)
        summary = analyze_traces(path)
        assert summary["n_cells_contaminated"] == 1
        assert summary["contamination_rate"] == 1.0
        cell = summary["per_cell"][0]
        assert cell["n_chamber_calls_before_submit"] == 2
        assert cell["first_chamber_tool_step"] == 1
        assert cell["chamber_tools_used"] == ["list_experiments", "query_dataset"]

    def test_chamber_after_submit_not_counted(self, tmp_path: Path) -> None:
        # No real trace would call tools after submit (the loop ends),
        # but defensively: anything at or after submit_step is excluded.
        events = [
            _ev("r1", "c1", 0, "build_datasheet"),
            _ev("r1", "c1", 1, "submit_claim_result"),
            _ev("r1", "c1", 2, "list_experiments"),  # ignored
        ]
        path = tmp_path / "latest_traces.x.jsonl"
        _write_trace(path, events)
        summary = analyze_traces(path)
        assert summary["n_cells_contaminated"] == 0

    def test_cell_without_submit_excluded_from_rate(self, tmp_path: Path) -> None:
        # Engine-error cell that never reaches submit. Should be counted
        # in n_cells_total but NOT in n_cells_with_submit, so the rate
        # denominator excludes it.
        events = [
            _ev("r1", "c1", 0, "build_datasheet"),
            _ev("r1", "c1", 1, "list_experiments"),
            # no submit_claim_result -- e.g. context overflow
        ]
        path = tmp_path / "latest_traces.x.jsonl"
        _write_trace(path, events)
        summary = analyze_traces(path)
        assert summary["n_cells_total"] == 1
        assert summary["n_cells_with_submit"] == 0
        assert summary["n_cells_contaminated"] == 0
        assert summary["contamination_rate"] == 0.0

    def test_multiple_cells_aggregated(self, tmp_path: Path) -> None:
        events = [
            # clean cell
            _ev("r1", "c1", 0, "build_datasheet"),
            _ev("r1", "c1", 1, "submit_claim_result"),
            # contaminated cell
            _ev("r2", "c2", 0, "list_experiments"),
            _ev("r2", "c2", 1, "build_datasheet"),
            _ev("r2", "c2", 2, "submit_claim_result"),
            # another contaminated cell, different chamber tool
            _ev("r3", "c3", 0, "cross_sensor_check"),
            _ev("r3", "c3", 1, "submit_claim_result"),
        ]
        path = tmp_path / "latest_traces.x.jsonl"
        _write_trace(path, events)
        summary = analyze_traces(path)
        assert summary["n_cells_total"] == 3
        assert summary["n_cells_with_submit"] == 3
        assert summary["n_cells_contaminated"] == 2
        assert summary["contamination_rate"] == pytest.approx(2 / 3)

    def test_session_start_sentinel_ignored(self, tmp_path: Path) -> None:
        # session_start lines carry no claim_id and must not crash or
        # contribute to any cell tally.
        events = [
            {"kind": "session_start", "session_id": "s", "worker": "main"},
            _ev("r1", "c1", 0, "build_datasheet"),
            _ev("r1", "c1", 1, "submit_claim_result"),
        ]
        path = tmp_path / "latest_traces.x.jsonl"
        _write_trace(path, events)
        summary = analyze_traces(path)
        assert summary["n_cells_total"] == 1
        assert summary["n_cells_with_submit"] == 1

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        summary = analyze_traces(tmp_path / "does-not-exist.jsonl")
        assert summary["n_cells_total"] == 0
        assert summary["n_cells_with_submit"] == 0
        assert summary["contamination_rate"] == 0.0


class TestTwoPassPhaseAware:
    """schema-v2 traces carry `phase`; the audit keys on it directly."""

    def test_two_pass_trace_is_clean_by_construction(self, tmp_path: Path) -> None:
        # The freeze means every chamber call is phase="chamber", after
        # submit_extraction. Contamination is 0 regardless of step ordering.
        events = [
            _ev("r1", "c1", 0, "build_datasheet", phase="extraction"),
            _ev("r1", "c1", 1, "search_text", phase="extraction"),
            _ev("r1", "c1", 2, "submit_extraction", phase="extraction"),
            _ev("r1", "c1", 3, "list_experiments", phase="chamber"),
            _ev("r1", "c1", 4, "cross_sensor_check", phase="chamber"),
            _ev("r1", "c1", 5, "submit_chamber_outcome", phase="chamber"),
        ]
        path = tmp_path / "latest_traces.x.jsonl"
        _write_trace(path, events)
        summary = analyze_traces(path)
        assert summary["n_cells_with_submit"] == 1
        assert summary["n_cells_contaminated"] == 0
        assert summary["contamination_rate"] == 0.0

    def test_extraction_phase_chamber_call_is_flagged(self, tmp_path: Path) -> None:
        # If a chamber tool ever fired in the extraction phase (it cannot under
        # the real loop), the phase-aware audit would catch it -- proving it
        # reads `phase`, not just step ordering.
        events = [
            _ev("r1", "c1", 0, "build_datasheet", phase="extraction"),
            _ev("r1", "c1", 1, "query_dataset", phase="extraction"),
            _ev("r1", "c1", 2, "submit_extraction", phase="extraction"),
            _ev("r1", "c1", 3, "list_experiments", phase="chamber"),
            _ev("r1", "c1", 4, "submit_chamber_outcome", phase="chamber"),
        ]
        path = tmp_path / "latest_traces.x.jsonl"
        _write_trace(path, events)
        summary = analyze_traces(path)
        assert summary["n_cells_contaminated"] == 1
        cell = summary["per_cell"][0]
        assert cell["chamber_tools_used"] == ["query_dataset"]

    def test_v2_cell_without_freeze_excluded_from_denominator(
        self, tmp_path: Path
    ) -> None:
        # A v2 cell that errored in phase 1 (never reached submit_extraction)
        # carries phase info but has no assessable extraction phase, so it is
        # excluded from the rate denominator -- same as the v1 no-submit case.
        events = [
            _ev("r1", "c1", 0, "build_datasheet", phase="extraction"),
            _ev("r1", "c1", 1, "search_text", phase="extraction"),
            # no submit_extraction -- e.g. context overflow before freezing
        ]
        path = tmp_path / "latest_traces.x.jsonl"
        _write_trace(path, events)
        summary = analyze_traces(path)
        assert summary["n_cells_total"] == 1
        assert summary["n_cells_with_submit"] == 0
        assert summary["contamination_rate"] == 0.0
