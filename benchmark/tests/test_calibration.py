"""Unit tests for chamber calibration metrics + cell loading.

No LLM, no matplotlib rendering -- only the data + math paths.
"""

from __future__ import annotations

import json

import pytest

from chamberbench.calibration import (
    CellRow,
    brier_score,
    load_ok_cells,
    per_model_summary,
)


class TestBrierScore:
    def test_empty(self):
        assert brier_score([], []) == 0.0

    def test_perfect_positive(self):
        # All outcomes True, all confidences 1.0 -- best possible.
        assert brier_score([1.0, 1.0, 1.0], [True, True, True]) == 0.0

    def test_perfect_negative(self):
        # All outcomes False, all confidences 0.0 -- also best possible.
        assert brier_score([0.0, 0.0], [False, False]) == 0.0

    def test_perfectly_wrong_positive(self):
        # All outcomes True but confidence 0.0 -- worst possible.
        assert brier_score([0.0, 0.0], [True, True]) == 1.0

    def test_perfectly_wrong_negative(self):
        assert brier_score([1.0, 1.0], [False, False]) == 1.0

    def test_all_positive_class_reduces_to_distance(self):
        # The all-positive case the chamber benchmark sits in. Brier
        # should equal mean((c - 1)**2).
        confs = [0.9, 0.8, 0.7]
        outs = [True, True, True]
        expected = ((0.9 - 1) ** 2 + (0.8 - 1) ** 2 + (0.7 - 1) ** 2) / 3
        assert brier_score(confs, outs) == pytest.approx(expected)

    def test_mixed_outcomes(self):
        # conf 0.8 vs True (loss 0.04), conf 0.3 vs False (loss 0.09)
        assert brier_score([0.8, 0.3], [True, False]) == pytest.approx(0.065)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            brier_score([0.5, 0.5], [True])


class TestLoadOkCells:
    @pytest.fixture
    def v2_baseline(self, tmp_path):
        """Minimal v2 baseline with one cell per (status × model) combo."""
        baseline = {
            "schema_version": 2,
            "claim_ids": ["claim-a", "claim-b"],
            "engines": ["agentic"],
            "models": ["m1"],
            "results": {
                "claim-a": {
                    "agentic": {
                        "m1": {
                            "status": "ok",
                            "claim_id": "claim-a",
                            "engine": "agentic",
                            "fidelity": {"confidence": 0.9, "overall_pass": True},
                            "reproducibility": {"verdict": "inconclusive"},
                            "latency_s": 10.0,
                            "n_tool_calls_by_tool": {
                                "build_datasheet": 1,
                                "search_text": 2,
                                "submit_claim_result": 1,
                            },
                        },
                        "m2": {
                            "status": "not_applicable",
                            "engine_error": "...",
                        },
                        "m3": {
                            "status": "pending_rerun",
                            "reason": "gateway down",
                        },
                    },
                },
                "claim-b": {
                    "agentic": {
                        "m1": {
                            "status": "ok",
                            "claim_id": "claim-b",
                            "engine": "agentic",
                            "fidelity": {"confidence": 0.95, "overall_pass": True},
                            "reproducibility": {"verdict": "pass"},
                            "latency_s": 20.0,
                            "n_tool_calls_by_tool": {"inspect_page": 3},
                        },
                    },
                },
            },
        }
        path = tmp_path / "baseline_chamber.json"
        path.write_text(json.dumps(baseline), encoding="utf-8")
        return path

    def test_filters_to_status_ok_only(self, v2_baseline):
        rows = load_ok_cells(v2_baseline)
        # 2 ok cells across the fixture (m1 on both claims; m2/m3 skipped).
        assert len(rows) == 2
        assert all(r.fidelity_pass for r in rows)

    def test_nav_tool_count_excludes_submit(self, v2_baseline):
        rows = load_ok_cells(v2_baseline)
        a = next(r for r in rows if r.claim_id == "claim-a")
        # build_datasheet (1) + search_text (2) = 3; submit_claim_result is excluded.
        assert a.n_nav_tool_calls == 3

    def test_repro_verdict_preserved(self, v2_baseline):
        rows = load_ok_cells(v2_baseline)
        verdicts = {r.claim_id: r.repro_verdict for r in rows}
        assert verdicts == {"claim-a": "inconclusive", "claim-b": "pass"}

    def test_v1_baseline_rejected(self, tmp_path):
        v1 = {"schema_version": 1, "results": {}}
        path = tmp_path / "baseline_chamber.json"
        path.write_text(json.dumps(v1), encoding="utf-8")
        with pytest.raises(ValueError, match="schema_version"):
            load_ok_cells(path)

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_ok_cells(tmp_path / "does_not_exist.json")


class TestPerModelSummary:
    def _row(self, model: str, conf: float, latency: float = 10.0) -> CellRow:
        return CellRow(
            model=model,
            engine="agentic",
            claim_id="x",
            confidence=conf,
            fidelity_pass=True,
            repro_verdict="pass",
            latency_s=latency,
            n_nav_tool_calls=5,
        )

    def test_groups_by_model_only(self):
        rows = [
            self._row("claudesonnet4.6", 0.9),
            self._row("claudesonnet4.6", 0.95),
            self._row("gpt-5.1", 0.8),
        ]
        s = per_model_summary(rows)
        assert set(s.keys()) == {"claudesonnet4.6", "gpt-5.1"}
        assert s["claudesonnet4.6"]["n"] == 2
        assert s["gpt-5.1"]["n"] == 1

    def test_stats_correct_for_single_model(self):
        rows = [self._row("claudesonnet4.6", 0.8), self._row("claudesonnet4.6", 1.0)]
        s = per_model_summary(rows)["claudesonnet4.6"]
        assert s["conf_mean"] == pytest.approx(0.9)
        assert s["conf_min"] == 0.8
        assert s["conf_max"] == 1.0
        # Brier on all-True outcomes: mean((c-1)^2)
        # = ((0.8-1)^2 + (1-1)^2) / 2 = 0.04 / 2 = 0.02
        assert s["brier"] == pytest.approx(0.02)

    def test_single_cell_has_zero_stdev(self):
        rows = [self._row("claudesonnet4.6", 0.9)]
        s = per_model_summary(rows)["claudesonnet4.6"]
        assert s["conf_stdev"] == 0.0

    def test_empty_input_returns_empty_dict(self):
        assert per_model_summary([]) == {}

    def test_unknown_model_skipped_from_summary(self):
        # MODEL_ORDER constrains which models the summary reports;
        # unknown models are silently skipped so the per-model figure
        # legend stays consistent across slices.
        rows = [self._row("some-unknown-model", 0.9)]
        s = per_model_summary(rows)
        assert s == {}
