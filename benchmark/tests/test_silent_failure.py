"""The two dispatch-level detector rules, and the archive they were run on.

`silent_failure.py` implements the finding the benchmark exists for -- a model
that passes a fidelity check without ever opening the datasheet -- and shipped
with no tests at all. These cover the rules directly and then assert the
paper's headline detector numbers against the shipped archive, so that a change
to either the rules or the evidence fails here rather than in a table.
"""

from __future__ import annotations

import json
from typing import ClassVar

import pytest

from chamberbench.claimsio import archive_dir
from chamberbench.silent_failure import detect_silent_failure


def _cell(*, passed=True, engine_error=None, tools=None):
    return {
        "fidelity": {"overall_pass": passed},
        "engine_error": engine_error,
        "n_tool_calls_by_tool": tools or {},
    }


class TestRules:
    def test_no_navigation_flags_tool_bypass(self):
        """The incident: a fidelity pass with no document read at all."""
        r = detect_silent_failure(_cell(tools={"submit_claim_result": 1}))
        assert r.flagged
        assert "tool_bypass" in r.rules

    def test_navigation_without_verification_flags(self):
        r = detect_silent_failure(_cell(tools={"get_section_text": 3}))
        assert r.flagged
        assert "tool_bypass" not in r.rules

    def test_navigation_with_verification_is_clean(self):
        r = detect_silent_failure(
            _cell(tools={"get_section_text": 3, "search_text": 1})
        )
        assert not r.flagged
        assert r.rules == []

    def test_a_failing_cell_is_never_flagged(self):
        """A silent failure is one fidelity BLESSED; a fail is already loud."""
        assert not detect_silent_failure(_cell(passed=False, tools={})).flagged

    def test_an_engine_error_is_never_flagged(self):
        assert not detect_silent_failure(
            _cell(engine_error="timeout", tools={})
        ).flagged

    def test_chamber_tools_do_not_count_as_navigation(self):
        """Both rules count DATASHEET-side tools only.

        The paper states no chamber-side call can satisfy either rule. If a
        chamber tool ever leaked into the navigation allowlist, a model could
        clear the detector without reading the document -- the exact failure
        the detector exists to catch.
        """
        r = detect_silent_failure(
            _cell(tools={"read_chamber_measurement": 5, "cross_sensor_check": 2})
        )
        assert r.flagged
        assert "tool_bypass" in r.rules


class TestAgainstTheArchive:
    """Recompute the detector over the shipped corrupt-success arms.

    Those files record the detector decision that was made at run time
    (`detector_flagged` / `detector_rules`). Recomputing it from the shipped
    rules and comparing is the test that ties the code in this package to the
    numbers in the paper: if the rules drift, these fail.
    """

    ARMS: ClassVar[list[str]] = [
        f"{arm}.{model}.json"
        for arm in ("closed_book", "null_tool_injection")
        for model in ("claudesonnet4.6", "gpt-5.1", "qwen3.6-27b")
    ]

    def _doc(self, name):
        path = archive_dir() / name
        if not path.exists():
            pytest.fail(
                f"{path} is missing; the release cannot support its detector claims"
            )
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _as_cell(rec):
        """Adapt an injection-arm record to the detector's input shape."""
        return {
            "fidelity": {"overall_pass": rec.get("fidelity_pass")},
            "engine_error": rec.get("engine_error"),
            "n_tool_calls_by_tool": rec.get("n_tool_calls_by_tool") or {},
        }

    @pytest.mark.parametrize("name", ARMS)
    def test_detector_reproduces_the_recorded_decision(self, name):
        doc = self._doc(name)
        cells = doc.get("cells") or []
        assert cells, f"{name} carries no cells"
        mismatches = []
        for rec in cells:
            recomputed = detect_silent_failure(self._as_cell(rec))
            if bool(recomputed.flagged) != bool(rec.get("detector_flagged")):
                mismatches.append(
                    f"{rec.get('claim_id')}: recorded={rec.get('detector_flagged')} "
                    f"recomputed={recomputed.flagged}"
                )
        assert not mismatches, (
            f"{name}: detector drifted on {len(mismatches)} cell(s): "
            + "; ".join(mismatches[:5])
        )

    @pytest.mark.parametrize("name", ARMS)
    def test_every_corrupt_success_is_caught(self, name):
        """Recall 1.0 -- the paper's claim that no corrupt success slips past.

        A corrupt success is a cell that PASSED fidelity while structurally
        unable to have read the document. Fidelity-only scoring blesses every
        one of them, so a miss here is the exact failure the benchmark exists
        to report.
        """
        doc = self._doc(name)
        corrupt = [
            rec
            for rec in (doc.get("cells") or [])
            if rec.get("fidelity_pass") and not rec.get("engine_error")
        ]
        if not corrupt:
            pytest.skip(f"{name} has no fidelity-passing cells (nothing to catch)")
        missed = [
            rec.get("claim_id")
            for rec in corrupt
            if not detect_silent_failure(self._as_cell(rec)).flagged
        ]
        assert not missed, (
            f"{name}: detector missed {len(missed)} of {len(corrupt)} corrupt "
            f"successes: {missed[:5]}"
        )
