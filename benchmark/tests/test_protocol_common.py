"""Unit tests for `chamberbench.protocols._common` -- pure logic, no chamber.

These verify the tightened condition-matching introduced in the
chamber-paper plan. The behavioural change vs. the original
`barometer_dc_accuracy._match_conditions` is that a recorded column whose
values lie outside the claim's stated value/min/max window now reports as
*unmatched*; before, only column non-existence reported unmatched.
"""

from __future__ import annotations

import pandas as pd
import pytest

from chamberbench.claims import ClaimSpec, OperatingCondition
from chamberbench.protocols._common import (
    LT_ACTUATORS,
    WT_ACTUATORS,
    match_conditions,
)


def _claim_with(*conditions: OperatingCondition) -> ClaimSpec:
    return ClaimSpec(
        id="test-claim",
        pdf_source="test.pdf",
        parameter="test",
        expected_unit="x",
        claim_kind="dc_accuracy",
        chamber_protocol="ignored",
        primary_chamber_variable="ignored",
        operating_conditions=list(conditions),
    )


class TestMatchConditions:
    def test_no_chamber_variable_is_unmatched(self):
        claim = _claim_with(OperatingCondition(name="external_ref", unit="V"))
        df = pd.DataFrame({"x": [1.0, 2.0]})
        matched, unmatched = match_conditions(claim, df)
        assert matched == []
        assert unmatched == ["external_ref"]

    def test_missing_column_is_unmatched(self):
        claim = _claim_with(OperatingCondition(name="t", chamber_variable="res_in"))
        df = pd.DataFrame({"other": [1.0]})
        matched, unmatched = match_conditions(claim, df)
        assert matched == []
        assert unmatched == ["t"]

    def test_present_column_no_constraint_matches(self):
        # No value/min/max set -- structural existence is enough.
        claim = _claim_with(OperatingCondition(name="t", chamber_variable="res_in"))
        df = pd.DataFrame({"res_in": [25.0, 25.5, 26.0]})
        matched, unmatched = match_conditions(claim, df)
        assert matched == ["t"]
        assert unmatched == []

    def test_value_in_window_matches(self):
        # Tight rtol=1e-3 around 25.0 is +/- 0.025; jitter at 1e-6 passes.
        claim = _claim_with(
            OperatingCondition(
                name="t", value=25.0, unit="C", chamber_variable="res_in"
            )
        )
        df = pd.DataFrame({"res_in": [25.0, 25.000001, 24.999999]})
        matched, _ = match_conditions(claim, df)
        assert matched == ["t"]

    def test_value_outside_window_is_unmatched(self):
        # Recorded chamber temperature at ~22 C does NOT match a claim that
        # pins the condition at 25 C. This is the bug the refactor fixes.
        claim = _claim_with(
            OperatingCondition(
                name="t", value=25.0, unit="C", chamber_variable="res_in"
            )
        )
        df = pd.DataFrame({"res_in": [22.0, 22.5, 23.0]})
        _, unmatched = match_conditions(claim, df)
        assert unmatched == ["t"]

    def test_range_in_window_matches(self):
        # -40 to 85 C operating range; chamber values inside.
        claim = _claim_with(
            OperatingCondition(
                name="t",
                min_value=-40.0,
                max_value=85.0,
                unit="C",
                chamber_variable="res_in",
            )
        )
        df = pd.DataFrame({"res_in": [22.0, 24.0, 30.0]})
        matched, _ = match_conditions(claim, df)
        assert matched == ["t"]

    def test_range_violation_is_unmatched(self):
        claim = _claim_with(
            OperatingCondition(
                name="t",
                min_value=20.0,
                max_value=30.0,
                unit="C",
                chamber_variable="res_in",
            )
        )
        df = pd.DataFrame({"res_in": [22.0, 24.0, 35.0]})  # 35 > 30
        _, unmatched = match_conditions(claim, df)
        assert unmatched == ["t"]

    def test_partial_range_lower_only(self):
        claim = _claim_with(
            OperatingCondition(
                name="t",
                min_value=20.0,
                unit="C",
                chamber_variable="res_in",
            )
        )
        df_pass = pd.DataFrame({"res_in": [21.0, 100.0]})
        df_fail = pd.DataFrame({"res_in": [10.0, 21.0]})
        assert match_conditions(claim, df_pass)[0] == ["t"]
        assert match_conditions(claim, df_fail)[1] == ["t"]

    def test_non_numeric_column_no_constraint_matches(self):
        # If the claim only requires structural existence and the column is
        # non-numeric (e.g. config flags), structural existence is enough.
        claim = _claim_with(OperatingCondition(name="cfg", chamber_variable="config"))
        df = pd.DataFrame({"config": ["standard", "standard"]})
        matched, _ = match_conditions(claim, df)
        assert matched == ["cfg"]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"value": 1.0},
            {"min_value": 0.0},
            {"max_value": 2.0},
            {"min_value": 0.0, "max_value": 2.0},
        ],
    )
    def test_non_numeric_column_with_any_constraint_is_unmatched(self, kwargs):
        # Si115x's t_vis_*/t_ir_* matchers will use min/max ranges, not just
        # value -- so all three constraint kinds need the same "cannot
        # verify against a string column" semantics, not just `value`.
        claim = _claim_with(
            OperatingCondition(name="cfg", chamber_variable="config", **kwargs)
        )
        df = pd.DataFrame({"config": ["standard", "standard"]})
        _, unmatched = match_conditions(claim, df)
        assert unmatched == ["cfg"]

    def test_value_and_min_set_together_uses_and_semantics(self):
        # The model allows both value AND min/max on the same condition.
        # Implementation runs all three checks AND-ed together, so a value
        # outside the [min, max] window is unmatched even if the recorded
        # values pass min/max alone, and vice versa. Pin this so a future
        # refactor that decides "if value is set, ignore min/max" must
        # break this test deliberately.
        claim = _claim_with(
            OperatingCondition(
                name="t",
                value=25.0,
                min_value=20.0,
                max_value=30.0,
                chamber_variable="res_in",
            )
        )
        # value satisfied, min/max satisfied -> matched
        df_pass = pd.DataFrame({"res_in": [25.0, 25.0001]})
        # min/max satisfied (22 in [20,30]) but value (25) not -> unmatched
        df_value_fails = pd.DataFrame({"res_in": [22.0, 22.0]})
        # value satisfied (25.5 ~= 25) but min/max would push it out -> n/a
        # (cannot construct: any df satisfying value=25 with rtol=1e-3 sits
        # inside [20, 30] trivially.)
        assert match_conditions(claim, df_pass)[0] == ["t"]
        assert match_conditions(claim, df_value_fails)[1] == ["t"]

    def test_nan_row_with_constraint_is_unmatched(self):
        # NaN comparisons silently return False under numpy broadcasting,
        # so a NaN row poisons `(arr >= lo) & (arr <= hi).all()` and the
        # condition reports unmatched. That's the right answer ("we cannot
        # verify"). Realistic for sensor warm-up rows in chamber data.
        claim = _claim_with(
            OperatingCondition(name="t", value=25.0, chamber_variable="res_in")
        )
        df = pd.DataFrame({"res_in": [25.0, float("nan"), 25.0]})
        _, unmatched = match_conditions(claim, df)
        assert unmatched == ["t"]

    def test_empty_dataframe_with_constraint_is_unmatched(self):
        claim = _claim_with(
            OperatingCondition(
                name="t",
                value=25.0,
                unit="C",
                chamber_variable="res_in",
            )
        )
        df = pd.DataFrame({"res_in": pd.Series([], dtype=float)})
        _, unmatched = match_conditions(claim, df)
        assert unmatched == ["t"]

    def test_empty_dataframe_no_constraint_matches(self):
        # Contract pinned in the docstring: structural existence is enough
        # when no value/min/max is set, even on an empty frame. Earlier
        # implementations had this swapped.
        claim = _claim_with(OperatingCondition(name="t", chamber_variable="res_in"))
        df = pd.DataFrame({"res_in": pd.Series([], dtype=float)})
        matched, _ = match_conditions(claim, df)
        assert matched == ["t"]

    def test_order_preserved(self):
        # Returned lists preserve the order in which conditions appear
        # on the claim spec.
        claim = _claim_with(
            OperatingCondition(name="a", chamber_variable="x"),
            OperatingCondition(name="b"),
            OperatingCondition(name="c", chamber_variable="y"),
        )
        df = pd.DataFrame({"x": [1.0], "y": [2.0]})
        matched, unmatched = match_conditions(claim, df)
        assert matched == ["a", "c"]
        assert unmatched == ["b"]

    def test_value_near_zero_uses_atol(self):
        # When the claim value is 0.0, relative tolerance collapses to
        # 0; absolute tolerance must accept tiny float jitter around 0.
        claim = _claim_with(
            OperatingCondition(name="zero", value=0.0, chamber_variable="x")
        )
        df_pass = pd.DataFrame({"x": [0.0, 1e-7, -1e-7]})
        df_fail = pd.DataFrame({"x": [0.0, 0.001, 0.0]})
        assert match_conditions(claim, df_pass)[0] == ["zero"]
        assert match_conditions(claim, df_fail)[1] == ["zero"]


class TestActuatorTuples:
    def test_wt_actuators_disjoint_from_lt(self):
        # If they shared columns, a claim against one chamber could end up
        # filtering by an actuator that doesn't exist in that dataset.
        assert set(WT_ACTUATORS).isdisjoint(set(LT_ACTUATORS))

    def test_actuators_are_tuples(self):
        # Tuples (not lists) so the constants don't get accidentally mutated
        # at module load time.
        assert isinstance(WT_ACTUATORS, tuple)
        assert isinstance(LT_ACTUATORS, tuple)


class TestUnmatchedLoadBearing:
    def test_no_load_bearing_returns_empty(self):
        from chamberbench.protocols._common import (
            unmatched_load_bearing,
        )

        claim = _claim_with(
            OperatingCondition(name="t", chamber_variable="res_in", load_bearing=False)
        )
        assert unmatched_load_bearing(claim) == []

    def test_load_bearing_with_chamber_variable_returns_empty(self):
        from chamberbench.protocols._common import (
            unmatched_load_bearing,
        )

        claim = _claim_with(
            OperatingCondition(name="t", chamber_variable="res_in", load_bearing=True)
        )
        assert unmatched_load_bearing(claim) == []

    def test_load_bearing_without_chamber_variable_is_returned(self):
        from chamberbench.protocols._common import (
            unmatched_load_bearing,
        )

        claim = _claim_with(
            OperatingCondition(name="ext_ref", load_bearing=True),
            OperatingCondition(name="t", chamber_variable="res_in", load_bearing=False),
        )
        assert unmatched_load_bearing(claim) == ["ext_ref"]


class TestStubVerdict:
    """End-to-end: a stub measurement routes to inconclusive via verdict()."""

    def test_stub_yields_inconclusive_verdict(self):
        from chamberbench.protocols._common import (
            make_stub_measurement,
        )
        from chamberbench.reproducibility import verdict

        claim = _claim_with(
            OperatingCondition(name="ext_ref", load_bearing=True),
        )
        m = make_stub_measurement(claim, unmatched_load_bearing_names=["ext_ref"])
        v = verdict(claim, m)
        assert v.verdict == "inconclusive"
        # Rationale comes from the stub's notes string, not the
        # generic "Unmatched load-bearing conditions" template.
        assert "Short-circuit" in v.rationale

    def test_reason_stub_yields_inconclusive_verdict(self):
        # The reason-based stub (e.g. linearity-not-implemented) must
        # also route to inconclusive even though no operating condition
        # is unmatched. This pins a later fix.
        from chamberbench.protocols._common import (
            make_stub_measurement,
        )
        from chamberbench.reproducibility import verdict

        claim = _claim_with(
            OperatingCondition(name="t", chamber_variable="res_in"),
        )
        m = make_stub_measurement(claim, reason="protocol not yet implemented")
        v = verdict(claim, m)
        assert v.verdict == "inconclusive"
        assert "protocol not yet implemented" in v.rationale


class TestResolutionGate:
    """A reproducibility PASS requires the chamber to resolve the claim.

    When the chamber-side measurement sigma exceeds the spec tolerance, a
    within-tolerance delta is not evidence of a pass -- the apparatus
    cannot tell a pass from a fail at that precision. The verdict must be
    inconclusive, not pass. (Pins the dps310-relative-accuracy mis-verdict.)
    """

    @staticmethod
    def _central_claim(tol: float) -> ClaimSpec:
        return ClaimSpec(
            id="resolution-gate-claim",
            pdf_source="x.pdf",
            parameter="relative accuracy",
            expected_unit="hPa",
            claim_kind="dc_accuracy",
            claimed_min=-tol,
            claimed_max=tol,
            tolerance_value=tol,
            tolerance_kind="absolute",
            chamber_protocol="ignored",
            primary_chamber_variable="ignored",
        )

    def test_sigma_exceeds_tolerance_blocks_pass(self):
        """delta within tolerance, but chamber sigma > tolerance -> inconclusive."""
        from chamberbench.claims import ChamberMeasurement
        from chamberbench.reproducibility import verdict

        claim = self._central_claim(tol=0.06)
        m = ChamberMeasurement(
            claim_id=claim.id,
            experiment_ids=["e"],
            measured_value=0.05,  # |delta| = 0.05 <= 0.06 spec tolerance
            measured_unit="hPa",
            measured_sigma=0.08,  # 0.08 > 0.06 -- chamber cannot resolve it
            measured_sigma_basis="cross_sensor",
        )
        v = verdict(claim, m)
        assert v.verdict == "inconclusive", v.rationale

    def test_sigma_within_tolerance_still_passes(self):
        """Control: delta within tolerance AND sigma <= tolerance -> pass."""
        from chamberbench.claims import ChamberMeasurement
        from chamberbench.reproducibility import verdict

        claim = self._central_claim(tol=0.06)
        m = ChamberMeasurement(
            claim_id=claim.id,
            experiment_ids=["e"],
            measured_value=0.05,  # |delta| = 0.05 <= 0.06
            measured_unit="hPa",
            measured_sigma=0.02,  # 0.02 <= 0.06 -- chamber resolves it
            measured_sigma_basis="cross_sensor",
        )
        v = verdict(claim, m)
        assert v.verdict == "pass", v.rationale


class TestLightSensorLinearity:
    """The linearity claim_kind routes to a reason-stub, not a NotImplementedError."""

    def test_linearity_routes_to_inconclusive_stub(self):
        from chamberbench.protocols import light_sensor
        from chamberbench.reproducibility import verdict

        claim = ClaimSpec(
            id="linearity-smoke",
            pdf_source="x.pdf",
            parameter="linearity test",
            expected_unit="R2",
            claim_kind="linearity",
            chamber_protocol="chamberbench.protocols.light_sensor",
            primary_chamber_variable="vis_1",
        )
        m = light_sensor.run(claim)
        # Stub marker must be set so quality_gates.H5 silences expected NaN.
        assert m.measured_sigma_basis == "stub"
        # Verdict must be inconclusive, not fail (avoids tripping the H5
        # hard gate on a known-deferred case).
        v = verdict(claim, m)
        assert v.verdict == "inconclusive"
        assert "linearity" in v.rationale.lower()


class TestResolveCacheRoot:
    def test_default_path(self, monkeypatch):
        from chamberbench.protocols._common import (
            resolve_cache_root,
        )

        monkeypatch.delenv("CHAMBER_CACHE_ROOT", raising=False)
        assert str(resolve_cache_root()) == "/tmp/cc_data"

    def test_env_override(self, monkeypatch):
        from chamberbench.protocols._common import (
            resolve_cache_root,
        )

        monkeypatch.setenv("CHAMBER_CACHE_ROOT", "/var/cache/chamber")
        assert str(resolve_cache_root()) == "/var/cache/chamber"


class TestMakeStubMeasurement:
    def test_stub_marks_sigma_basis(self):
        from chamberbench.protocols._common import (
            make_stub_measurement,
        )

        claim = _claim_with(
            OperatingCondition(name="ext_ref", load_bearing=True),
            OperatingCondition(name="t", chamber_variable="res_in"),
        )
        m = make_stub_measurement(claim, ["ext_ref"])
        # H5 keys on this exact value -- must not change without updating
        # quality_gates._check_protocol_errors.
        assert m.measured_sigma_basis == "stub"

    def test_stub_value_and_sigma_are_nan(self):
        import math

        from chamberbench.protocols._common import (
            make_stub_measurement,
        )

        claim = _claim_with(OperatingCondition(name="ext_ref", load_bearing=True))
        m = make_stub_measurement(claim, ["ext_ref"])
        assert math.isnan(m.measured_value)
        assert math.isnan(m.measured_sigma)
        assert m.sample_n == 0

    def test_stub_lists_partition_by_chamber_variable_presence(self):
        from chamberbench.protocols._common import (
            make_stub_measurement,
        )

        claim = _claim_with(
            OperatingCondition(name="ext_ref"),  # no chamber_variable
            OperatingCondition(name="t", chamber_variable="res_in"),
        )
        m = make_stub_measurement(claim, ["ext_ref"])
        assert "t" in m.matched_conditions
        assert "ext_ref" in m.unmatched_conditions

    def test_stub_notes_names_unmatched(self):
        from chamberbench.protocols._common import (
            make_stub_measurement,
        )

        claim = _claim_with(OperatingCondition(name="ext_ref", load_bearing=True))
        m = make_stub_measurement(claim, ["ext_ref"])
        assert "ext_ref" in m.notes
        assert "Short-circuit" in m.notes

    def test_stub_with_reason(self):
        # Reason-based stub for protocol-not-implemented and similar cases.
        from chamberbench.protocols._common import (
            make_stub_measurement,
        )

        claim = _claim_with(OperatingCondition(name="t", chamber_variable="res_in"))
        m = make_stub_measurement(claim, reason="not yet implemented")
        assert m.measured_sigma_basis == "stub"
        assert "not yet implemented" in m.notes

    def test_stub_requires_exactly_one_kind(self):
        # Caller bug: must specify either unmatched_load_bearing_names OR
        # reason, never both, never neither.
        from chamberbench.protocols._common import (
            make_stub_measurement,
        )

        claim = _claim_with(OperatingCondition(name="t", chamber_variable="res_in"))
        with pytest.raises(ValueError, match="exactly one"):
            make_stub_measurement(claim)
        with pytest.raises(ValueError, match="exactly one"):
            make_stub_measurement(claim, ["x"], reason="y")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
