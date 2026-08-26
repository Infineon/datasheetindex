"""Unit tests for eval helpers -- no LLM required."""

from chamberbench.grading import (
    check_list_contains,
    check_value_contains,
    check_value_pattern,
    evaluate_case,
    find_result,
    serialize_numerical,
)
from chamberbench.models import ConditionedValue, ExtractionResult, ParameterResult


def _make_result(name: str, found: bool = True, **kwargs) -> ParameterResult:
    return ParameterResult(
        parameter=name, found=found, confidence=kwargs.pop("confidence", 0.9), **kwargs
    )


def _make_extraction(*results: ParameterResult) -> ExtractionResult:
    return ExtractionResult(pdf_source="test.pdf", results=list(results))


class TestFindResult:
    def test_exact_match(self):
        ext = _make_extraction(_make_result("Ron"))
        assert find_result(ext, "Ron") is not None

    def test_case_insensitive(self):
        ext = _make_extraction(_make_result("Supply Voltage"))
        assert find_result(ext, "supply voltage") is not None

    def test_not_found(self):
        ext = _make_extraction(_make_result("Ron"))
        assert find_result(ext, "Vgs(th)") is None


class TestSerializeNumerical:
    def test_with_values(self):
        r = _make_result(
            "Ron",
            values=[
                ConditionedValue(
                    conditions="Vgs=10V",
                    min_value=1.5,
                    typical_value=1.9,
                    max_value=2.3,
                    unit="mOhm",
                )
            ],
        )
        s = serialize_numerical(r)
        assert "1.5" in s
        assert "1.9" in s
        assert "2.3" in s
        assert "mOhm" in s

    def test_empty_values(self):
        r = _make_result("Ron", values=[])
        assert serialize_numerical(r) == ""


class TestCheckValueContains:
    def test_all_present(self):
        ok, _ = check_value_contains("4.5 V to 5.5 V", ["4.5", "5.5"])
        assert ok

    def test_missing(self):
        ok, reason = check_value_contains("4.5 V", ["4.5", "5.5"])
        assert not ok
        assert "5.5" in reason


class TestCheckValuePattern:
    def test_match(self):
        ok, _ = check_value_pattern("5 Mbit/s", r"5\s*Mbit/s")
        assert ok

    def test_no_match(self):
        ok, _ = check_value_pattern("3 Mbit/s", r"5\s*Mbit/s")
        assert not ok


class TestCheckListContains:
    def test_all_match(self):
        ok, _ = check_list_contains(["4KB Sector", "32KB Block"], ["4KB", "32KB"])
        assert ok

    def test_partial_above_threshold(self):
        # 3/4 = 75% >= 70%
        ok, _ = check_list_contains(["A", "B", "C"], ["A", "B", "C", "D"])
        assert ok

    def test_partial_below_threshold(self):
        # 1/4 = 25% < 70%
        ok, _ = check_list_contains(["A"], ["A", "B", "C", "D"])
        assert not ok

    def test_empty_expected(self):
        ok, _ = check_list_contains(["A"], [])
        assert ok


class TestEvaluateCase:
    def test_found_correct_value_match(self):
        r = _make_result(
            "Ron",
            values=[ConditionedValue(typical_value=1.9, unit="mOhm")],
        )
        ev = evaluate_case(
            r, {"found": True, "confidence_min": 0.7, "value_contains": ["1.9", "mOhm"]}
        )
        assert ev["overall_pass"]
        assert ev["found_correct"]
        assert ev["value_pass"]

    def test_found_mismatch(self):
        r = _make_result("Ron", found=False)
        ev = evaluate_case(r, {"found": True, "confidence_min": 0.7})
        assert not ev["overall_pass"]
        assert not ev["found_correct"]
        assert "Found mismatch" in ev["failure_reason"]

    def test_not_found_correct(self):
        r = _make_result("Vio", found=False)
        ev = evaluate_case(r, {"found": False, "confidence_min": 0.7})
        assert ev["overall_pass"]

    def test_missing_parameter(self):
        ev = evaluate_case(None, {"found": True, "confidence_min": 0.7})
        assert not ev["overall_pass"]
        assert "missing" in ev["failure_reason"]

    def test_missing_parameter_expected_not_found(self):
        ev = evaluate_case(None, {"found": False, "confidence_min": 0.7})
        assert ev["overall_pass"]

    def test_low_confidence(self):
        r = _make_result("Ron", confidence=0.5)
        ev = evaluate_case(r, {"found": True, "confidence_min": 0.7})
        assert not ev["overall_pass"]
        assert "Confidence" in ev["failure_reason"]

    def test_boolean_match(self):
        r = _make_result("Standby Mode", bool_value=True)
        ev = evaluate_case(
            r, {"found": True, "confidence_min": 0.7, "bool_value": True}
        )
        assert ev["overall_pass"]

    def test_boolean_mismatch(self):
        r = _make_result("Standby Mode", bool_value=False)
        ev = evaluate_case(
            r, {"found": True, "confidence_min": 0.7, "bool_value": True}
        )
        assert not ev["overall_pass"]
        assert "Boolean mismatch" in ev["failure_reason"]

    def test_list_match(self):
        r = _make_result("Package", list_value=["SOIC-8", "SON-8"])
        ev = evaluate_case(
            r, {"found": True, "confidence_min": 0.7, "list_contains": ["SOIC", "SON"]}
        )
        assert ev["overall_pass"]


class TestExplicitAbsenceTolerance:
    """`found=true, bool=false` must count as "feature absent".

    golden_dataset.yaml has documented both encodings as acceptable since
    2026-03-19 (nxp-tja1051-standby), but evaluate_case compared `found`
    strictly, so the case was a ~50/50 coin flip for BOTH engines rather than a
    signal about either -- it failed a pai run while a lite run passed by luck.
    """

    def test_explicit_absence_passes(self):
        # The BETTER answer: grounded in the datasheet explicitly saying "not supported".
        r = _make_result("Standby Mode", found=True, bool_value=False, confidence=0.95)
        ev = evaluate_case(r, {"found": False, "confidence_min": 0.7})
        assert ev["overall_pass"]
        assert ev["found_correct"]
        assert ev["failure_reason"] is None

    def test_plain_not_found_still_passes(self):
        # The other encoding must keep working -- this is a widening, not a swap.
        r = _make_result("Standby Mode", found=False, confidence=0.95)
        ev = evaluate_case(r, {"found": False, "confidence_min": 0.7})
        assert ev["overall_pass"]

    def test_explicit_presence_still_fails(self):
        # The tolerance must NOT swallow a real error: claiming the feature EXISTS
        # when ground truth says it does not is still a failure.
        r = _make_result("Standby Mode", found=True, bool_value=True, confidence=0.95)
        ev = evaluate_case(r, {"found": False, "confidence_min": 0.7})
        assert not ev["overall_pass"]
        assert not ev["found_correct"]
        assert "Found mismatch" in ev["failure_reason"]

    def test_undetermined_bool_still_fails(self):
        # THE boundary. Tri-state `not_specified` means "the model could not tell",
        # which must never score as "the datasheet says no" -- that conflation is
        # exactly what ParameterResult's tri-state enum exists to prevent. This is
        # what makes evaluate_case's `is False` identity check (rather than a
        # falsy/`== False` test) mandatory: bool_or_none() returns None here.
        r = _make_result("Standby Mode", found=True, confidence=0.95)
        assert r.has_bool() is False  # guards the premise: this really is not_specified
        ev = evaluate_case(r, {"found": False, "confidence_min": 0.7})
        assert not ev["overall_pass"]
        assert not ev["found_correct"]

    def test_explicit_absence_still_honors_confidence_floor(self):
        # The tolerance widens `found`, not the confidence gate.
        r = _make_result("Standby Mode", found=True, bool_value=False, confidence=0.5)
        ev = evaluate_case(r, {"found": False, "confidence_min": 0.7})
        assert not ev["overall_pass"]
        assert ev["found_correct"]  # the found-encoding was accepted...
        assert "Confidence" in ev["failure_reason"]  # ...but confidence still gates


# TestEvalResultsCollectorDiagnostics (3 tests) was removed when this suite
# was extracted. It exercised the pytest results collector that drives a live
# chamber run -- part of the agent harness, not of the grading surface this
# package ships. The behaviour it covered (process_diagnostics reaching the
# recorded cell) is not reachable from anything here.


class TestSourceTextDoesNotSatisfyValueChecks:
    """A quoted row must not stand in for an extracted number.

    `serialize_numerical` used to fold `source_text` into the string that
    `value_contains` / `value_pattern` are matched against. That made the value
    gate satisfiable by ECHOING the datasheet line: a result whose min/typ/max
    were wrong still passed, because the expected tokens appeared in the quote.
    The gate then measured "quoted the right row" rather than "extracted the
    right number", which is most of what it exists to measure.
    """

    def _echoing(self, **kwargs):
        from chamberbench.models import ConditionedValue, ParameterResult

        return ParameterResult(
            parameter="Supply Voltage",
            found=True,
            confidence=0.95,
            values=[
                ConditionedValue(
                    source_text="Supply voltage VDD 2.15 3.3 5.5 V", **kwargs
                )
            ],
        )

    def test_wrong_numbers_are_not_rescued_by_an_echoed_quote(self):
        wrong = self._echoing(
            min_value=99.0, typical_value=98.0, max_value=97.0, unit="V"
        )

        ev = evaluate_case(
            wrong,
            {"found": True, "confidence_min": 0.8, "value_contains": ["2.15", "5.5"]},
        )

        assert ev["value_pass"] is False
        assert ev["overall_pass"] is False

    def test_right_numbers_still_pass(self):
        right = self._echoing(
            min_value=2.15, typical_value=3.3, max_value=5.5, unit="V"
        )

        ev = evaluate_case(
            right,
            {"found": True, "confidence_min": 0.8, "value_contains": ["2.15", "5.5"]},
        )

        assert ev["overall_pass"] is True

    def test_value_pattern_is_not_rescued_either(self):
        wrong = self._echoing(
            min_value=99.0, typical_value=98.0, max_value=97.0, unit="V"
        )

        # The pattern must be one the ECHO would satisfy -- otherwise the test
        # passes because the regex simply does not match, proving nothing.
        ev = evaluate_case(
            wrong,
            {"found": True, "confidence_min": 0.8, "value_pattern": r"2\.15"},
        )

        assert ev["value_pass"] is False

    def test_serialize_numerical_excludes_source_text(self):
        r = self._echoing(min_value=2.15, typical_value=3.3, max_value=5.5, unit="V")

        s = serialize_numerical(r)

        assert "2.15" in s and "V" in s
        assert "Supply voltage VDD" not in s
