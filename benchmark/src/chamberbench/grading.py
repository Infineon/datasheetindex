"""Evaluation helpers: value matching, result lookup, case evaluation."""

from __future__ import annotations

import re
from typing import Any

from chamberbench.models import ExtractionResult, ParameterResult

# Minimum fraction of expected list items that must match.
LIST_MATCH_THRESHOLD = 0.70


# The confidence floor applied when a claim does not state one. Named here,
# once, because three independent copies of this literal used to agree only by
# coincidence -- this one, `ClaimSpec.confidence_min`'s default, and
# `score_rederivation.DEFAULT_FLOOR`. Changing any one of them silently made
# two scorers grade the same claim at different floors, with no error.
DEFAULT_CONFIDENCE_FLOOR = 0.7


def find_result(
    extraction: ExtractionResult, parameter_name: str
) -> ParameterResult | None:
    """Find a ParameterResult by name (case-insensitive)."""
    target = parameter_name.lower()
    for r in extraction.results:
        if r.parameter.lower() == target:
            return r
    return None


def serialize_numerical(result: ParameterResult) -> str:
    """Serialize the EXTRACTED numbers + unit into a searchable string.

    `source_text` is deliberately excluded. It used to be appended here, which
    made the value gate satisfiable by ECHOING the datasheet: a result whose
    min/typ/max were wrong still passed `value_contains` because the expected
    tokens appeared inside the quoted row. The gate then measured "quoted the
    right row" rather than "extracted the right number" -- most of what it
    exists to measure. Demonstrated by construction in
    tests/test_grading.py::test_serialize_numerical_excludes_source_text.

    Anything a case legitimately needs to match on other than a number is still
    reachable: `evaluate_case` appends `text_value` and `list_value` to the
    haystack separately, and the unit is included below.
    """
    parts: list[str] = []
    for v in result.values:
        tokens: list[str] = []
        if v.has_min():
            tokens.append(str(v.min_value))
        if v.has_typical():
            tokens.append(str(v.typical_value))
        if v.has_max():
            tokens.append(str(v.max_value))
        if v.unit:
            tokens.append(v.unit)
        parts.append(" ".join(tokens))
    return " ".join(parts)


def check_value_contains(haystack: str, needles: list[str]) -> tuple[bool, str]:
    """Check that all needles appear in haystack (case-insensitive)."""
    lower = haystack.lower()
    for needle in needles:
        if str(needle).lower() not in lower:
            return False, f"Missing expected value: {needle}"
    return True, ""


def check_value_pattern(haystack: str, pattern: str) -> tuple[bool, str]:
    """Check regex pattern against haystack."""
    if re.search(pattern, haystack, re.IGNORECASE):
        return True, ""
    return False, f"Does not match pattern: {pattern}"


def check_list_contains(actual: list[str], expected: list[str]) -> tuple[bool, str]:
    """Check that >= LIST_MATCH_THRESHOLD of expected items appear in actual."""
    if not expected:
        return True, ""
    actual_lower = [a.lower() for a in actual]
    matched = 0
    for exp in expected:
        exp_lower = exp.lower()
        if any(exp_lower in a for a in actual_lower):
            matched += 1
    ratio = matched / len(expected)
    if ratio >= LIST_MATCH_THRESHOLD:
        return True, ""
    return (
        False,
        f"List match {matched}/{len(expected)} ({ratio:.0%}) < threshold {LIST_MATCH_THRESHOLD:.0%}",
    )


def evaluate_case(
    result: ParameterResult | None, expected: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate a single parameter result against ground truth.

    Returns a dict with: found_expected, found_actual, found_correct,
    value_pass, confidence, failure_reason, overall_pass.
    """
    expected_found = expected["found"]

    # Handle missing parameter
    if result is None:
        return {
            "found_expected": expected_found,
            "found_actual": False,
            "found_correct": not expected_found,
            "value_pass": not expected_found,
            "confidence": 0.0,
            "failure_reason": (
                "parameter missing from agent output" if expected_found else None
            ),
            "overall_pass": not expected_found,
        }

    # "Feature is absent" has TWO semantically identical encodings, and which one a
    # run produces is not deterministic -- the same engine on the same PDF returns
    # each on different runs:
    #   found=false                  -> the agent found nothing to report
    #   found=true, bool_value=false -> the datasheet EXPLICITLY says "not supported"
    # The second is the better answer (it is grounded in a quotation), so failing it
    # scores an engine down for being more informative. golden_dataset.yaml has said
    # both are acceptable since 2026-03-19 (see nxp-tja1051-standby's comment), but
    # this function never implemented it and compared `found` strictly -- so the case
    # was a ~50/50 coin flip for BOTH engines, not a signal about either. It failed a
    # pai run while a lite run passed purely by luck; both engines were verified to
    # return found=true/bool=false on demand.
    #
    # Deliberately narrow: only when the case expects NOT-found, and only when the
    # result carries an explicit boolean false. `is False` (not `== False`, not
    # `not ...`) is load-bearing: bool_or_none() returns None for the tri-state
    # `not_specified`, and `None is False` is False, so an undetermined bool cannot
    # satisfy this. That keeps "the model could not tell" from scoring as "the
    # datasheet says no" -- the exact conflation ParameterResult's tri-state enum
    # exists to prevent. (An added `has_bool()` here would read as the guard but be
    # pure redundancy: mutation-tested, no test can tell the two apart.)
    explicit_absence = (
        not expected_found and result.found and result.bool_or_none() is False
    )
    found_correct = (expected_found == result.found) or explicit_absence
    confidence = result.confidence
    confidence_ok = confidence >= expected.get(
        "confidence_min", DEFAULT_CONFIDENCE_FLOOR
    )

    evaluation: dict[str, Any] = {
        "found_expected": expected_found,
        "found_actual": result.found,
        "found_correct": found_correct,
        "value_pass": True,  # default, overridden below
        "confidence": confidence,
        "failure_reason": None,
        "overall_pass": False,
    }

    if not found_correct:
        evaluation["failure_reason"] = (
            f"Found mismatch: expected {expected_found}, got {result.found}"
        )
        evaluation["value_pass"] = False
        return evaluation

    # For not-found cases, no value to check
    if not expected_found:
        evaluation["overall_pass"] = found_correct and confidence_ok
        if not confidence_ok:
            evaluation["failure_reason"] = (
                f"Confidence {confidence:.2f} < {expected.get('confidence_min', DEFAULT_CONFIDENCE_FLOOR)}"
            )
        return evaluation

    # Value matching for found cases
    value_pass = True
    failure_reason = None

    # Numerical: check value_contains / value_pattern against serialized values
    if "value_contains" in expected:
        haystack = serialize_numerical(result)
        # Also include text_value and list_value for auto-format cases
        if result.text_value:
            haystack += " " + result.text_value
        if result.list_value:
            haystack += " " + " ".join(result.list_value)
        ok, reason = check_value_contains(haystack, expected["value_contains"])
        if not ok:
            value_pass = False
            failure_reason = reason

    if value_pass and "value_pattern" in expected:
        haystack = serialize_numerical(result)
        if result.text_value:
            haystack += " " + result.text_value
        if result.list_value:
            haystack += " " + " ".join(result.list_value)
        ok, reason = check_value_pattern(haystack, expected["value_pattern"])
        if not ok:
            value_pass = False
            failure_reason = reason

    # Boolean (tri-state: compare the resolved True/False/None, not the enum string)
    if (
        value_pass
        and "bool_value" in expected
        and result.bool_or_none() != expected["bool_value"]
    ):
        value_pass = False
        failure_reason = f"Boolean mismatch: expected {expected['bool_value']}, got {result.bool_or_none()}"

    # List
    if value_pass and "list_contains" in expected:
        ok, reason = check_list_contains(result.list_value, expected["list_contains"])
        if not ok:
            value_pass = False
            failure_reason = reason

    evaluation["value_pass"] = value_pass
    reasons: list[str] = []
    if not value_pass and failure_reason:
        reasons.append(failure_reason)
    if not confidence_ok:
        reasons.append(
            f"Confidence {confidence:.2f} < {expected.get('confidence_min', DEFAULT_CONFIDENCE_FLOOR)}"
        )
    if reasons:
        evaluation["failure_reason"] = "; ".join(reasons)

    evaluation["overall_pass"] = found_correct and value_pass and confidence_ok
    return evaluation
