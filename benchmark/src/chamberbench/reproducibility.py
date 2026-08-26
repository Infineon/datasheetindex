"""Reproducibility verdict logic for the chamber benchmark.

Pure, deterministic; no LLM. Takes the chamber-side measurement from
`chamberbench.protocols.*` and the claim from a `ClaimSpec`, and returns
pass / fail / inconclusive.

`ClaimSpec` is the DATASHEET's claim, not the agent's answer -- it is
hand-authored from the document and carries the claimed bounds and operating
conditions. `verdict()` never sees an extraction, and nothing on this path
does. That is what makes the reproducibility axis independent of fidelity
rather than a second opinion about the same evidence, and it is why the two
verdicts can disagree informatively.

(An earlier version of this docstring said "the agent-side claim from a
ClaimSpec", which reads as though an extraction were an input. It is not,
and the phrasing misled at least one reader into believing the chamber
grounded the extraction rather than the claim.)

Verdict rule (from docs/datasheetindex_chamber_benchmark.md):
1. If any *load-bearing* operating condition is unmatched -> inconclusive.
2. If chamber sigma > spec_tolerance -> inconclusive: the apparatus cannot
   resolve the claim at the stated precision, so a "pass" is not
   supportable even if the point estimate lands within tolerance. (Range
   claims are exempt -- containment in a wide window does not depend on
   resolving the tolerance.)
3. If |measured - claimed_central| <= spec_tolerance -> pass.
4. If |measured - claimed_central| <= combined_uncertainty -> inconclusive.
5. Otherwise -> fail.

`spec_tolerance` derives from the claim:
  - tolerance_kind == "absolute"  -> tolerance_value
  - tolerance_kind == "relative"  -> tolerance_value * |claimed_central|
  - tolerance_kind == "spec_derived" -> half-width of [claimed_min, claimed_max]
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .uncertainty import combined_uncertainty

if TYPE_CHECKING:
    from chamberbench.claims import (
        ChamberMeasurement,
        ClaimSpec,
        ReproducibilityVerdict,
    )


def verdict(
    claim: ClaimSpec,
    measurement: ChamberMeasurement,
) -> ReproducibilityVerdict:
    """Compute the pass / fail / inconclusive verdict for one claim."""
    from chamberbench.claims import ReproducibilityVerdict

    central = _claimed_central(claim)
    spec_tol = _resolve_spec_tolerance(claim, central)
    combined = combined_uncertainty(measurement.measured_sigma, spec_tol)
    sigma_basis = measurement.measured_sigma_basis

    # 0) Stub measurement -> inconclusive. The protocol module
    # short-circuited because it could not compute a meaningful value
    # (load-bearing condition unmatched, protocol not yet implemented,
    # degenerate input, etc.). Treat as inconclusive regardless of
    # downstream comparison logic; combined_uncertainty is NaN-poisoned
    # at this point and not meaningful.
    if sigma_basis == "stub":
        return ReproducibilityVerdict(
            claim_id=claim.id,
            verdict="inconclusive",
            rationale=measurement.notes
            or "Protocol returned a stub measurement; verdict cannot be computed.",
            spec_tolerance=spec_tol,
            combined_uncertainty=None,
            matched_subset=claim.realizable_subset,
        )

    # 1) Unmatched load-bearing conditions -> inconclusive. (Stubs above
    # already cover the load-bearing-no-chamber-variable case; this fires
    # for non-stub measurements where a load-bearing constraint is
    # violated by the loaded experiment's recorded values, e.g. claim
    # says T=25 C and the chamber recorded T=22 C.)
    load_bearing_unmatched = [
        cond.name
        for cond in claim.operating_conditions
        if cond.load_bearing and cond.name in measurement.unmatched_conditions
    ]
    if load_bearing_unmatched:
        return ReproducibilityVerdict(
            claim_id=claim.id,
            verdict="inconclusive",
            rationale=(
                "Unmatched load-bearing conditions: "
                + ", ".join(load_bearing_unmatched)
                + ". Chamber cannot exercise the stated regime."
            ),
            spec_tolerance=spec_tol,
            combined_uncertainty=combined,
            matched_subset=claim.realizable_subset,
        )

    delta = abs(measurement.measured_value - central) if central is not None else None

    if central is None:
        # Range claims: the claim is a [min,max] window. The value is
        # reproducible if the chamber measurement lies inside it; if it
        # falls outside but within the chamber-side combined uncertainty
        # of either bound, downgrade to inconclusive rather than fail.
        if not (claim.has_min() and claim.has_max()):
            return ReproducibilityVerdict(
                claim_id=claim.id,
                verdict="inconclusive",
                rationale=(
                    "Range claim is missing one of claimed_min / claimed_max; cannot evaluate."
                ),
                spec_tolerance=spec_tol,
                combined_uncertainty=combined,
                matched_subset=claim.realizable_subset,
            )
        m = measurement.measured_value
        lo, hi = claim.claimed_min, claim.claimed_max
        if lo <= m <= hi:
            return ReproducibilityVerdict(
                claim_id=claim.id,
                verdict="pass",
                rationale=(
                    f"Measured {m:g} {measurement.measured_unit} "
                    f"is inside [{lo:g}, {hi:g}] "
                    f"(sigma_basis={measurement.measured_sigma_basis})."
                ),
                spec_tolerance=spec_tol,
                combined_uncertainty=combined,
                matched_subset=claim.realizable_subset,
            )
        # Outside the strict window. Use combined uncertainty as a soft band.
        boundary_distance = lo - m if m < lo else m - hi
        if boundary_distance <= combined:
            return ReproducibilityVerdict(
                claim_id=claim.id,
                verdict="inconclusive",
                rationale=(
                    f"Measured {m:g} {measurement.measured_unit} is outside "
                    f"[{lo:g}, {hi:g}] by {boundary_distance:.4g}, within "
                    f"combined uncertainty {combined:.4g} "
                    f"(sigma_basis={measurement.measured_sigma_basis})."
                ),
                spec_tolerance=spec_tol,
                combined_uncertainty=combined,
                matched_subset=claim.realizable_subset,
            )
        return ReproducibilityVerdict(
            claim_id=claim.id,
            verdict="fail",
            rationale=(
                f"Measured {m:g} {measurement.measured_unit} lies "
                f"{boundary_distance:.4g} outside [{lo:g}, {hi:g}], "
                f"exceeding combined uncertainty {combined:.4g}."
            ),
            spec_tolerance=spec_tol,
            combined_uncertainty=combined,
            matched_subset=claim.realizable_subset,
        )

    # `delta` is non-None on the central-value path (range path returned earlier).
    assert delta is not None

    # Resolution gate: a "pass" requires the chamber to actually resolve the
    # claim. When the chamber-side measurement sigma exceeds the spec
    # tolerance, a delta within tolerance is not evidence -- the apparatus
    # cannot distinguish a pass from a fail at that precision. Route to
    # inconclusive. (Range claims, handled above, are exempt: containment in
    # a wide window does not depend on resolving the tolerance.)
    if measurement.measured_sigma > spec_tol:
        return ReproducibilityVerdict(
            claim_id=claim.id,
            verdict="inconclusive",
            rationale=(
                f"chamber sigma={measurement.measured_sigma:.4g} exceeds "
                f"spec_tolerance={spec_tol:.4g}; the apparatus cannot resolve "
                f"this claim at the stated precision (|delta|={delta:.4g}, "
                f"sigma_basis={sigma_basis})."
            ),
            delta=delta,
            spec_tolerance=spec_tol,
            combined_uncertainty=combined,
            matched_subset=claim.realizable_subset,
        )

    if delta <= spec_tol:
        return ReproducibilityVerdict(
            claim_id=claim.id,
            verdict="pass",
            rationale=(
                f"|delta|={delta:.4g} <= spec_tolerance={spec_tol:.4g}; "
                f"chamber sigma={measurement.measured_sigma:.4g} "
                f"(sigma_basis={sigma_basis})."
            ),
            delta=delta,
            spec_tolerance=spec_tol,
            combined_uncertainty=combined,
            matched_subset=claim.realizable_subset,
        )

    if delta <= combined:
        return ReproducibilityVerdict(
            claim_id=claim.id,
            verdict="inconclusive",
            rationale=(
                f"spec_tolerance={spec_tol:.4g} < |delta|={delta:.4g} <= "
                f"combined={combined:.4g}; disagreement smaller than combined "
                f"agent+chamber uncertainty (sigma_basis={sigma_basis})."
            ),
            delta=delta,
            spec_tolerance=spec_tol,
            combined_uncertainty=combined,
            matched_subset=claim.realizable_subset,
        )

    return ReproducibilityVerdict(
        claim_id=claim.id,
        verdict="fail",
        rationale=(
            f"|delta|={delta:.4g} exceeds combined uncertainty={combined:.4g} "
            f"(spec_tolerance={spec_tol:.4g}, sigma_basis={sigma_basis})."
        ),
        delta=delta,
        spec_tolerance=spec_tol,
        combined_uncertainty=combined,
        matched_subset=claim.realizable_subset,
    )


def run_protocol(claim: ClaimSpec) -> ChamberMeasurement:
    """Dispatch to the protocol module named by `claim.chamber_protocol`."""
    module = importlib.import_module(claim.chamber_protocol)
    if not hasattr(module, "run"):
        raise AttributeError(
            f"Protocol module {claim.chamber_protocol} has no run() function"
        )
    return module.run(claim)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _claimed_central(claim: ClaimSpec) -> float | None:
    """Best central value to compare measurements against. None for pure range claims."""
    if claim.has_typical():
        return claim.claimed_typical
    if claim.has_min() and claim.has_max():
        # For range claims the verdict logic uses the full [min,max] window
        # rather than a midpoint -- return None to signal that.
        if claim.claim_kind == "range":
            return None
        return 0.5 * (claim.claimed_min + claim.claimed_max)
    if claim.has_min():
        return claim.claimed_min
    if claim.has_max():
        return claim.claimed_max
    return None


def _resolve_spec_tolerance(claim: ClaimSpec, central: float | None) -> float:
    """Convert the claim's tolerance_kind/value into an absolute tolerance."""
    if claim.tolerance_kind == "absolute" and claim.has_explicit_tolerance():
        return float(claim.tolerance_value)
    if claim.tolerance_kind == "relative" and claim.has_explicit_tolerance():
        if central is None:
            return 0.0
        return float(claim.tolerance_value) * abs(central)
    # spec_derived: half the [min, max] window, or |bound| if only one bound.
    if claim.has_min() and claim.has_max():
        return 0.5 * (claim.claimed_max - claim.claimed_min)
    if claim.has_min():
        return abs(claim.claimed_min)
    if claim.has_max():
        return abs(claim.claimed_max)
    return 0.0
