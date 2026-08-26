"""Reproducibility-perturbation experiment for the chamber paper.

Holds the real chamber measurement fixed and perturbs only the document-side
claimed bound, so the deterministic reproducibility verdict transitions
pass -> inconclusive -> fail as the faithfully-extracted claim diverges from
physical reality. The mirror, on the claim axis, of
scripts/fault_injection.py (which injects faults on the process axis).

The pure sweep this script drives (`sweep_claimed_max`, `perturb_claimed_max`,
`TARGET_CLAIM_ID`) lives in `chamberbench.perturbation`, which the offline
test suite exercises directly; this script is only the driver around it, plus
the optional Tier B live run.

Run:
    uv run python scripts/perturbation.py --out /tmp/perturbation_sweep.json
    uv run python scripts/perturbation.py --out /tmp/perturbation_sweep.json --end-to-end
    uv run python scripts/perturbation.py --build-only

``--build-only`` builds and verifies the perturbed PDF that Tier B's live agent
run is graded against, then exits before any model call. It needs no
credentials -- it only downloads the source datasheet (if not already cached)
and edits it locally with pymupdf -- so a reader can inspect the experiment's
input for free, without paying for the agent run that consumes it.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from chamberbench.claims import TraceStep
from chamberbench.claimsio import corpus_dir, load_claim, short_path
from chamberbench.credentials import setup_credentials
from chamberbench.grading import evaluate_case
from chamberbench.harness.anthropic_path import (
    _resolve_pdf_to_local,
    extract_chamber_agentic,
)
from chamberbench.perturbation import TARGET_CLAIM_ID, sweep_claimed_max
from chamberbench.reproducibility import run_protocol, verdict

PERTURBED_PDF = corpus_dir() / "barometer_perturbed_945.pdf"


def _sweep_grid(measured: float, combined: float, n: int = 21) -> list[float]:
    """claimed_max values from solidly-pass (above measured) to solidly-fail."""
    hi = measured + 2.0 * combined
    lo = measured - 3.0 * combined
    return [round(float(x), 4) for x in np.linspace(hi, lo, n)]


def run_tier_a() -> dict[str, Any]:
    claim = load_claim(TARGET_CLAIM_ID)
    measurement = run_protocol(claim)  # real DPS310 cross-sensor measurement
    combined = measurement.measured_sigma  # spec_tol == 0 => combined == sigma
    grid = _sweep_grid(measurement.measured_value, combined)
    rows = sweep_claimed_max(claim, measurement, grid)
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "claim_id": claim.id,
        "measured_value": measurement.measured_value,
        "measured_unit": measurement.measured_unit,
        "measured_sigma": measurement.measured_sigma,
        "sigma_basis": measurement.measured_sigma_basis,
        "rows": rows,
    }


def _print_tier_a(result: dict[str, Any]) -> None:
    print("=" * 64)
    print("TIER A -- reproducibility under perturbation (dps310-operating-range)")
    print("=" * 64)
    print(
        "  measured "
        + format(result["measured_value"], ".4f")
        + " "
        + result["measured_unit"]
        + " +/- "
        + format(result["measured_sigma"], ".4f")
        + " ("
        + result["sigma_basis"]
        + ")"
    )
    for r in result["rows"]:
        print(
            "  claimed_max="
            + format(r["claimed_max"], ".4f")
            + "  divergence="
            + format(r["divergence"], "+.4f")
            + "  bnd_dist="
            + format(r["boundary_distance"], ".4f")
            + "  verdict="
            + r["verdict"]
        )
    print("=" * 64)


def _build_perturbed_datasheet(pdf_source: str) -> Path:
    """Build a perturbed copy of the DPS310 datasheet for the Tier B
    end-to-end demonstration: the operating-range upper bound 1200 -> 945
    hPa, so a faithful extraction (fidelity PASS) disagrees with the real
    chamber pressure (reproducibility FAIL). Controlled vehicle only -- not
    a claim that the real datasheet is wrong.
    """
    import pymupdf

    src = Path(_resolve_pdf_to_local(pdf_source))
    doc = pymupdf.open(src)
    total = 0
    for page in doc:
        # NOTE: every "1200" in this datasheet is a pressure-range value;
        # verify this assumption before adapting to another PDF.
        rects = page.search_for("1200")
        if not rects:
            continue
        for r in rects:
            page.add_redact_annot(r, text="945", fontsize=8)
        page.apply_redactions()
        total += len(rects)
    if total == 0:
        raise RuntimeError("could not find any '1200' in " + str(src))
    PERTURBED_PDF.parent.mkdir(parents=True, exist_ok=True)
    doc.save(PERTURBED_PDF)
    doc.close()
    return PERTURBED_PDF


def _verify_perturbed_datasheet() -> None:
    """Fail loudly if any page text still reads 1200, or 945 is missing from
    the headline (page 1) and Table 3 (page 6)."""
    import pymupdf

    doc = pymupdf.open(PERTURBED_PDF)
    texts = [page.get_text() for page in doc]
    doc.close()
    for i, t in enumerate(texts):
        assert "1200" not in t, "perturbed PDF still contains '1200' on page " + str(
            i + 1
        )
    assert "945" in texts[0], "page 1 text missing '945'"
    assert len(texts) >= 6, "perturbed PDF has fewer than 6 pages"
    assert "945" in texts[5], "page 6 (Table 3) text missing '945'"
    print(
        "verified: all '1200' pressure-range mentions now read 945 (pages 1 and 6 confirmed)"
    )


def _perturbed_claim() -> Any:
    base = load_claim(TARGET_CLAIM_ID)
    return base.model_copy(
        update={
            "claimed_max": 945.0,
            "value_contains": ["300", "945", "hPa"],
            "source_text": "Operation range: Pressure: 300 - 945 hPa. Temperature: -40 - 85 C.",
            "pdf_source": str(PERTURBED_PDF),
        }
    )


async def run_tier_b(model: str = "claudesonnet4.6") -> dict[str, Any]:
    """Run the real agent on the perturbed datasheet; grade fidelity and
    reproducibility. Expected: fidelity PASS, reproducibility fail."""
    # Ensure the perturbed datasheet exists (it is a gitignored generated artifact).
    if not PERTURBED_PDF.exists():
        base = load_claim(TARGET_CLAIM_ID)
        _build_perturbed_datasheet(base.pdf_source)
        _verify_perturbed_datasheet()

    setup_credentials()
    claim = _perturbed_claim()
    steps: list[TraceStep] = []
    result = await extract_chamber_agentic(claim, model=model, trace_sink=steps.append)

    expected = {
        "found": True,
        "confidence_min": claim.confidence_min,
        "value_contains": list(claim.value_contains),
    }
    fid = evaluate_case(result.extracted, expected)
    measurement = run_protocol(claim)  # real 945.285 hPa, unaffected by the bound
    v = verdict(claim, measurement)

    out = {
        "model": model,
        "fidelity_overall_pass": bool(fid["overall_pass"]),
        "reproducibility_verdict": v.verdict,
        "measured_value": measurement.measured_value,
        "claimed_max": claim.claimed_max,
        "delta": measurement.measured_value - claim.claimed_max,
        "combined_uncertainty": v.combined_uncertainty,
        "n_tool_calls": sum(1 for s in steps if s.kind == "tool_call"),
    }
    print(
        "  Tier B ["
        + model
        + "]: fidelity_pass="
        + str(out["fidelity_overall_pass"])
        + " reproducibility="
        + out["reproducibility_verdict"]
        + " (delta="
        + format(out["delta"], ".3f")
        + " hPa, combined="
        + format(out["combined_uncertainty"], ".4f")
        + ")"
    )
    return out


def build_only() -> Path:
    """Build the perturbed PDF and verify it, with no agent involved.

    This is the free half of Tier B: constructing the perturbed datasheet and
    checking that every "1200" pressure-range mention now reads "945" is pure
    local PDF editing plus a text-layer check, needing no credentials and no
    model call. Only *grading* the agent against this PDF (Tier B proper,
    ``run_tier_b``) is billable.
    """
    base = load_claim(TARGET_CLAIM_ID)
    path = _build_perturbed_datasheet(base.pdf_source)
    _verify_perturbed_datasheet()
    print(f"wrote {short_path(path)}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproducibility-perturbation experiment"
    )
    parser.add_argument(
        "--end-to-end",
        action="store_true",
        help="run the real agent on the perturbed datasheet and append a fidelity+reproducibility grade (Tier B)",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="build and verify the perturbed PDF, print where it was written, "
        "and exit before any model call or Tier A sweep; needs no credentials",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=False,
        help="output artifact path; the archive is read-only evidence and is "
        "never a valid target. Required unless --build-only is given.",
    )
    args = parser.parse_args()

    if args.build_only:
        build_only()
        return 0

    if args.out is None:
        parser.error("--out is required unless --build-only is given")

    result = run_tier_a()
    _print_tier_a(result)
    if args.end_to_end:
        import asyncio

        result["tier_b"] = asyncio.run(run_tier_b())
    args.out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {short_path(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
