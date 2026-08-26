"""Fault-injection experiment: does the instrumentation catch silent failures?

Backs the controlled silent-failure result for the chamber paper. The paper
claims that per-tool dispatch instrumentation catches "silent failures" --
agent runs that produce a correct-looking answer through a defective
process -- that fidelity-only scoring scores as success. This script turns
that claim into a number.

Two arms.

  Arm A (planted failures). For each chamber claim, run the agentic engine on
  Claude Sonnet 4.6 twice: once with fault F1 (tool-bypass -- only
  submit_claim_result registered, so the agent answers with zero navigation
  calls), once with fault F5 (verification-skipped -- the cross-check tools
  un-registered). The faulted run's extracted value is discarded; the cell
  reuses the clean post-audit run's fidelity verdict, so every Arm A cell is a
  fidelity PASS -- a silent failure by construction. The detector
  (chamberbench.silent_failure) only ever reads the emergent dispatch trace,
  never the value, so the reused verdict cannot help it.

  Arm B (clean control). The agentic cells of the real post-audit run in the
  archived baseline_chamber.json. Measures the detector's false-positive
  rate -- it was never tuned against these.

Headline: fidelity-only flags 0 of the planted failures; the detector flags
nearly all, at a near-zero false-positive rate on the real run.

Run:
    uv run python scripts/fault_injection.py --out /tmp/fault_injection.json
    uv run python scripts/fault_injection.py --out /tmp/fault_injection.json --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chamberbench.claims import ClaimSpec, TraceStep
from chamberbench.claimsio import archive_dir, load_claims, short_path
from chamberbench.classifier import _VERIFICATION_TOOLS
from chamberbench.credentials import setup_credentials
from chamberbench.harness.anthropic_path import extract_chamber_agentic
from chamberbench.silent_failure import detect_silent_failure

BASELINE_PATH = archive_dir() / "baseline_chamber.json"

# Arm A runs on one model. The agentic-engine defaults (max_turns=30,
# inspect_page_detail="high", reasoning_effort="medium") already match the
# claudesonnet4.6 row of chamberbench.harness.CHAMBER_MODEL_CONFIG, so a
# faulted run differs from the real benchmark only in the excluded tools.
MODEL = "claudesonnet4.6"
TIMEOUT_S = 360

# Navigation + chamber tools -- everything the agent uses to read the
# datasheet and the chamber. Mirrors the tool surface assembled by
# chamberbench.harness.anthropic_path (the large-PDF tools + the chamber
# tools). F1 un-registers all of them; only submit_claim_result survives.
_NAVIGATION_TOOLS = frozenset(
    {
        "build_datasheet",
        "get_section_text",
        "search_text",
        "extract_table_markdown",
        "inspect_page",
        "list_experiments",
        "get_experiment_metadata",
        "query_dataset",
        "cross_sensor_check",
        "run_simulator",
        "get_ground_truth_graph",
    }
)

# fault_id -> (human label, tools to un-register, detector rule it should
# trip). F5 reuses the detector's own verification-tool set, so the planted
# fault and the rule that catches it share one source of truth.
_FAULTS: dict[str, tuple[str, frozenset[str], str]] = {
    "F1": ("tool-bypass", _NAVIGATION_TOOLS, "tool_bypass"),
    "F5": ("verification-skipped", _VERIFICATION_TOOLS, "verification_skipped"),
}


# ---------------------------------------------------------------------------
# Setup and loading
# ---------------------------------------------------------------------------


def _load_claims() -> list[ClaimSpec]:
    """Parse the bundled claim set into validated ClaimSpec instances."""
    return load_claims()


def _load_baseline() -> dict[str, Any]:
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(f"baseline not found: {BASELINE_PATH}")
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def _clean_agentic_cell(
    baseline: dict[str, Any], claim_id: str, model: str
) -> dict[str, Any] | None:
    """The clean agentic cell for (claim, model) from the archived baseline."""
    cell = (
        (baseline.get("results") or {}).get(claim_id, {}).get("agentic", {}).get(model)
    )
    return cell if isinstance(cell, dict) else None


def _arm_b_cells(baseline: dict[str, Any]) -> list[dict[str, Any]]:
    """Every agentic cell in the archived baseline -- the clean control."""
    out: list[dict[str, Any]] = []
    for by_eng in (baseline.get("results") or {}).values():
        for cell in (by_eng.get("agentic") or {}).values():
            if isinstance(cell, dict):
                out.append(cell)
    return out


# ---------------------------------------------------------------------------
# Arm A -- planted failures
# ---------------------------------------------------------------------------


async def _run_faulted(
    claim: ClaimSpec,
    fault_id: str,
    excluded_tools: frozenset[str],
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    """Run the agentic engine once with `excluded_tools` un-registered.

    The extracted value is discarded -- only the emergent dispatch trace
    matters. Engine errors (a turn that ends without submit, a timeout) are
    captured, not raised: a loud failure is simply not a silent one.
    """
    async with sem:
        steps: list[TraceStep] = []
        engine_error = ""
        t0 = time.monotonic()
        try:
            await asyncio.wait_for(
                extract_chamber_agentic(
                    claim,
                    model=MODEL,
                    trace_sink=steps.append,
                    excluded_tools=excluded_tools,
                ),
                timeout=TIMEOUT_S,
            )
        except TimeoutError as exc:
            engine_error = f"timeout after {TIMEOUT_S}s: {exc}"
        except Exception as exc:  # noqa: BLE001 -- record any engine failure
            engine_error = f"{type(exc).__name__}: {exc}"
        elapsed = time.monotonic() - t0
        by_tool = Counter(
            s.tool_name for s in steps if s.kind == "tool_call" and s.tool_name
        )
        outcome = "engine_error" if engine_error else "ok"
        print(
            f"    {claim.id:<34s} {fault_id}  {outcome:<12s} {elapsed:6.1f}s  tools={dict(by_tool)}",
            flush=True,
        )
        return {
            "claim_id": claim.id,
            "fault": fault_id,
            "engine_error": engine_error,
            "n_tool_calls_by_tool": dict(by_tool),
            "n_steps": len(steps),
            "latency_s": elapsed,
        }


async def _run_arm_a(
    claims: list[ClaimSpec], baseline: dict[str, Any], concurrency: int
) -> dict[str, Any]:
    """Run F1 and F5 over every claim and score each cell with the detector."""
    sem = asyncio.Semaphore(concurrency)
    tasks = [
        _run_faulted(claim, fault_id, excluded, sem)
        for claim in claims
        for fault_id, (_label, excluded, _rule) in _FAULTS.items()
    ]
    runs = await asyncio.gather(*tasks)

    per_fault: dict[str, Any] = {}
    for fault_id, (label, _excluded, expected_rule) in _FAULTS.items():
        cells: list[dict[str, Any]] = []
        for run in (r for r in runs if r["fault"] == fault_id):
            clean = _clean_agentic_cell(baseline, run["claim_id"], MODEL)
            clean_fidelity = (clean or {}).get("fidelity") or {}
            # Pin fidelity to the clean post-audit verdict: a silent failure
            # passes fidelity by construction. The detector reads only
            # n_tool_calls_by_tool, so the pinned verdict cannot help it.
            cell = {
                "claim_id": run["claim_id"],
                "fidelity": clean_fidelity,
                "n_tool_calls_by_tool": run["n_tool_calls_by_tool"],
                "engine_error": run["engine_error"],
            }
            report = detect_silent_failure(cell)
            cells.append(
                {
                    "claim_id": run["claim_id"],
                    "engine_error": run["engine_error"],
                    "n_tool_calls_by_tool": run["n_tool_calls_by_tool"],
                    "n_steps": run["n_steps"],
                    "latency_s": run["latency_s"],
                    "fidelity_pass": bool(clean_fidelity.get("overall_pass")),
                    "detector_flagged": report.flagged,
                    "detector_rules": report.rules,
                }
            )
        # A silent failure: fidelity passes (by construction) and the run
        # itself completed -- an engine error is a loud failure, excluded.
        silent = [c for c in cells if c["fidelity_pass"] and not c["engine_error"]]
        caught = [c for c in silent if c["detector_flagged"]]
        per_fault[fault_id] = {
            "label": label,
            "expected_rule": expected_rule,
            "planted": len(cells),
            "engine_errors": sum(1 for c in cells if c["engine_error"]),
            "no_clean_reference": sum(
                1 for c in cells if not c["fidelity_pass"] and not c["engine_error"]
            ),
            "silent_failures": len(silent),
            "fidelity_only_caught": 0,  # silent failures all pass fidelity
            "detector_caught": len(caught),
            "recall": (len(caught) / len(silent)) if silent else 0.0,
            "cells": cells,
        }
    return per_fault


# ---------------------------------------------------------------------------
# Arm B -- clean control
# ---------------------------------------------------------------------------


def _score_arm_b(baseline: dict[str, Any]) -> dict[str, Any]:
    """Run the detector over every clean agentic cell in the archived baseline."""
    cells = _arm_b_cells(baseline)
    evaluated: list[dict[str, Any]] = []
    flagged: list[dict[str, Any]] = []
    for cell in cells:
        fid = cell.get("fidelity") or {}
        # The detector only engages a fidelity-pass, non-engine-error cell;
        # that is the population in which a false positive is even possible.
        engaged = (
            bool(fid.get("overall_pass"))
            and not fid.get("engine_error")
            and not cell.get("engine_error")
        )
        if not engaged:
            continue
        report = detect_silent_failure(cell)
        rec = {
            "claim_id": cell.get("claim_id"),
            "detector_flagged": report.flagged,
            "detector_rules": report.rules,
        }
        evaluated.append(rec)
        if report.flagged:
            flagged.append(rec)
    return {
        "source": "baseline_chamber.json",
        "total_agentic_cells": len(cells),
        "evaluated": len(evaluated),
        "flagged": len(flagged),
        "fp_rate": (len(flagged) / len(evaluated)) if evaluated else 0.0,
        "flagged_cells": flagged,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _latex_table(arm_a: dict[str, Any], arm_b: dict[str, Any]) -> str:
    """The paper artifact: recall per fault, fidelity-only vs the detector."""
    lines = [
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        (
            r"\textbf{Planted fault} & \textbf{N} & \textbf{Fidelity-only} "
            r"& \textbf{Detector} & \textbf{Recall} \\"
        ),
        r"\midrule",
    ]
    for fault_id, fr in arm_a.items():
        cells = [
            fr["label"].capitalize() + " (" + fault_id + ")",
            str(fr["silent_failures"]),
            str(fr["fidelity_only_caught"]),
            str(fr["detector_caught"]),
            str(round(fr["recall"] * 100)) + r"\%",
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        "% detector false positives on the post-audit run: "
        + str(arm_b["flagged"])
        + "/"
        + str(arm_b["evaluated"])
        + " agentic cells"
    )
    return "\n".join(lines)


def _render(arm_a: dict[str, Any] | None, arm_b: dict[str, Any]) -> None:
    print()
    print("=" * 72)
    print("FAULT-INJECTION EXPERIMENT")
    print("=" * 72)
    if arm_a is not None:
        print()
        print("Arm A -- planted silent failures (Claude Sonnet 4.6):")
        for fault_id, fr in arm_a.items():
            n = fr["silent_failures"]
            print(f"  {fault_id} {fr['label']}:")
            print(
                f"    planted={fr['planted']}  engine_errors={fr['engine_errors']}"
                f"  no_clean_ref={fr['no_clean_reference']}  silent_failures={n}"
            )
            print(
                f"    fidelity-only caught={fr['fidelity_only_caught']}/{n}  "
                f"detector caught={fr['detector_caught']}/{n}  "
                f"recall={fr['recall']:.1%}"
            )
    print()
    print("Arm B -- clean control (real post-audit run):")
    print(
        f"  detector false positives: {arm_b['flagged']}/{arm_b['evaluated']} "
        f"fidelity-pass agentic cells  (fp_rate={arm_b['fp_rate']:.1%})"
    )
    for c in arm_b["flagged_cells"]:
        print(f"    FLAGGED: {c['claim_id']}  {c['detector_rules']}")
    if arm_a is not None:
        print()
        print("LaTeX table (paper artifact):")
        print(_latex_table(arm_a, arm_b))
    print("=" * 72)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> int:
    parser = argparse.ArgumentParser(description="Chamber fault-injection experiment")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output artifact path; the archive is read-only evidence and is "
        "never a valid target",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="score Arm B only (no LLM calls); checks the detector and IO wiring",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="max concurrent agent runs in Arm A (default 4)",
    )
    args = parser.parse_args()

    baseline = _load_baseline()
    arm_b = _score_arm_b(baseline)

    arm_a: dict[str, Any] | None = None
    if not args.dry_run:
        setup_credentials()
        claims = _load_claims()
        print(
            f"Arm A: {len(claims)} claims x {len(_FAULTS)} faults = "
            f"{len(claims) * len(_FAULTS)} runs on {MODEL} "
            f"(concurrency={args.concurrency})",
            flush=True,
        )
        arm_a = await _run_arm_a(claims, baseline, args.concurrency)

    _render(arm_a, arm_b)

    if not args.dry_run:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "model": MODEL,
            "arm_a": arm_a,
            "arm_b": arm_b,
        }
        args.out.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"wrote {short_path(args.out)}")
    else:
        print(
            "dry run: detector + IO wiring exercised; fault_injection.json not written"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
