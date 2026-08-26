"""Multi-model recall arm for the silent-failure detector.

Arm A of ``scripts/fault_injection.py`` measures the detector's recall on
planted silent failures for Claude Sonnet 4.6 only. This script extends the
same recall measurement to the other benchmark models (Qwen3.6-27B and
GPT-5.1), so recall is reported across all three model families. Tool
exclusion is honoured on both the Anthropic-shape engine (Claude, Qwen) and
the OpenAI Responses-API engine (GPT-5.1).

Same two planted faults (F1 tool-bypass, F5 verification-skipped), same
detector, and the same clean-fidelity reuse as the original Arm A; only the
model and its per-model knobs vary. The output filename is built from
``--model`` -- see the module-level ``replace('.', '_')`` note below, which
is what produced the archived ``fault_injection_gpt-5_1.json`` and
``fault_injection_qwen3_6-27b.json``.

Run (needs gateway credentials via env or a project .env; for Qwen set
CHAMBER_QWEN_ENABLE_THINKING=false to avoid the reasoning-mode engine-error
storm documented in the paper):

    CHAMBER_QWEN_ENABLE_THINKING=false \\
        uv run python scripts/fault_injection_multimodel.py \\
            --model qwen3.6-27b --out /tmp/results
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

from chamberbench.credentials import setup_credentials
from chamberbench.harness import CHAMBER_MODEL_CONFIG
from chamberbench.harness.anthropic_path import extract_chamber_agentic
from chamberbench.silent_failure import detect_silent_failure
from fault_injection import (
    _FAULTS,
    TIMEOUT_S,
    _clean_agentic_cell,
    _load_baseline,
    _load_claims,
)


def _model_kwargs(model: str) -> dict[str, Any]:
    """Per-model engine knobs from the benchmark harness config (max_turns, detail)."""
    cfg = CHAMBER_MODEL_CONFIG.get(model, {})
    kwargs: dict[str, Any] = {}
    if "max_turns" in cfg:
        kwargs["max_turns"] = cfg["max_turns"]
    if "inspect_page_detail" in cfg:
        kwargs["inspect_page_detail"] = cfg["inspect_page_detail"]
    if "reasoning_effort" in cfg:
        kwargs["reasoning_effort"] = cfg["reasoning_effort"]
    return kwargs


async def _run_faulted(
    claim: Any,
    fault_id: str,
    excluded: frozenset[str],
    model: str,
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    """Run the agentic engine once with `excluded` un-registered; capture the trace."""
    async with sem:
        steps: list[Any] = []
        engine_error = ""
        t0 = time.monotonic()
        try:
            await asyncio.wait_for(
                extract_chamber_agentic(
                    claim,
                    model=model,
                    trace_sink=steps.append,
                    excluded_tools=excluded,
                    **_model_kwargs(model),
                ),
                timeout=TIMEOUT_S,
            )
        except TimeoutError as exc:
            engine_error = f"timeout after {TIMEOUT_S}s: {exc}"
        except Exception as exc:  # noqa: BLE001 -- record any engine failure, do not raise
            engine_error = f"{type(exc).__name__}: {exc}"
        elapsed = time.monotonic() - t0
        by_tool = Counter(
            s.tool_name for s in steps if s.kind == "tool_call" and s.tool_name
        )
        outcome = "engine_error" if engine_error else "ok"
        print(
            f"    {claim.id:<34s} [{model}] {fault_id}  {outcome:<12s} {elapsed:6.1f}s  tools={dict(by_tool)}",
            flush=True,
        )
        return {
            "claim_id": claim.id,
            "fault": fault_id,
            "engine_error": engine_error,
            "n_tool_calls_by_tool": dict(by_tool),
            "latency_s": elapsed,
        }


async def _run_arm_a(
    claims: list[Any], baseline: dict[str, Any], model: str, concurrency: int
) -> dict[str, Any]:
    """Run F1 and F5 over every claim for one model and score each with the detector."""
    sem = asyncio.Semaphore(concurrency)
    tasks = [
        _run_faulted(claim, fault_id, excluded, model, sem)
        for claim in claims
        for fault_id, (_label, excluded, _rule) in _FAULTS.items()
    ]
    runs = await asyncio.gather(*tasks)

    per_fault: dict[str, Any] = {}
    for fault_id, (label, _excluded, expected_rule) in _FAULTS.items():
        cells: list[dict[str, Any]] = []
        for run in (r for r in runs if r["fault"] == fault_id):
            clean = _clean_agentic_cell(baseline, run["claim_id"], model)
            clean_fidelity = (clean or {}).get("fidelity") or {}
            cell = {
                "claim_id": run["claim_id"],
                "fidelity": clean_fidelity,
                "n_tool_calls_by_tool": run["n_tool_calls_by_tool"],
                "engine_error": run["engine_error"],
            }
            report = detect_silent_failure(cell)
            cells.append(
                {
                    **run,
                    "fidelity_pass": bool(clean_fidelity.get("overall_pass")),
                    "detector_flagged": report.flagged,
                    "detector_rules": report.rules,
                }
            )
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
            "detector_caught": len(caught),
            "recall": (len(caught) / len(silent)) if silent else 0.0,
            "cells": cells,
        }
    return per_fault


async def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-model silent-failure recall")
    parser.add_argument(
        "--model",
        default="qwen3.6-27b",
        help="chamber model to fault (default qwen3.6-27b)",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="results directory; the archive is read-only evidence and is "
        "never a valid target. The output filename is built from --model "
        "inside this directory (see the module docstring)",
    )
    args = parser.parse_args()

    setup_credentials()
    claims = _load_claims()
    baseline = _load_baseline()
    print(
        f"Arm A ({args.model}): {len(claims)} claims x {len(_FAULTS)} faults = "
        f"{len(claims) * len(_FAULTS)} runs (concurrency={args.concurrency})",
        flush=True,
    )
    arm_a = await _run_arm_a(claims, baseline, args.model, args.concurrency)

    print()
    print("=" * 72)
    print(f"RECALL -- {args.model}")
    print("=" * 72)
    for fault_id, fr in arm_a.items():
        print(
            f"  {fault_id} {fr['label']}: planted={fr['planted']} engine_errors={fr['engine_errors']} "
            f"silent={fr['silent_failures']} detector_caught={fr['detector_caught']} recall={fr['recall']:.1%}"
        )

    # Filename construction preserved exactly: this is what produced the
    # archived fault_injection_gpt-5_1.json and fault_injection_qwen3_6-27b.json
    # names, now rooted at --out instead of a module-level RESULTS_DIR.
    out = args.out / f"fault_injection_{args.model.replace('.', '_')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "model": args.model,
                "arm_a": arm_a,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
