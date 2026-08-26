"""Null-tool injection: corrupt success with the tools actually called.

QWEN NOTE, learned the hard way. Running qwen3.6-27b here with reasoning ON
produces a 23-of-25 storm of ``SubmitToolNotCalledError`` -- the documented
upstream Qwen3 + vLLM interaction (QwenLM/Qwen3 #1817) where the model plans a
tool call inside its thinking block and then ends the turn without emitting the
tool tokens. That is an artifact of the configuration, NOT evidence about the
model's behaviour under degraded tools, and an earlier run of this script
mistook one for the other and reported Qwen as "attempted and excluded".
``CHAMBER_QWEN_ENABLE_THINKING=false`` is therefore forced for qwen unless the
caller overrides it explicitly, and the setting is recorded in the artifact so no
future reader has to guess which configuration produced a number. Check, too,
that the gateway you are pointed at actually serves qwen3.6-27b: an alias that
silently resolves to 3.5 produces a run that looks fine and is not the one you
asked for. See ``gateway/README.md``.


Rebuttal experiment for the EMNLP 2026 Industry Track submission. Reviewers
made the same objection about the sibling fault-injection experiment
(``scripts/fault_injection.py``): its planted faults *withhold* exactly the
tools the detector checks, so recall is guaranteed by construction, and
nothing measures the detector against the organic shape of corrupt success --
a run that calls the tools and answers from memory anyway.

This script constructs that case and measures it.

  Arm C (null-navigation). Every tool stays registered and callable. The
  datasheet-side tools are dispatched normally but their *output* is replaced
  with the tool's own natural empty-result message ("No matches found.", "No
  tables found on this page."). This models a silently degraded document-tool
  layer -- a broken index, an unparsable scan, a bad deploy -- which is a real
  production failure mode, not a contrived one. The agent therefore issues
  navigation and cross-check calls (so ``nav > 0`` and the cross-check
  predicate is satisfied) while receiving no document content at all.

Crucially, and unlike the planted faults in the sibling experiment, fidelity
here is scored on the run's **actual** extracted value. Nothing is pinned. A
cell that still passes fidelity did so from parametric memory, with no
document content in its context: an organic-shaped corrupt success. The
headline is therefore not a recall number the design guarantees, but two
measured quantities:

  * the corrupt-success rate -- how often the agent answers correctly anyway;
  * the current detector's recall on that class, which we expect to be ~0,
    because both rules are presence predicates and the tools were present.

A near-zero recall here is the honest quantification of the blind spot the
reviewers identify. It is reported as a negative result about the detector.

We also record, for each corrupt success, whether the submitted ``source_text``
is non-empty -- i.e. whether the model fabricated a quotation it never received.
That is the signal a future grounding rule would key on. We do NOT report a
recall for such a rule here: under nulled outputs nothing is groundable, so any
grounding rule scores 100% by construction, which is the very tautology this
experiment exists to avoid.

Run:
    uv run python scripts/null_tool.py --out /tmp/null_tool_injection.json
    uv run python scripts/null_tool.py --out /tmp/null_tool_injection.json --limit 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import Counter
from collections.abc import MutableMapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chamberbench.claims import ClaimSpec, TraceStep
from chamberbench.claimsio import load_claims, short_path
from chamberbench.credentials import setup_credentials
from chamberbench.grading import evaluate_case
from chamberbench.harness import anthropic_path
from chamberbench.harness.anthropic_path import extract_chamber_agentic
from chamberbench.silent_failure import detect_silent_failure

DEFAULT_MODEL = "claudesonnet4.6"
TIMEOUT_S = 360

# The datasheet-side tools whose output is nulled. Chamber tools are left
# untouched: under the two-pass freeze they run only after the extraction is
# submitted, so they cannot contribute document content to the answer.
_NULLED_TOOLS: dict[str, str] = {
    "build_datasheet": "Datasheet loaded: 0 sections indexed, 0 pages with extractable text.",
    "get_section_text": "No text found for that section.",
    "search_text": "No matches found.",
    "extract_table_markdown": "No tables found on this page.",
    "inspect_page": "No extractable content on this page.",
}

# Every document- and chamber-side tool. Closed-book un-registers all of them,
# so the agent has no channel to the document and must answer from memory.
_ALL_READ_TOOLS = frozenset(
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


def _install_wrong_content_dispatch(decoy_pdf: str) -> None:
    """Route every document tool to a DIFFERENT datasheet, keeping it healthy.

    This is the probe the reviewers named and the null arm did not run: the
    agent receives fluent, real datasheet prose that simply does not describe
    the part being asked about -- a mis-wired index or a wrong-document deploy,
    which we have actually seen in production. Unlike the null arm, the tools
    return *content*, so the model is misled rather than starved, and the
    decline cue ("No matches found.") is absent.

    A cell that still passes fidelity here did so despite every retrieved span
    being about the wrong component: an organic corrupt success with a fully
    healthy dispatch record.
    """
    from chamberbench.harness.datasheet_tools import _make_large_pdf_tools

    decoy_tools = _make_large_pdf_tools(decoy_pdf, inspect_page_detail="high")[0]
    decoy_by_name = {t.name: t for t in decoy_tools}
    real_build = anthropic_path._build_two_pass_state

    def _decoy_build(
        pdf_tools: list[Any], ch_tools: list[Any], excluded: frozenset[str]
    ) -> Any:
        state = real_build(pdf_tools, ch_tools, excluded)
        for tool in state.pass1_tools:
            decoy = decoy_by_name.get(getattr(tool, "name", ""))
            if decoy is not None:
                tool.call = decoy.call
        return state

    anthropic_path._build_two_pass_state = _decoy_build
    try:
        from chamberbench.harness import openai_path

        openai_path._build_two_pass_state = _decoy_build
    except ImportError:  # pragma: no cover -- optional provider path
        pass


def _install_null_dispatch() -> None:
    """Replace document-tool *output* while leaving the tools registered.

    Patching ``_dispatch_tool`` rather than the tool factory is deliberate: the
    tool list the model sees, the schemas, and the prompt are all untouched, so
    the agent's decision to call a tool is unperturbed. Only the content coming
    back changes. The call still appears in the dispatch record, which is the
    whole point -- the detector's presence predicates must see a healthy trace.
    """
    real_dispatch = anthropic_path._dispatch_tool

    async def _null_dispatch(tool_use: Any, by_name: dict[str, Any]) -> Any:
        replacement = _NULLED_TOOLS.get(tool_use.name)
        if replacement is not None:
            return replacement
        return await real_dispatch(tool_use, by_name)

    anthropic_path._dispatch_tool = _null_dispatch

    # The Responses-API engine (GPT-5.1) does not route through
    # `_dispatch_tool` -- it invokes `by_name[name].call(...)` inline. Both
    # engines do build their tool lists through `_build_two_pass_state`, so
    # nulling each tool's `.call` there covers both transports with one seam.
    # Name, description and input_schema are untouched, so the tool surface the
    # model sees is identical and its decision to call is unperturbed.
    real_build = anthropic_path._build_two_pass_state

    def _null_build(
        pdf_tools: list[Any], ch_tools: list[Any], excluded: frozenset[str]
    ) -> Any:
        state = real_build(pdf_tools, ch_tools, excluded)
        for tool in (*state.pass1_tools, *state.pass2_tools):
            replacement = _NULLED_TOOLS.get(getattr(tool, "name", ""))
            if replacement is not None:
                tool.call = _make_null_call(replacement)
        return state

    anthropic_path._build_two_pass_state = _null_build
    try:
        from chamberbench.harness import openai_path

        # The OpenAI module imported the symbol at load time, so its own
        # namespace needs the replacement too.
        openai_path._build_two_pass_state = _null_build
    except ImportError:  # pragma: no cover -- optional provider path
        pass


def _make_null_call(replacement: str) -> Any:
    async def _call(kwargs: dict[str, Any]) -> str:
        del kwargs  # the whole point: the arguments are ignored, no content is returned
        return replacement

    return _call


def _score_fidelity(claim: ClaimSpec, result: Any) -> dict[str, Any]:
    """Grade the run's real extracted value, exactly as the benchmark does."""
    expected = {
        "found": True,
        "confidence_min": claim.confidence_min,
        "value_contains": list(claim.value_contains),
    }
    extracted = getattr(result, "extracted", None)
    return evaluate_case(extracted, expected)


def _submitted_source_text(result: Any) -> str:
    extracted = getattr(result, "extracted", None)
    if extracted is None:
        return ""
    return (getattr(extracted, "source_text", "") or "").strip()


async def _run_one(
    claim: ClaimSpec,
    model: str,
    sem: asyncio.Semaphore,
    excluded: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    async with sem:
        steps: list[TraceStep] = []
        engine_error = ""
        result = None
        t0 = time.monotonic()
        try:
            result = await asyncio.wait_for(
                extract_chamber_agentic(
                    claim, model=model, trace_sink=steps.append, excluded_tools=excluded
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
        fidelity: dict[str, Any] = {"overall_pass": False}
        source_text = ""
        if result is not None and not engine_error:
            fidelity = _score_fidelity(claim, result)
            source_text = _submitted_source_text(result)

        cell = {
            "claim_id": claim.id,
            "fidelity": fidelity,
            "n_tool_calls_by_tool": dict(by_tool),
            "engine_error": engine_error,
        }
        report = detect_silent_failure(cell)

        nav = sum(v for k, v in by_tool.items() if k in _NULLED_TOOLS)
        xcheck = sum(
            v
            for k, v in by_tool.items()
            if k in ("search_text", "extract_table_markdown")
        )
        status = (
            "ENGINE_ERR"
            if engine_error
            else ("FID_PASS" if fidelity.get("overall_pass") else "fid_fail")
        )
        print(
            f"    {claim.id:<34s} {status:<11s} {elapsed:6.1f}s  "
            f"nav={nav:<3d} xcheck={xcheck:<3d} flagged={report.flagged}",
            flush=True,
        )
        return {
            "claim_id": claim.id,
            "engine_error": engine_error,
            "latency_s": elapsed,
            "n_tool_calls_by_tool": dict(by_tool),
            "nav_calls": nav,
            "crosscheck_calls": xcheck,
            "fidelity_pass": bool(fidelity.get("overall_pass")),
            "fidelity_failure_reason": fidelity.get("failure_reason"),
            "detector_flagged": report.flagged,
            "detector_rules": report.rules,
            "submitted_source_text": source_text[:400],
            "fabricated_quotation": bool(source_text),
        }


def _render(cells: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [c for c in cells if not c["engine_error"]]
    engine_errors = [c for c in cells if c["engine_error"]]
    # A corrupt success: the answer passed fidelity although no document
    # content ever reached the model.
    corrupt = [c for c in completed if c["fidelity_pass"]]
    caught = [c for c in corrupt if c["detector_flagged"]]
    # Did the trace look healthy to the presence predicates?
    # Computed over ALL completed runs, not just corrupt ones: with zero corrupt
    # successes the old per-corrupt count stored 0 and contradicted the reported
    # "25/25 healthy trace". Both were true of different populations; only this
    # one is the population the claim is about.
    healthy_trace = [
        c for c in completed if c["nav_calls"] > 0 and c["crosscheck_calls"] > 0
    ]
    fabricated = [c for c in corrupt if c["fabricated_quotation"]]

    summary = {
        "planted": len(cells),
        "engine_errors": len(engine_errors),
        "completed": len(completed),
        "corrupt_successes": len(corrupt),
        "corrupt_success_rate": (len(corrupt) / len(completed)) if completed else 0.0,
        "detector_caught": len(caught),
        # Undefined, not zero, when no corrupt success occurred: the detector is
        # only defined over fidelity-passing cells, so an empty corrupt set means
        # no run ever reached a rule evaluation.
        "detector_recall": (len(caught) / len(corrupt)) if corrupt else None,
        "completed_with_healthy_trace": len(healthy_trace),
        "corrupt_with_fabricated_quotation": len(fabricated),
    }

    print()
    print("=" * 72)
    print("NULL-TOOL INJECTION -- corrupt success with the tools actually called")
    print("=" * 72)
    print(f"  planted runs           : {summary['planted']}")
    print(f"  engine errors (loud)   : {summary['engine_errors']}")
    print(f"  completed runs         : {summary['completed']}")
    print()
    print(
        f"  CORRUPT SUCCESSES      : {summary['corrupt_successes']}/{summary['completed']}"
        f"  ({summary['corrupt_success_rate']:.0%} of completed runs)"
    )
    print(
        f"  completed runs with healthy trace (nav>0, cross-check>0): "
        f"{summary['completed_with_healthy_trace']}/{summary['completed']}"
    )
    print(
        f"    of which submitted a fabricated quotation             : {summary['corrupt_with_fabricated_quotation']}"
    )
    print()
    if summary["detector_recall"] is None:
        print(
            "  DETECTOR RECALL on this class: UNDEFINED -- zero fidelity-passing cells,"
        )
        print(
            "    so no run ever entered the detector's domain. This is not 0% recall."
        )
    else:
        print(
            f"  DETECTOR RECALL on this class: {summary['detector_caught']}/"
            f"{summary['corrupt_successes']} = {summary['detector_recall']:.0%}"
        )
    print()
    print("  (Recall on the rule-aligned planted faults of")
    print("   scripts/fault_injection.py is 50/50, by construction.)")
    print("=" * 72)
    return summary


def resolve_qwen_thinking(
    model: str, env: MutableMapping[str, str] | None = None
) -> bool:
    """Force reasoning off for qwen unless the caller has said otherwise.

    See the module docstring: with reasoning on, this arm measures an upstream
    tool-call bug (QwenLM/Qwen3 #1817) rather than the model, and it does so
    silently. Three behaviours, all of which
    ``test_null_tool_forces_thinking_off_for_qwen`` drives:

    * a qwen model with the variable unset -- forced to ``"false"``;
    * a qwen model with the variable already set -- left alone, because the
      archive ships a deliberate ``thinking_on`` arm produced that way;
    * a non-qwen model -- the variable is not written at all, so nothing is
      forced onto an engine the bug does not affect.

    Writes into the real ``os.environ`` by default because
    ``chamberbench.harness.anthropic_path`` reads the variable back from there
    at call time; ``env`` exists so the decision can be exercised without
    mutating the process.

    Returns whether reasoning is enabled for this run, for the run banner and
    for the ``qwen_enable_thinking`` field recorded in the artifact.
    """
    if env is None:
        env = os.environ
    if "qwen" in model.lower() and "CHAMBER_QWEN_ENABLE_THINKING" not in env:
        env["CHAMBER_QWEN_ENABLE_THINKING"] = "false"
        print(
            "note: forcing CHAMBER_QWEN_ENABLE_THINKING=false (upstream Qwen3+vLLM tool-call bug)"
        )
    return env.get("CHAMBER_QWEN_ENABLE_THINKING", "true").strip().lower() != "false"


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Null-tool injection (rebuttal experiment)"
    )
    parser.add_argument(
        "--mode",
        default="null",
        choices=("null", "closed-book", "wrong-content"),
        help=(
            "null: document tools return empty-result strings. "
            "closed-book: document tools un-registered entirely, so the model must "
            "answer from memory -- measures P(memory correct), the base rate the "
            "corrupt-success class depends on. "
            "wrong-content: document tools serve a DIFFERENT datasheet."
        ),
    )
    parser.add_argument(
        "--decoy", default="", help="decoy PDF for --mode wrong-content"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output artifact path; the archive is read-only evidence and is "
        "never a valid target",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="run only the first N claims (smoke test)"
    )
    args = parser.parse_args()

    setup_credentials()
    excluded: frozenset[str] = frozenset()
    if args.mode == "null":
        _install_null_dispatch()
    elif args.mode == "wrong-content":
        if not args.decoy:
            parser.error("--mode wrong-content requires --decoy <pdf>")
        _install_wrong_content_dispatch(args.decoy)
    else:  # closed-book -- no patching; the tools simply are not registered
        excluded = _ALL_READ_TOOLS

    claims = load_claims()
    if args.limit:
        claims = claims[: args.limit]

    thinking = resolve_qwen_thinking(args.model)

    print(
        f"mode={args.mode}: {len(claims)} claims on {args.model} (concurrency={args.concurrency})"
        + (f", decoy={args.decoy}" if args.decoy else "")
        + (f", enable_thinking={thinking}" if "qwen" in args.model.lower() else ""),
        flush=True,
    )
    sem = asyncio.Semaphore(args.concurrency)
    cells = await asyncio.gather(
        *[_run_one(c, args.model, sem, excluded) for c in claims]
    )

    summary = _render(list(cells))
    args.out.write_text(
        json.dumps(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "model": args.model,
                "mode": args.mode,
                "decoy": args.decoy,
                # Recorded because an engine-error rate is meaningless without it.
                "qwen_enable_thinking": thinking
                if "qwen" in args.model.lower()
                else None,
                "nulled_tools": sorted(_NULLED_TOOLS),
                "summary": summary,
                "cells": list(cells),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote {short_path(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
