"""Fidelity on a fourth component, to test whether the result generalises.

A reviewer asked whether the dispatch-instrumentation findings hold past the
three-component corpus. This runs the unmodified agentic
engine over `data/claims_a4988.yaml` -- 12 hand-curated claims from the
20-page Allegro A4988 motor driver already in the chamber's bill of materials
and already excluded from the paper on an agentic-navigation-stress rationale.

Fidelity only, and deliberately so: the chamber cannot stage a motor driver, so
the chamber tools are un-registered and no claim receives a reproducibility
verdict. What this measures is whether extraction and the dispatch record behave
on a document outside the corpus the rules were developed against.

Scored under both matchers. The substring matcher is what the paper uses, so it
is comparable with the frozen 25; exact numeric matching is reported because 7 of
these 18 numeric needles are two characters or fewer -- 39%, against the frozen
set's 27%, so this file leans on short needles harder than the set whose needles
we already conceded were weak.

Two things this document is not: unseen (it is the same PDF the rebuttal's
wrong-content arm served as a decoy, so both models had been observed
transcribing some of these rows before the claims were written) and hard (20
pages, the shortest in the corpus, which is why Appendix D excluded it). Report
the navigation mean against the multi-model fault-injection figures and expect
it to come in *lower*.

The claims in `data/claims_a4988.yaml` carry `pdf_source: "A4988"` -- a bare
part label, where `data/claims.yaml` carries a fetchable URL. It resolves to
nothing, so **`--pdf` is required**, exactly as `null_tool.py --mode
wrong-content` requires `--decoy`. Point it at a local copy of the 20-page
Allegro A4988 datasheet; it is mirrored in the Causal Chambers repository as
`hardware/datasheets/motor_driver.pdf`, and `docs/reproducing.md`'s corpus
table carries the revision and checksum. An unresolvable `pdf_source` is a
loud, pre-flight failure here -- it is a setup mistake, and recording twelve
`engine_error` cells and exiting 0 would present it as a result.

Run:
    uv run python scripts/fourth_component.py --model claudesonnet4.6 \
        --pdf /path/to/A4988.pdf --out /tmp/a4988_fidelity.json
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
from chamberbench.claimsio import A4988_CLAIMS_FILENAME, load_claims, short_path
from chamberbench.credentials import setup_credentials
from chamberbench.grading import evaluate_case
from chamberbench.harness.anthropic_path import (
    _resolve_pdf_to_local,
    extract_chamber_agentic,
)
from chamberbench.silent_failure import _DATASHEET_NAV_TOOLS, detect_silent_failure
from strict_fidelity_rescore import _score as _strict_score

TIMEOUT_S = 360

# Chamber tools are un-registered: this component has no chamber-side protocol.
_CHAMBER_TOOLS = frozenset(
    {
        "list_experiments",
        "get_experiment_metadata",
        "query_dataset",
        "cross_sensor_check",
        "run_simulator",
        "get_ground_truth_graph",
    }
)


def _score(claim: ClaimSpec, result: Any) -> tuple[dict[str, Any], bool]:
    """Substring fidelity (the paper's matcher) plus a strict re-score."""
    extracted = getattr(result, "extracted", None)
    expected = {
        "found": True,
        "confidence_min": claim.confidence_min,
        "value_contains": list(claim.value_contains),
    }
    fidelity = evaluate_case(extracted, expected)
    strict = False
    if extracted is not None:
        payload = (
            extracted.model_dump()
            if hasattr(extracted, "model_dump")
            else dict(extracted)
        )
        strict, _missed = _strict_score(payload, list(claim.value_contains))
    return fidelity, bool(strict)


async def _run_one(
    claim: ClaimSpec, model: str, sem: asyncio.Semaphore
) -> dict[str, Any]:
    async with sem:
        steps: list[TraceStep] = []
        engine_error = ""
        result = None
        t0 = time.monotonic()
        try:
            result = await asyncio.wait_for(
                extract_chamber_agentic(
                    claim,
                    model=model,
                    trace_sink=steps.append,
                    excluded_tools=_CHAMBER_TOOLS,
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
        strict = False
        if result is not None and not engine_error:
            fidelity, strict = _score(claim, result)

        cell = {
            "claim_id": claim.id,
            "fidelity": fidelity,
            "n_tool_calls_by_tool": dict(by_tool),
            "engine_error": engine_error,
        }
        report = detect_silent_failure(cell)
        status = (
            "ENGINE_ERR"
            if engine_error
            else ("PASS" if fidelity.get("overall_pass") else "fail")
        )
        print(
            f"    {claim.id:<38s} {status:<11s} {elapsed:6.1f}s  "
            f"strict={'Y' if strict else 'n'} tools={sum(by_tool.values()):<3d} flagged={report.flagged}",
            flush=True,
        )
        # Persist the extracted payload, not just the verdict. The first run of
        # this script stored pass booleans and tool counts only, which makes the
        # 12/12 impossible to re-score without paying for another stochastic run
        # -- the audit gap a reviewer correctly objected to. Everything the two
        # matchers read (`strict_fidelity_rescore._haystack`) is kept here so a
        # third party can recompute both scores offline.
        extracted = ((result.model_dump() if result is not None else {}) or {}).get(
            "extracted"
        ) or {}
        return {
            "claim_id": claim.id,
            "engine_error": engine_error,
            "latency_s": elapsed,
            "n_tool_calls_by_tool": dict(by_tool),
            "n_nav_calls": sum(
                v for k, v in by_tool.items() if k in _DATASHEET_NAV_TOOLS
            ),
            "fidelity_pass": bool(fidelity.get("overall_pass")),
            "strict_pass": strict,
            "fidelity_failure_reason": fidelity.get("failure_reason"),
            "detector_flagged": report.flagged,
            "detector_rules": report.rules,
            "extracted": extracted,
        }


def _resolve_corpus(
    claims: list[ClaimSpec], pdf: str, parser: argparse.ArgumentParser
) -> list[ClaimSpec]:
    """Point every claim at a resolvable PDF, or exit with a usable message.

    A missing corpus file is a setup mistake, not a result. Left to the run
    loop it becomes twelve `FileNotFoundError` cells swallowed into
    `engine_error`, a summary of twelve failures, and exit 0 -- an artifact
    that looks like an experiment and reports nothing. Checked once, up front,
    before a single billable call.
    """
    if pdf:
        path = Path(pdf)
        if not path.is_file():
            parser.error(f"--pdf: no such file: {pdf}")
        claims = [c.model_copy(update={"pdf_source": str(path)}) for c in claims]

    unresolvable = []
    for claim in claims:
        source = claim.pdf_source
        if source.lower().startswith(("http://", "https://")):
            # A URL is resolved (and cached) by the engine on first use;
            # checking it here would mean downloading during pre-flight.
            continue
        try:
            # The engine's own resolver, so this check cannot drift from it.
            # For a non-URL source it only tests existence -- no network.
            _resolve_pdf_to_local(source)
        except FileNotFoundError:
            unresolvable.append((claim.id, source))
    if unresolvable:
        shown = ", ".join(f"{cid} ({src!r})" for cid, src in unresolvable[:3])
        parser.error(
            f"{len(unresolvable)} of {len(claims)} claims have a pdf_source that "
            f"is neither a URL nor an existing file: {shown}"
            f"{' ...' if len(unresolvable) > 3 else ''}. "
            "data/claims_a4988.yaml carries the bare part label 'A4988' where "
            "data/claims.yaml carries a fetchable URL -- pass "
            "--pdf /path/to/A4988.pdf. The document is mirrored in the Causal "
            "Chambers repository as hardware/datasheets/motor_driver.pdf; see "
            "docs/reproducing.md for the revision and checksum."
        )
    return claims


async def main() -> int:
    parser = argparse.ArgumentParser(description="Fourth-component fidelity (A4988)")
    parser.add_argument("--model", default="claudesonnet4.6")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--pdf",
        default="",
        help="local path to the A4988 datasheet PDF. Required: the claim file's "
        "pdf_source is the bare part label 'A4988', which resolves to nothing. "
        "Mirrors null_tool.py's --decoy.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output artifact path; the archive is read-only evidence and is "
        "never a valid target",
    )
    args = parser.parse_args()

    # Corpus before credentials: both are setup mistakes, but this one is free
    # to check and is the one a reader hits after following the documented
    # install, so report it without requiring a key first.
    claims = _resolve_corpus(load_claims(A4988_CLAIMS_FILENAME), args.pdf, parser)
    setup_credentials()

    print(
        f"A4988 fidelity: {len(claims)} claims on {args.model} (concurrency={args.concurrency})",
        flush=True,
    )
    sem = asyncio.Semaphore(args.concurrency)
    cells = list(await asyncio.gather(*[_run_one(c, args.model, sem) for c in claims]))

    done = [c for c in cells if not c["engine_error"]]
    summary = {
        "claims": len(cells),
        "engine_errors": len(cells) - len(done),
        "fidelity_pass": sum(c["fidelity_pass"] for c in done),
        "strict_pass": sum(c["strict_pass"] for c in done),
        "flagged": sum(c["detector_flagged"] for c in done),
        # Report BOTH denominators: an all-tool mean quoted against a
        # navigation-only figure from another script compares different
        # quantities -- an error that made this component look like it
        # matched the clean-corpus dispatch profile when it is in fact cheaper.
        "mean_all_tool_calls_per_claim": (
            sum(sum(c["n_tool_calls_by_tool"].values()) for c in done) / len(done)
            if done
            else 0.0
        ),
        "mean_nav_calls_per_claim": (
            sum(c["n_nav_calls"] for c in done) / len(done) if done else 0.0
        ),
    }
    print()
    print("=" * 72)
    print(
        f"A4988 ({args.model}): fidelity {summary['fidelity_pass']}/{len(done)} "
        f"(strict {summary['strict_pass']}/{len(done)}), "
        f"engine errors {summary['engine_errors']}, "
        f"detector flags {summary['flagged']}, "
        f"nav calls/claim {summary['mean_nav_calls_per_claim']:.2f} "
        f"(all tools {summary['mean_all_tool_calls_per_claim']:.2f})"
    )
    print("=" * 72)

    args.out.write_text(
        json.dumps(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "model": args.model,
                "component": "A4988",
                "scope": "fidelity-only",
                "summary": summary,
                "cells": cells,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {short_path(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
