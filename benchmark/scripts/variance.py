"""Chamber variance experiment: repeated runs for run-to-run variance.

The chamber paper reports every cost / latency / fidelity number from a
single run; this script runs the agentic engine N times over all 25 claims
x 3 models so the paper can report mean +/- std and defend each point
estimate.

Repeat 1 is the existing post-audit run already archived in
baseline_chamber.json -- imported read-only, not re-run. Repeats 2-3 are
run live. Reproducibility is agent-independent (run_protocol / verdict
take only the ClaimSpec), so it is not re-run: variance is a purely
agent-side story (fidelity, confidence, latency, cost).

The pure aggregation this script drives (`aggregate_variance`,
`import_repeat_one`) lives in `chamberbench.variance`, which the offline
test suite exercises directly; this script is only the live-run driver
around it -- it makes real, billable calls against the model gateway.

Run:
    uv run python scripts/variance.py --out /tmp/variance_chamber.json
    uv run python scripts/variance.py --out /tmp/variance_chamber.json --dry-run
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

from chamberbench import harness
from chamberbench.claims import ClaimSpec, TraceStep
from chamberbench.claimsio import archive_dir, load_claims, short_path
from chamberbench.grading import evaluate_case
from chamberbench.harness.anthropic_path import extract_chamber_agentic
from chamberbench.variance import aggregate_variance, import_repeat_one

BASELINE_PATH = archive_dir() / "baseline_chamber.json"
MODELS = ("claudesonnet4.6", "gpt-5.1", "qwen3.6-27b")
_DEFAULT_FRESH_REPEATS = 2
_DEFAULT_TIMEOUT_S = 360

# Per-model max concurrent agentic runs. qwen3.6-27b is a self-hosted,
# single-pod vLLM deployment: concurrent agentic loops degrade its
# generation (the model ends a turn without emitting the
# submit_claim_result tool call), so it runs sequentially. The
# API-backed providers handle concurrency fine. A positive --concurrency
# overrides this for every model.
_MODEL_CONCURRENCY = {"claudesonnet4.6": 4, "gpt-5.1": 4, "qwen3.6-27b": 1}
_FALLBACK_CONCURRENCY = 4


# ---------------------------------------------------------------------------
# Live run
# ---------------------------------------------------------------------------


def _score_fidelity(claim: ClaimSpec, claim_result: Any) -> dict[str, Any]:
    """Fidelity via the shared grading surface (chamberbench.grading)."""
    expected = {
        "found": True,
        "confidence_min": claim.confidence_min,
        "value_contains": list(claim.value_contains),
    }
    return evaluate_case(claim_result.extracted, expected)


def _model_cfg(model: str) -> dict[str, Any]:
    """Per-model agentic knobs, from the shared CHAMBER_MODEL_CONFIG."""
    cfg = harness.model_config(model)
    return {
        "max_turns": cfg["max_turns"],
        "inspect_page_detail": cfg["inspect_page_detail"],
        "reasoning_effort": cfg.get("reasoning_effort", "medium"),
    }


def _resolve_concurrency(model: str, override: int) -> int:
    """Max concurrent runs for `model`; a positive `override` wins for all."""
    if override > 0:
        return override
    return _MODEL_CONCURRENCY.get(model, _FALLBACK_CONCURRENCY)


async def _run_cell(
    claim: ClaimSpec,
    model: str,
    cfg: dict[str, Any],
    timeout_s: int,
    sem: asyncio.Semaphore,
) -> tuple[str, dict[str, Any]]:
    """Run one agentic cell and project it into the variance cell shape.

    Engine errors (a timeout, a turn ending without submit) are captured,
    not raised: a loud failure still counts as a failed repeat.
    """
    async with sem:
        steps: list[TraceStep] = []
        engine_error = ""
        claim_result: Any = None
        t0 = time.monotonic()
        try:
            claim_result = await asyncio.wait_for(
                extract_chamber_agentic(
                    claim,
                    model=model,
                    max_turns=cfg["max_turns"],
                    trace_sink=steps.append,
                    inspect_page_detail=cfg["inspect_page_detail"],
                    reasoning_effort=cfg["reasoning_effort"],
                ),
                timeout=timeout_s,
            )
        except TimeoutError as exc:
            engine_error = f"timeout after {timeout_s}s: {exc}"
        except Exception as exc:  # noqa: BLE001 -- record any engine failure
            engine_error = f"{type(exc).__name__}: {exc}"
        elapsed = time.monotonic() - t0

        by_tool = Counter(
            s.tool_name for s in steps if s.kind == "tool_call" and s.tool_name
        )
        usage = harness.rollup_cell_usage(steps)
        if claim_result is None:
            fidelity = {
                "found_expected": True,
                "found_actual": False,
                "found_correct": False,
                "value_pass": False,
                "confidence": 0.0,
                "failure_reason": engine_error or "no result",
                "overall_pass": False,
                "engine_error": True,
            }
        else:
            fidelity = _score_fidelity(claim, claim_result)
        cell = {
            "fidelity": fidelity,
            "latency_s": elapsed,
            "usage": usage,
            "engine_error": engine_error,
            "n_tool_calls_by_tool": dict(by_tool),
            "n_steps": len(steps),
        }
        outcome = (
            "engine_error"
            if engine_error
            else ("pass" if fidelity.get("overall_pass") else "fail")
        )
        print(
            f"    {claim.id:<34s} {model:<18s} {outcome:<12s} {elapsed:6.1f}s",
            flush=True,
        )
        if engine_error:
            print(f"      engine_error: {engine_error}", flush=True)
        return claim.id, cell


def _carried_forward_runs(
    existing: dict[str, list[dict[str, Any]]],
    models_in_run: list[str] | tuple[str, ...],
) -> dict[str, list[dict[str, Any]]]:
    """Models from a prior variance artifact this run does not cover.

    Lets a partial re-run (e.g. just qwen3.6-27b after an infra outage)
    merge into the existing artifact instead of clobbering it.
    """
    in_run = set(models_in_run)
    return {m: reps for m, reps in existing.items() if m not in in_run}


def _load_existing_runs(out_path: Path) -> dict[str, list[dict[str, Any]]]:
    """The `runs` block of an existing variance artifact at `out_path`, or {}."""
    if not out_path.exists():
        return {}
    try:
        return json.loads(out_path.read_text(encoding="utf-8")).get("runs", {})
    except (json.JSONDecodeError, OSError):
        return {}


class _Accumulator:
    """Holds runs across all models and writes the variance artifact durably.

    The file is rewritten after every fresh (model, repeat) completes, so a
    crash loses at most one repeat's worth of one model. Writes are
    serialised with a lock since the three model tasks run concurrently.
    A run covering only a subset of models merges into a prior artifact
    rather than clobbering the models it does not touch.
    """

    def __init__(
        self,
        models: list[str],
        claim_ids: list[str],
        n_repeats: int,
        out_path: Path,
    ) -> None:
        self.models = list(models)
        self.claim_ids = list(claim_ids)
        self.n_repeats = n_repeats
        self.out_path = out_path
        self.runs: dict[str, list[dict[str, Any]]] = {}
        self._lock = asyncio.Lock()

    def seed(self, repeat_one: dict[str, dict[str, Any]]) -> None:
        # Carry forward models from a prior artifact this run does not
        # cover, so a partial re-run merges instead of clobbering.
        self.runs = _carried_forward_runs(
            _load_existing_runs(self.out_path), self.models
        )
        for model in self.models:
            self.runs[model] = [repeat_one[model]]

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "timestamp": datetime.now(UTC).isoformat(),
            "n_repeats": self.n_repeats,
            "engine": "agentic",
            "models": [m for m in MODELS if m in self.runs],
            "claim_ids": self.claim_ids,
            "runs": self.runs,
            "aggregate": aggregate_variance(self.runs),
        }

    def write(self) -> None:
        self.out_path.write_text(
            json.dumps(self._payload(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    async def record_repeat(self, model: str, repeat: dict[str, Any]) -> None:
        async with self._lock:
            self.runs[model].append(repeat)
            self.write()


async def run_model(
    model: str,
    claims: list[ClaimSpec],
    fresh_repeats: int,
    concurrency: int,
    timeout_s: int,
    acc: _Accumulator,
) -> None:
    """Run the fresh repeats for one model and record each durably."""
    cfg = _model_cfg(model)
    sem = asyncio.Semaphore(concurrency)
    for r in range(2, 2 + fresh_repeats):
        started = datetime.now(UTC).isoformat()
        print(
            f"  {model}: repeat {r}/{1 + fresh_repeats} -- {len(claims)} claims",
            flush=True,
        )
        results = await asyncio.gather(
            *(_run_cell(c, model, cfg, timeout_s, sem) for c in claims)
        )
        cells = {cid: cell for cid, cell in results}
        await acc.record_repeat(
            model,
            {"repeat": r, "source": "live", "started": started, "cells": cells},
        )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def _render(acc: _Accumulator) -> None:
    agg = aggregate_variance(acc.runs)
    print()
    print("=" * 72)
    print("CHAMBER VARIANCE")
    print("=" * 72)
    for model in acc.models:
        a = agg.get(model)
        if not a:
            continue
        fid, conf, lat = a["fidelity"], a["confidence"], a["latency_s"]
        stab = a["claim_stability"]
        print(f"\n  {model}")
        print(
            f"    fidelity      per_run={fid['per_run']}  mean={_fmt(fid['mean'])}  std={_fmt(fid['std'])}"
        )
        print(
            f"    confidence    per_run={[_fmt(x) for x in conf['per_run']]}  "
            f"mean={_fmt(conf['mean'])}  std={_fmt(conf['std'])}"
        )
        print(
            f"    latency_s     per_run={[_fmt(x) for x in lat['per_run']]}  "
            f"mean={_fmt(lat['mean'])}  std={_fmt(lat['std'])}"
        )
        print(
            f"    engine_errors per_run={a['engine_errors']['per_run']}  total={a['engine_errors']['total']}"
        )
        print(
            f"    claim_stability  stable={stab['stable']}  flipped={stab['flipped']}"
        )
        for fc in stab["flipped_claims"]:
            print(f"      flipped: {fc['id']}  {fc['pattern']}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> int:
    parser = argparse.ArgumentParser(description="Chamber variance experiment")
    parser.add_argument(
        "--models",
        default="",
        help="comma-separated model subset; default all three",
    )
    parser.add_argument(
        "--fresh-repeats",
        type=int,
        default=_DEFAULT_FRESH_REPEATS,
        help="repeats run live, on top of the imported repeat 1 (default 2)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=0,
        help="max concurrent agent runs; 0 (default) uses the per-model defaults (qwen3.6-27b sequential, others 4)",
    )
    parser.add_argument(
        "--max-claims",
        type=int,
        default=0,
        help="cap the claim set (0 = all 25); for cheap smoke runs",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=_DEFAULT_TIMEOUT_S,
        help="per-cell wall-clock ceiling in seconds (default 360)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output artifact path; the archive is read-only evidence and is "
        "never a valid target -- point a single-model re-run at a side file "
        "to run it concurrently with another variance process, then merge",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="import repeat 1 and write the artifact; no LLM calls",
    )
    args = parser.parse_args()

    out_path = args.out
    models = [m.strip() for m in args.models.split(",") if m.strip()] or list(MODELS)
    claims = load_claims()
    if args.max_claims > 0:
        claims = claims[: args.max_claims]
    claim_ids = [c.id for c in claims]

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    repeat_one = import_repeat_one(baseline, models, claim_ids)

    acc = _Accumulator(models, claim_ids, 1 + args.fresh_repeats, out_path=out_path)
    acc.seed(repeat_one)
    acc.write()  # durable artifact carries repeat 1 from the start

    if args.dry_run:
        print(
            f"dry run: imported repeat 1 for {len(models)} models x {len(claim_ids)} claims; no LLM calls"
        )
        _render(acc)
        print(f"wrote {short_path(out_path)}")
        return 0

    harness.setup_gateway_credentials()
    conc = {m: _resolve_concurrency(m, args.concurrency) for m in models}
    n_runs = len(models) * len(claim_ids) * args.fresh_repeats
    print(
        f"variance run: {len(models)} models x {len(claim_ids)} claims "
        f"x {args.fresh_repeats} fresh repeats = {n_runs} agentic runs",
        flush=True,
    )
    print(f"  concurrency per model: {conc}", flush=True)

    outcomes = await asyncio.gather(
        *(
            run_model(m, claims, args.fresh_repeats, conc[m], args.timeout, acc)
            for m in models
        ),
        return_exceptions=True,
    )
    for model, outcome in zip(models, outcomes, strict=True):
        if isinstance(outcome, Exception):
            print(
                f"  WARNING: model {model} failed: {type(outcome).__name__}: {outcome}",
                flush=True,
            )

    acc.write()
    _render(acc)
    print(f"wrote {short_path(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
