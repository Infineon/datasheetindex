"""The chamber-benchmark runner and its command-line entry point.

Merges the private repository's ``tests/eval/conftest_chamber.py`` (claim
loading, ``ChamberResultsCollector``) and ``tests/eval/test_chamber.py``
(the per-claim run body) into one module that is a *program*, not a test
suite: it makes live, billable calls against the model gateway, so it must
never be pytest-collectable. No function here is named ``test_*``, and
``benchmark/pyproject.toml`` only points ``testpaths`` at ``tests/`` -- see
``tests/test_harness_run.py`` for the check that pins both.

Run one model/engine combination over the bundled claim set:

    uv run chamber-run --model claudesonnet4.6 --engine agentic --out results/

Grading (fidelity scoring against the claim's expected substrings, and the
offline reproducibility protocol) still happens per claim, exactly as it did
under pytest -- there is no separate "grading pass" this file defers to. What
changed in the port is only pytest's own scaffolding: fixtures became plain
function calls, ``pytest.fail`` became an exception the caller can catch, and
the parametrized (claim, engine) matrix became one engine per invocation
(pick it with ``--engine``; run both by invoking this twice).
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import sys
import time
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import yaml

from chamberbench.claims import (
    ChamberMeasurement,
    ClaimResult,
    ClaimSpec,
    ReproducibilityVerdict,
    TraceStep,
)
from chamberbench.claimsio import archive_dir, load_claims, short_path
from chamberbench.grading import evaluate_case
from chamberbench.harness import (
    _VALID_REASONING_EFFORTS,
    model_config,
    rollup_cell_usage,
    setup_gateway_credentials,
)
from chamberbench.reproducibility import run_protocol, verdict

#: Top-level module names the ``harness`` extra installs, and the only ones
#: whose absence means "the extra is not installed". Spelled as modules rather
#: than distributions because that is what ``ImportError.name`` carries:
#: ``python-dotenv`` imports as ``dotenv``, and ``datasheetindex`` brings
#: ``pymupdf`` (which imports under both its own name and ``fitz``).
_HARNESS_EXTRA_MODULES = frozenset(
    {
        "anthropic",
        "openai",
        "httpx",
        "httpx2",
        "requests",
        "tenacity",
        "dotenv",
        "datasheetindex",
        "pymupdf",
        "fitz",
    }
)


def _is_missing_harness_extra(exc: ImportError) -> bool:
    """Is this ImportError "the extra is not installed", or a real breakage?

    ``ImportError.name`` is the module the import machinery was resolving:
    the missing third-party package for an uninstalled extra, but *this*
    package's own module for a renamed symbol or a circular import. Only the
    first is an install problem, and only the first may be answered with an
    install hint -- reporting the second as "not installed" sends a reader to
    reinstall something that is already there, with the traceback suppressed.
    """
    return (exc.name or "").split(".")[0] in _HARNESS_EXTRA_MODULES


# The engines and the tool surface are the only imports here that need the
# `harness` extra. A reader who followed the documented Tier 1 install --
# `uv pip install -e '.[test]'`, which deliberately installs no model client --
# otherwise meets `ModuleNotFoundError: No module named 'anthropic'` and a
# traceback through this file, which says nothing about what to do. Raising
# SystemExit from module scope prints the hint and stops, with no traceback,
# on both `chamber-run` and `python -m chamberbench.harness.run`.
#
# Narrowed to the extra's own modules: an ImportError raised from *inside*
# `anthropic_path` / `datasheet_tools` -- a renamed symbol, a circular import,
# a genuinely broken install -- is re-raised with its traceback intact rather
# than mislabelled as a missing dependency on a machine where the extra is
# installed. `_is_missing_harness_extra` is what draws that line.
try:
    from chamberbench.harness.anthropic_path import (
        extract_chamber_agentic,
        extract_chamber_baseline,
    )
    from chamberbench.harness.datasheet_tools import InspectDetail
except ImportError as exc:
    if not _is_missing_harness_extra(exc):
        raise
    print(
        f"chamber-run needs the harness extra, which is not installed ({exc}).\n"
        "\n"
        "    uv pip install -e '.[harness]'\n"
        "\n"
        "Tier 1 (`.[test]`) installs the grading surface and the archive only, "
        "with no model client:\n"
        "every number in the paper is re-derivable from `archive/` without "
        "this. See benchmark/README.md,\n"
        "'Running the harness'.",
        file=sys.stderr,
    )
    raise SystemExit(2) from None

# Per-claim wall-clock ceiling. Env-overridable because the qwen baseline
# renders every datasheet page and runs much longer than the other two arms
# on the biggest datasheet -- raise it (e.g. CHAMBER_TIMEOUT_S=900) for that
# run rather than for the other two.
CHAMBER_TIMEOUT_S = int(os.environ.get("CHAMBER_TIMEOUT_S", "360"))

# Sonnet is the canonical-baseline model. When a collector's `model` matches
# this, it also writes the un-suffixed canonical artifact paths so
# `chamberbench.quality_gates` / `chamberbench.classifier` / the cost-summary
# script keep working without a `--model` argument of their own.
CHAMBER_DEFAULT_MODEL = "claudesonnet4.6"


# ---------------------------------------------------------------------------
# Claim loading
# ---------------------------------------------------------------------------


def load_chamber_claims(path: Path | None = None) -> list[ClaimSpec]:
    """Parse a claim set into validated ``ClaimSpec`` instances.

    With no ``path``, loads the bundled claim set through
    ``chamberbench.claimsio.load_claims`` (which resolves
    ``CHAMBERBENCH_DATA_DIR`` for a reader pointing this at a re-derived
    claim set). With an explicit ``path`` (the CLI's ``--claims``), reads
    that file directly -- ``claimsio``'s own loader takes a filename relative
    to its data directory, not an arbitrary path.

    Raises ``ValueError`` if any claim id repeats: the collector keys cells
    by ``(claim_id, engine)`` and a duplicate would silently collapse the
    agreement matrix.
    """
    if path is None:
        claims = load_claims()
    else:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        claims = [ClaimSpec.model_validate(c) for c in raw["claims"]]
    ids = [c.id for c in claims]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        source = path if path is not None else "the bundled claim set"
        raise ValueError(f"duplicate claim ids in {source}: {duplicates}")
    return claims


def _claim_to_expected(claim: ClaimSpec) -> dict[str, Any]:
    """Build the `expected` dict consumed by `evaluate_case` from a ClaimSpec.

    Matches the shape produced by a production golden dataset: `found`,
    `confidence_min`, `value_contains`. Only `found` is required; missing
    fields fall back to permissive defaults.
    """
    return {
        "found": True,
        "confidence_min": claim.confidence_min,
        "value_contains": list(claim.value_contains),
    }


def _load_bearing_names(claim: ClaimSpec) -> list[str]:
    """Names of operating conditions the curator marks load-bearing.

    Used by the classifier to detect condition_omission via set-difference
    against the agent's own `extracted_conditions`.
    """
    return [c.name for c in claim.operating_conditions if c.load_bearing]


def _datasheetindex_version() -> str | None:
    """Resolved version of the `datasheetindex` distribution, or None.

    Stamped into every run's summary so a reader can tell which tool
    surface produced it -- the committed archive predates a tool-surface
    change in that library and records no version at all, which is exactly
    the ambiguity this guards against for every run from here on.
    """
    try:
        return importlib.metadata.version("datasheetindex")
    except importlib.metadata.PackageNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Per-run collector
# ---------------------------------------------------------------------------


class ChamberResultsCollector:
    """Accumulate fidelity x reproducibility results plus per-step traces.

    Produces on-disk artifacts under `results_dir`, written at the end of a
    run via `write_summary()` / `close()`:

      latest_chamber.{model}.json -- aggregate summary.
      latest_traces.{model}.jsonl -- one TraceStep per line, appended in
        agent-loop order across all claims.

    When `model == CHAMBER_DEFAULT_MODEL` the collector also writes the
    canonical un-suffixed paths (`latest_chamber.json`, `latest_traces.jsonl`)
    so `chamberbench.quality_gates` / `chamberbench.classifier` keep working
    unchanged against a freshly produced run.

    `results_dir` and `model` are both required, with no default -- in
    particular no default pointing at `chamberbench.claimsio.archive_dir()`,
    which is committed evidence and must never be a write target. The
    constructor also rejects `results_dir` resolving to the archive
    directory outright, as a second line of defense beyond "no default".
    """

    def __init__(self, results_dir: Path, model: str) -> None:
        results_dir = Path(results_dir)
        if results_dir.resolve() == archive_dir().resolve():
            raise ValueError(
                "results_dir must not be the archive directory; "
                "the archive is committed evidence and is read-only"
            )
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.datasheetindex_version = _datasheetindex_version()
        # results[(claim_id, engine)] -> per-cell record
        self.records: dict[tuple[str, str], dict[str, Any]] = {}
        # Stable per-process session id. The classifier filters to the most
        # recent session by reading the trailing `session_start` sentinel
        # of the JSONL.
        self.session_id = (
            f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        )
        self._trace_path = self.results_dir / f"latest_traces.{model}.jsonl"
        self._trace_handle: Any = None
        self._session_started = False

    # -- trace sink -------------------------------------------------------

    def trace_sink(self) -> Any:
        """Return a callable suitable for `extract_chamber_agentic`'s
        `trace_sink` arg. Each call appends one JSON line to the per-model
        trace file."""

        def _sink(step: TraceStep) -> None:
            self._open_trace()
            assert self._trace_handle is not None
            self._trace_handle.write(step.model_dump_json() + "\n")
            self._trace_handle.flush()

        return _sink

    def _open_trace(self) -> None:
        """Lazy open of the JSONL; truncates on first open, one
        `session_start` sentinel line per session so downstream tooling can
        partition traces by session id."""
        if self._trace_handle is not None:
            return
        self._trace_handle = self._trace_path.open("w", encoding="utf-8")
        if not self._session_started:
            sentinel = {
                "schema_version": 2,
                "kind": "session_start",
                "session_id": self.session_id,
                "timestamp": datetime.now(UTC).isoformat(),
            }
            self._trace_handle.write(json.dumps(sentinel) + "\n")
            self._trace_handle.flush()
            self._session_started = True

    # -- per-cell recording ----------------------------------------------

    def record(
        self,
        claim_id: str,
        engine: str,
        *,
        fidelity: dict[str, Any],
        repro: ReproducibilityVerdict | None,
        measurement: ChamberMeasurement | None,
        claim_result: ClaimResult | None,
        latency_s: float,
        n_steps: int = 0,
        n_tool_calls_by_tool: dict[str, int] | None = None,
        # Per-cell token usage rolled up from TraceStep events. Schema:
        # {"input_tokens", "output_tokens", "cache_read_tokens",
        #  "cache_creation_tokens"}.
        usage: dict[str, int] | None = None,
        engine_error: str = "",
        repro_error: str = "",
        load_bearing_condition_names: list[str] | None = None,
        # Back-compat alias: a single `error` string routes to engine_error
        # when no explicit split is given.
        error: str = "",
    ) -> None:
        if error and not engine_error and not repro_error:
            engine_error = error
        self.records[(claim_id, engine)] = {
            "claim_id": claim_id,
            "engine": engine,
            "session_id": self.session_id,
            "fidelity": fidelity,
            "reproducibility": (repro.model_dump() if repro is not None else None),
            "measurement": (
                measurement.model_dump() if measurement is not None else None
            ),
            "claim_result": (
                claim_result.model_dump() if claim_result is not None else None
            ),
            "latency_s": latency_s,
            "n_steps": n_steps,
            "n_tool_calls_by_tool": n_tool_calls_by_tool or {},
            "usage": dict(usage)
            if usage
            else {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_creation_tokens": 0,
            },
            "engine_error": engine_error,
            "repro_error": repro_error,
            "load_bearing_condition_names": list(load_bearing_condition_names or []),
        }

    # -- session-finalisation --------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "model": self.model,
            "datasheetindex_version": self.datasheetindex_version,
            "results_dir": short_path(self.results_dir),
            "results": {
                f"{cid}|{eng}": rec for (cid, eng), rec in self.records.items()
            },
        }

    def write_summary(self) -> Path:
        """Write per-model summary; mirror to the canonical path for the
        default model.

        Returns the per-model path. The canonical mirror is written as a
        side effect when `self.model == CHAMBER_DEFAULT_MODEL` so existing
        gates / classifier invocations work unchanged.
        """
        payload = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        out = self.results_dir / f"latest_chamber.{self.model}.json"
        out.write_text(payload, encoding="utf-8")
        if self.model == CHAMBER_DEFAULT_MODEL:
            (self.results_dir / "latest_chamber.json").write_text(
                payload, encoding="utf-8"
            )
        return out

    def close(self) -> None:
        if self._trace_handle is not None:
            try:
                self._trace_handle.close()
            finally:
                self._trace_handle = None
        if self.model == CHAMBER_DEFAULT_MODEL and self._trace_path.exists():
            canonical = self.results_dir / "latest_traces.jsonl"
            canonical.write_bytes(self._trace_path.read_bytes())


# ---------------------------------------------------------------------------
# Reasoning-effort resolution
# ---------------------------------------------------------------------------


def _resolve_reasoning_effort(model: str) -> str:
    """Responses-API reasoning effort for `model`.

    Defaults from `model_config(model)` when the alias has one, else
    "medium"; `CHAMBER_REASONING_EFFORT` overrides either. Validated against
    `_VALID_REASONING_EFFORTS` (none/minimal/low/medium/high) so a typo'd
    override fails loudly here rather than as an opaque 400 from the
    gateway. Ignored by the Anthropic/vLLM engine path; honoured by the
    OpenAI Responses-API path and Claude's `effort` knob.
    """
    default = str(model_config(model).get("reasoning_effort", "medium"))
    effort = os.environ.get("CHAMBER_REASONING_EFFORT", default)
    if effort not in _VALID_REASONING_EFFORTS:
        raise ValueError(
            f"CHAMBER_REASONING_EFFORT={effort!r} is not a valid Responses-API "
            f"effort level; expected one of {sorted(_VALID_REASONING_EFFORTS)}"
        )
    return effort


# ---------------------------------------------------------------------------
# Per-claim run
# ---------------------------------------------------------------------------


async def run_claim(
    claim: ClaimSpec,
    *,
    model: str,
    engine: str,
    collector: ChamberResultsCollector,
) -> ClaimResult:
    """Run one (claim, engine) cell: extract, score fidelity, replay the
    offline reproducibility protocol, and record the cell into `collector`.

    Ported from the private repository's ``test_chamber_claim`` (one
    parametrized pytest case per (claim, engine)): the extraction call, the
    fidelity/reproducibility scoring, and the recorded cell shape are
    unchanged. What is gone is pytest itself -- `pytest.fail` on an engine
    error becomes a plain `RuntimeError` (still recorded first, so a
    caller that keeps iterating other claims still gets a full agreement
    matrix), and the two closing sanity checks are plain `assert`s rather
    than test assertions.

    `max_turns` / `inspect_page_detail` come from `model_config(model)` --
    never hardcoded here -- so an unrecognised model alias falls through to
    the Sonnet-shaped ceiling instead of a config-shaped false failure, and
    a small-context alias like the qwen arm gets its low vision-detail tier
    instead of overflowing its window.
    """
    cfg = model_config(model)
    max_turns = int(cast(int, cfg["max_turns"]))
    inspect_page_detail = cast(InspectDetail, cfg["inspect_page_detail"])
    reasoning_effort = _resolve_reasoning_effort(model)

    in_memory_steps: list[TraceStep] = []
    file_sink = collector.trace_sink()

    def _dual_sink(step: TraceStep) -> None:
        in_memory_steps.append(step)
        file_sink(step)

    t0 = time.monotonic()
    claim_result: ClaimResult | None = None
    engine_error = ""
    try:
        if engine == "agentic":
            claim_result = await asyncio.wait_for(
                extract_chamber_agentic(
                    claim,
                    model=model,
                    max_turns=max_turns,
                    trace_sink=_dual_sink,
                    inspect_page_detail=inspect_page_detail,
                    reasoning_effort=reasoning_effort,
                ),
                timeout=CHAMBER_TIMEOUT_S,
            )
        else:
            claim_result = await asyncio.wait_for(
                extract_chamber_baseline(
                    claim,
                    model=model,
                    trace_sink=_dual_sink,
                    reasoning_effort=reasoning_effort,
                ),
                timeout=CHAMBER_TIMEOUT_S,
            )
    except TimeoutError as exc:
        engine_error = f"timeout after {CHAMBER_TIMEOUT_S}s: {exc}"
    except Exception as exc:  # noqa: BLE001
        engine_error = f"{type(exc).__name__}: {exc}"
    elapsed = time.monotonic() - t0

    by_tool_counter: Counter[str] = Counter(
        s.tool_name for s in in_memory_steps if s.kind == "tool_call" and s.tool_name
    )
    n_steps = len(in_memory_steps)
    usage_dict = rollup_cell_usage(in_memory_steps)

    if claim_result is None:
        collector.record(
            claim_id=claim.id,
            engine=engine,
            fidelity={
                "found_expected": True,
                "found_actual": False,
                "found_correct": False,
                "value_pass": False,
                "confidence": 0.0,
                "failure_reason": engine_error or "no result",
                "overall_pass": False,
                "engine_error": True,
            },
            repro=None,
            measurement=None,
            claim_result=None,
            latency_s=elapsed,
            n_steps=n_steps,
            n_tool_calls_by_tool=dict(by_tool_counter),
            usage=usage_dict,
            error=engine_error,
            load_bearing_condition_names=_load_bearing_names(claim),
        )
        raise RuntimeError(f"engine {engine!r} failed for {claim.id!r}: {engine_error}")

    # 1. Fidelity scoring against the curator's expected substrings.
    fidelity = evaluate_case(claim_result.extracted, _claim_to_expected(claim))

    # 2. Reproducibility: chamber-side, no LLM. Engine-side and protocol-side
    #    errors land in different fields so quality_gates can branch.
    repro = None
    measurement = None
    repro_error = ""
    try:
        measurement = run_protocol(claim)
        repro = verdict(claim, measurement)
    except Exception as exc:  # noqa: BLE001
        repro_error = f"protocol error: {type(exc).__name__}: {exc}"

    collector.record(
        claim_id=claim.id,
        engine=engine,
        fidelity=fidelity,
        repro=repro,
        measurement=measurement,
        claim_result=claim_result,
        latency_s=elapsed,
        n_steps=n_steps,
        n_tool_calls_by_tool=dict(by_tool_counter),
        usage=usage_dict,
        engine_error="",
        repro_error=repro_error,
        load_bearing_condition_names=_load_bearing_names(claim),
    )

    # Sanity invariants (not pytest assertions -- plain guards). A claim
    # result claiming found=True with no extracted payload, or a confident
    # found=False, is malformed enough to be worth failing loudly on rather
    # than silently recording.
    assert claim_result.extracted is not None, (
        "agent returned found=True with extracted=None"
    )
    if claim_result.found:
        assert claim_result.extracted.values, (
            f"agent returned found=True for {claim.id!r} but extracted.values is empty"
        )
    assert claim_result.found is True or claim_result.confidence < 0.5, (
        f"agent reported found=False with high confidence={claim_result.confidence:.2f} "
        f"for {claim.id!r}; investigate before treating this as a real not-in-datasheet signal"
    )

    return claim_result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chamber-run",
        description="Run the chamber benchmark against a LiteLLM gateway.",
    )
    parser.add_argument(
        "--model", default="claudesonnet4.6", help="gateway model alias"
    )
    parser.add_argument(
        "--engine",
        default="agentic",
        choices=("agentic", "baseline"),
        help="agentic tool loop, or the single-pass baseline",
    )
    parser.add_argument(
        "--claims", type=Path, default=None, help="claims YAML (default: bundled)"
    )
    parser.add_argument("--out", type=Path, required=True, help="results directory")
    parser.add_argument(
        "--claim-id", action="append", default=None, help="run only these claims"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_gateway_credentials()
    claims = load_chamber_claims(args.claims)
    if args.claim_id:
        wanted = set(args.claim_id)
        claims = [c for c in claims if c.id in wanted]
    collector = ChamberResultsCollector(results_dir=args.out, model=args.model)
    try:
        for claim in claims:
            try:
                asyncio.run(
                    run_claim(
                        claim, model=args.model, engine=args.engine, collector=collector
                    )
                )
            except Exception as exc:  # noqa: BLE001
                # A single claim's failure is already recorded as an
                # engine_error cell inside run_claim; keep going so the rest
                # of the matrix still gets written.
                print(f"claim {claim.id} failed: {exc}", file=sys.stderr)
        summary = collector.write_summary()
    finally:
        collector.close()
    print("wrote", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
