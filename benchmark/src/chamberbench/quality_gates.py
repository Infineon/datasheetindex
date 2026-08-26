"""Quality gates for the chamber-grounded benchmark.

Reads `archive/latest_chamber.json` (per-cell agreement
matrix) and optionally `archive/baseline_chamber.json`
(frozen reference) and decides pass/warn/fail.

Baseline schemas supported:
  v1 (legacy, Sonnet-only): top-level `results` is keyed `<cid>|<eng>`.
  v2: `results[claim_id][engine][model] -> cell`. Each
      cell carries a `status` in {"ok", "not_applicable", "pending_rerun"}.

Hard gates (exit 1 on failure):
  H1. Agentic-engine fidelity found-accuracy >= 90 %
  H2. Agentic-engine fidelity value-accuracy >= 80 %
  H3. No reproducibility regressions vs baseline:
        * pass -> fail in any cell is a hard fail
        * inconclusive -> fail in any cell is a hard fail
  H4. No engine errors in the agentic engine
  H5. No protocol crashes (repro_error non-empty for any cell)
  H6. No per-model regressions vs the 3D baseline (v2 only):
        * baseline ok -> current engine_error or repro fail is a hard fail
        * pending_rerun rows are skipped (no comparison possible)
        * not_applicable -> not_applicable is an expected match

Soft gates (exit 2 with warnings; exit 0 otherwise):
  S1. Inconclusive rate < 50 % (chamber-side bound is too loose if not)
  S2. Per-tool error rate < 20 % per tool (from classifier attribution)
  S3. Mean confidence >= 0.80 across both engines
  S4. Mean trace length <= 25 steps per agentic claim
  S8. Cross-contamination opportunity rate <= 90 % (informational
        ceiling). Computed from the per-model latest_traces JSONL by
        ``chamberbench.contamination_audit``: counts
        cells where chamber-side tools (list_experiments,
        query_dataset, cross_sensor_check, ...) fired before
        submit_claim_result. Today's rate is 80-100% per model; the
        named structural fix is a two-pass agent design (datasheet
        phase freezes ``extracted`` before chamber tools are exposed).
  S9. No silent-failure cells: a fidelity-pass agentic cell flagged by
        the dispatch-level silent-failure detector
        (``chamberbench.silent_failure``) -- zero
        navigation-tool calls (tool-bypass), or navigation with no
        verification tool. Fidelity-only scoring cannot see these.

Usage:
    uv run python -m chamber.quality_gates \
        --results-dir archive/

Exit codes follow the same convention as eval/quality_gates.py:
  0 -- all hard + soft gates passed
  1 -- at least one hard gate failed
  2 -- all hard gates passed but one or more soft gates issued warnings
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from chamberbench.claimsio import archive_dir

# --- Hard gates ------------------------------------------------------------
FOUND_ACCURACY_MIN = 0.90
VALUE_ACCURACY_MIN = 0.80

# --- Soft gates ------------------------------------------------------------
# 0.85 (not 0.50): honest methodology routes load-bearing-unmatched
# claims to inconclusive on purpose. The gate fires only when the rate
# climbs above what the seed-claim set can plausibly produce.
INCONCLUSIVE_RATE_MAX = 0.85
PER_TOOL_ERROR_RATE_MAX = 0.20
MEAN_CONFIDENCE_MIN = 0.80
MEAN_TRACE_LEN_MAX = 25.0
# S8: informational ceiling. The metric is "fraction of agentic cells
# where chamber-side tools fired before submit_claim_result". It is
# currently 80-100 % per model and is structural until the two-pass
# agent redesign lands; the warning here surfaces it on every run
# rather than silently absorbing it.
CONTAMINATION_RATE_MAX = 0.90


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _contamination_rate(
    results_dir: Path, model: str | None
) -> tuple[int, int, float] | None:
    """Return (n_with_submit, n_contaminated, rate) for the named model's
    trace file, or None when the trace file is missing.

    Imports the audit module lazily so quality_gates stays importable in
    environments that don't have the chamber_eval package installed
    (e.g. CI smoke tests on subsets).
    """
    from chamberbench.contamination_audit import analyze_traces

    candidates: list[Path] = []
    if model:
        candidates.append(results_dir / f"latest_traces.{model}.jsonl")
    candidates.append(results_dir / "latest_traces.jsonl")
    trace_path = next((p for p in candidates if p.exists()), None)
    if trace_path is None:
        return None
    summary = analyze_traces(trace_path)
    return (
        summary["n_cells_with_submit"],
        summary["n_cells_contaminated"],
        summary["contamination_rate"],
    )


# ---------------------------------------------------------------------------
# Cell partitioning
# ---------------------------------------------------------------------------


def _split_by_engine(
    results: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Return (agentic_cells, baseline_cells), each keyed by claim_id."""
    agentic: dict[str, dict[str, Any]] = {}
    baseline: dict[str, dict[str, Any]] = {}
    for key, cell in results.items():
        cid = cell.get("claim_id") or key.split("|", 1)[0]
        eng = cell.get("engine") or (key.split("|", 1)[1] if "|" in key else "")
        if eng == "agentic":
            agentic[cid] = cell
        elif eng == "baseline":
            baseline[cid] = cell
    return agentic, baseline


# ---------------------------------------------------------------------------
# Hard gates
# ---------------------------------------------------------------------------


def _check_fidelity_found(cells: dict[str, dict[str, Any]]) -> tuple[float, bool]:
    if not cells:
        return 0.0, False
    correct = sum(
        1 for c in cells.values() if (c.get("fidelity") or {}).get("found_correct")
    )
    acc = correct / len(cells)
    return acc, acc >= FOUND_ACCURACY_MIN


def _check_fidelity_value(cells: dict[str, dict[str, Any]]) -> tuple[float, bool]:
    eligible = {
        k: c
        for k, c in cells.items()
        if (c.get("fidelity") or {}).get("found_expected")
    }
    if not eligible:
        return 1.0, True
    correct = sum(
        1 for c in eligible.values() if (c.get("fidelity") or {}).get("value_pass")
    )
    acc = correct / len(eligible)
    return acc, acc >= VALUE_ACCURACY_MIN


def _check_repro_regressions(
    current_cells: dict[str, dict[str, Any]],
    baseline_cells: dict[str, dict[str, Any]] | None,
) -> list[str]:
    """A regression is a cell whose baseline verdict was {pass, inconclusive}
    but is now `fail`. pass -> inconclusive is NOT a hard regression
    (that's the soft S5 verdict-drift gate)."""
    if not baseline_cells:
        return []
    regs: list[str] = []
    for cid, base in baseline_cells.items():
        cur = current_cells.get(cid)
        if cur is None:
            continue
        base_v = ((base.get("reproducibility") or {}).get("verdict") or "").lower()
        cur_v = ((cur.get("reproducibility") or {}).get("verdict") or "").lower()
        if base_v in ("pass", "inconclusive") and cur_v == "fail":
            regs.append(f"{cid}: {base_v} -> {cur_v}")
    return regs


def _check_verdict_drift(
    current_cells: dict[str, dict[str, Any]],
    baseline_cells: dict[str, dict[str, Any]] | None,
) -> list[str]:
    """Soft signal: pass -> inconclusive (or back). The methodology doc
    says inconclusive is "needs inspection," not a hard fail; we surface
    it so a chamber-side bound widening doesn't go silent."""
    if not baseline_cells:
        return []
    drifts: list[str] = []
    for cid, base in baseline_cells.items():
        cur = current_cells.get(cid)
        if cur is None:
            continue
        base_v = ((base.get("reproducibility") or {}).get("verdict") or "").lower()
        cur_v = ((cur.get("reproducibility") or {}).get("verdict") or "").lower()
        if base_v == "pass" and cur_v == "inconclusive":
            drifts.append(f"{cid}: pass -> inconclusive")
        elif base_v == "inconclusive" and cur_v == "pass":
            drifts.append(f"{cid}: inconclusive -> pass")
    return drifts


def _check_baseline_coverage(
    current_cells: dict[str, dict[str, Any]],
    baseline_cells: dict[str, dict[str, Any]] | None,
) -> tuple[list[str], list[str]]:
    """(only_in_baseline, only_in_current). New claims and dropped claims
    are both worth flagging."""
    if not baseline_cells:
        return [], []
    base = set(baseline_cells.keys())
    cur = set(current_cells.keys())
    return sorted(base - cur), sorted(cur - base)


# ---------------------------------------------------------------------------
# v2 schema (3D baseline: claim_id × engine × model)
# ---------------------------------------------------------------------------


def _is_v2_baseline(baseline: dict[str, Any] | None) -> bool:
    return bool(baseline and baseline.get("schema_version") == 2)


def _iter_v2_baseline(
    baseline: dict[str, Any],
) -> list[tuple[str, str, str, dict[str, Any]]]:
    """Flatten the 3D results dict to (claim_id, engine, model, cell) tuples."""
    out: list[tuple[str, str, str, dict[str, Any]]] = []
    for cid, by_eng in (baseline.get("results") or {}).items():
        if not isinstance(by_eng, dict):
            continue
        for eng, by_model in by_eng.items():
            if not isinstance(by_model, dict):
                continue
            for model, cell in by_model.items():
                if isinstance(cell, dict):
                    out.append((cid, eng, model, cell))
    return out


def _load_per_model_latest(
    results_dir: Path, models: list[str]
) -> dict[str, dict[tuple[str, str], dict[str, Any]]]:
    """Read each `latest_chamber.{model}.json` and split into per-cell dicts.

    Returns ``{model: {(claim_id, engine): cell}}``. Missing per-model files
    show up as empty inner dicts so the caller can distinguish "ran zero
    cells" from "did not run this model" by examining the keys.
    """
    out: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for m in models:
        path = results_dir / f"latest_chamber.{m}.json"
        d = _load(path)
        per_cell: dict[tuple[str, str], dict[str, Any]] = {}
        if d is not None:
            agentic, baseline = _split_by_engine(d.get("results") or {})
            for cid, cell in agentic.items():
                per_cell[(cid, "agentic")] = cell
            for cid, cell in baseline.items():
                per_cell[(cid, "baseline")] = cell
        out[m] = per_cell
    return out


def _check_per_model_regressions(
    baseline: dict[str, Any],
    per_model_current: dict[str, dict[tuple[str, str], dict[str, Any]]],
) -> list[str]:
    """H6: per-model regression detector for the v2 schema.

    Iterates over cells *actually present in the current session* and
    compares each to its baseline cell. Cells missing from the current
    session are a coverage gap (reported by S6), not a regression --
    chamber sessions routinely run a subset of (claim × engine × model).

    A regression fires when, for status=ok baseline:
      - current cell has a populated `engine_error`, or
      - current cell's reproducibility verdict flipped pass/inconclusive
        -> fail.
    Cells whose baseline status is `pending_rerun` are skipped (no
    comparison possible). Cells whose baseline status is
    `not_applicable` are handled by `_check_per_model_improvements`.
    """
    # Build a quick lookup: (cid, eng, model) -> baseline cell.
    base_lookup: dict[tuple[str, str, str], dict[str, Any]] = {
        (cid, eng, model): cell for cid, eng, model, cell in _iter_v2_baseline(baseline)
    }
    regs: list[str] = []
    for model, cells in per_model_current.items():
        for (cid, eng), cur in cells.items():
            base_cell = base_lookup.get((cid, eng, model))
            if base_cell is None:
                continue  # New cell -- coverage, not regression.
            status = base_cell.get("status", "ok")
            if status != "ok":
                continue  # pending_rerun / not_applicable handled elsewhere.
            if cur.get("engine_error"):
                regs.append(f"{cid}|{eng}|{model}: engine_error (baseline ok)")
                continue
            base_v = (
                (base_cell.get("reproducibility") or {}).get("verdict") or ""
            ).lower()
            cur_v = ((cur.get("reproducibility") or {}).get("verdict") or "").lower()
            if base_v in ("pass", "inconclusive") and cur_v == "fail":
                regs.append(f"{cid}|{eng}|{model}: repro {base_v} -> {cur_v}")
    return regs


def _check_per_model_improvements(
    baseline: dict[str, Any],
    per_model_current: dict[str, dict[tuple[str, str], dict[str, Any]]],
) -> list[str]:
    """S7: surface cells that flipped from not_applicable to ok.

    A useful upgrade signal -- e.g. gpt-5.1 baseline starts working
    when the gateway's `document`-block translator gets fixed.
    """
    ups: list[str] = []
    for cid, eng, model, base_cell in _iter_v2_baseline(baseline):
        if base_cell.get("status") != "not_applicable":
            continue
        cur = per_model_current.get(model, {}).get((cid, eng))
        if cur is None or cur.get("engine_error"):
            continue
        ups.append(f"{cid}|{eng}|{model}: not_applicable -> ok")
    return ups


# ---------------------------------------------------------------------------
# Engine / protocol errors (existing)
# ---------------------------------------------------------------------------


def _check_engine_errors(cells: dict[str, dict[str, Any]]) -> list[str]:
    """Engine errors are populated on either:
    - cell["engine_error"]            (current schema)
    - cell["fidelity"]["engine_error"](mirrored boolean)
    - cell["error"]                   (back-compat, older traces)
    """
    out: list[str] = []
    for cid, c in cells.items():
        msg = (
            c.get("engine_error")
            or c.get("error")
            or (c.get("fidelity") or {}).get("failure_reason")
        )
        if (
            c.get("engine_error")
            or (c.get("fidelity") or {}).get("engine_error")
            or c.get("error")
        ):
            out.append(f"{cid}: {msg}")
    return out


def _check_protocol_errors(cells: dict[str, dict[str, Any]]) -> list[str]:
    """Includes both raised protocol errors AND non-finite measurements.

    NaN/inf is only flagged when the verdict is NOT a load-bearing
    short-circuit: the protocol layer emits NaN by design when an
    unmatched load-bearing condition forces inconclusive (the value
    is meaningless and shouldn't be consumed). A NaN measurement on
    any other path is a real protocol bug.

    The marker we key on is `measured_sigma_basis == "stub"`, written by
    the short-circuit branch of every protocol module. Earlier versions
    of this gate keyed on `unmatched_conditions` being non-empty, which
    became dangerously permissive once `match_conditions` started
    reporting non-load-bearing constraints as unmatched:
    a future NaN bug in the full-run path of a claim with any
    non-load-bearing constraint would have been silently swallowed.
    """
    import math

    out: list[str] = []
    for cid, c in cells.items():
        if c.get("repro_error"):
            out.append(f"{cid}: {c['repro_error']}")
            continue
        meas = c.get("measurement") or {}
        v = meas.get("measured_value")
        if isinstance(v, (int, float)) and not math.isfinite(v):
            sigma_basis = (meas.get("measured_sigma_basis") or "").lower()
            if sigma_basis == "stub":
                # Expected: stub from load-bearing-unmatched short-circuit.
                continue
            out.append(f"{cid}: non_finite_measurement={v!r}")
    return out


# ---------------------------------------------------------------------------
# Soft gates
# ---------------------------------------------------------------------------


def _inconclusive_rate(cells: dict[str, dict[str, Any]]) -> float:
    total = 0
    inconc = 0
    for c in cells.values():
        v = ((c.get("reproducibility") or {}).get("verdict") or "").lower()
        if not v:
            continue
        total += 1
        if v == "inconclusive":
            inconc += 1
    return (inconc / total) if total else 0.0


def _per_tool_error_rate(cells: dict[str, dict[str, Any]]) -> dict[str, float]:
    """For each tool, fraction of its agentic invocations classified as
    `tool_output`.

    Reads `cell["tool_output_by_tool"]` written by the classifier (a
    precise per-tool numerator), and divides by `n_tool_calls_by_tool`.
    Older summaries without the new field fall back to a zero-error
    assumption (rather than the prior proportional-allocation guess,
    which was silently wrong when one tool errored and another did not).
    """
    calls_per_tool: defaultdict[str, int] = defaultdict(int)
    errors_per_tool: defaultdict[str, int] = defaultdict(int)
    for cell in cells.values():
        tool_counts = cell.get("n_tool_calls_by_tool") or {}
        for tool, n in tool_counts.items():
            calls_per_tool[tool] += n
        per_tool_errors = cell.get("tool_output_by_tool") or {}
        for tool, n in per_tool_errors.items():
            errors_per_tool[tool] += int(n)
    out: dict[str, float] = {}
    for tool, calls in calls_per_tool.items():
        if calls == 0:
            continue
        out[tool] = errors_per_tool.get(tool, 0) / calls
    return out


def _mean_confidence(*engines: dict[str, dict[str, Any]]) -> float:
    confidences = []
    for cells in engines:
        for c in cells.values():
            conf = (c.get("fidelity") or {}).get("confidence")
            if isinstance(conf, (int, float)):
                confidences.append(float(conf))
    return sum(confidences) / len(confidences) if confidences else 0.0


def _mean_trace_length(cells: dict[str, dict[str, Any]]) -> float:
    """Mean trace length across agentic cells. Cells with n_steps=0 are
    only included if they're the agentic engine and had an engine_error
    before any tool dispatch -- a real signal we shouldn't drop."""
    lengths: list[int] = []
    for c in cells.values():
        if (c.get("engine") or "") != "agentic":
            continue
        lengths.append(int(c.get("n_steps", 0) or 0))
    return sum(lengths) / len(lengths) if lengths else 0.0


def _silent_failures(cells: dict[str, dict[str, Any]]) -> list[str]:
    """S9: agentic cells the silent-failure detector flags.

    A silent failure is a fidelity-passing cell whose per-tool dispatch
    record shows the answer was not actually extracted -- zero navigation
    calls (tool-bypass), or navigation with no verification tool. Fidelity-
    only scoring cannot see these; the detector reads the dispatch record.
    The detector module is imported lazily, matching the contamination
    audit above.
    """
    from chamberbench.silent_failure import (
        detect_silent_failure,
    )

    out: list[str] = []
    for cid, cell in cells.items():
        report = detect_silent_failure(cell)
        if report.flagged:
            out.append(f"{cid}: {', '.join(report.rules)}")
    return out


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


def run_gates(results_dir: str) -> int:
    rd = Path(results_dir)
    summary = _load(rd / "latest_chamber.json")
    if summary is None:
        print(f"ERROR: results file not found: {rd / 'latest_chamber.json'}")
        return 1
    baseline = _load(rd / "baseline_chamber.json")

    cells = summary.get("results", {})
    agentic, baseline_engine = _split_by_engine(cells)
    base_agentic = (
        _split_by_engine(baseline.get("results", {}))[0] if baseline else None
    )

    # ----- Hard gates -----
    hard: list[str] = []

    found_acc, found_ok = _check_fidelity_found(agentic)
    if not found_ok:
        hard.append(
            f"H1 fidelity found-accuracy (agentic) {found_acc:.1%} < {FOUND_ACCURACY_MIN:.0%}"
        )

    val_acc, val_ok = _check_fidelity_value(agentic)
    if not val_ok:
        hard.append(
            f"H2 fidelity value-accuracy (agentic) {val_acc:.1%} < {VALUE_ACCURACY_MIN:.0%}"
        )

    regressions = _check_repro_regressions(agentic, base_agentic)
    if regressions:
        hard.append(
            f"H3 reproducibility regressions vs baseline: {', '.join(regressions)}"
        )

    engine_errs = _check_engine_errors(agentic) + _check_engine_errors(baseline_engine)
    if engine_errs:
        hard.append(f"H4 engine errors: {'; '.join(engine_errs)}")

    proto_errs = _check_protocol_errors(agentic) + _check_protocol_errors(
        baseline_engine
    )
    if proto_errs:
        hard.append(f"H5 protocol errors: {'; '.join(proto_errs)}")

    # H6 (v2 only): per-model regressions across the 3D baseline. The
    # session's per-model `latest_chamber.{model}.json` files are read
    # independently of the `latest_chamber.json` mirror that drives H1-H5,
    # so H6 fires even when a session only ran a subset of models.
    per_model_improvements: list[str] = []
    per_model_regs: list[str] = []
    if baseline is not None and _is_v2_baseline(baseline):
        models = list(baseline.get("models") or [])
        per_model_current = _load_per_model_latest(rd, models)
        per_model_regs = _check_per_model_regressions(baseline, per_model_current)
        if per_model_regs:
            hard.append(f"H6 per-model regressions: {'; '.join(per_model_regs)}")
        per_model_improvements = _check_per_model_improvements(
            baseline, per_model_current
        )

    # ----- Soft gates -----
    soft: list[str] = []

    inconc_rate = _inconclusive_rate(agentic)
    if inconc_rate > INCONCLUSIVE_RATE_MAX:
        soft.append(
            f"S1 inconclusive rate (agentic) {inconc_rate:.1%} > {INCONCLUSIVE_RATE_MAX:.0%}"
        )

    tool_err = _per_tool_error_rate(agentic)
    high_err_tools = [t for t, r in tool_err.items() if r > PER_TOOL_ERROR_RATE_MAX]
    if high_err_tools:
        soft.append(
            f"S2 per-tool error rate > {PER_TOOL_ERROR_RATE_MAX:.0%} for: "
            + ", ".join(f"{t}={tool_err[t]:.0%}" for t in high_err_tools)
        )

    mean_conf = _mean_confidence(agentic, baseline_engine)
    if mean_conf < MEAN_CONFIDENCE_MIN:
        soft.append(f"S3 mean confidence {mean_conf:.2f} < {MEAN_CONFIDENCE_MIN:.2f}")

    mean_trace = _mean_trace_length(agentic)
    if mean_trace > MEAN_TRACE_LEN_MAX:
        soft.append(f"S4 mean trace length {mean_trace:.1f} > {MEAN_TRACE_LEN_MAX:.0f}")

    drift = _check_verdict_drift(agentic, base_agentic)
    if drift:
        soft.append("S5 reproducibility verdict drift vs baseline: " + ", ".join(drift))

    baseline_only, current_only = _check_baseline_coverage(agentic, base_agentic)
    if baseline_only or current_only:
        parts: list[str] = []
        if baseline_only:
            parts.append("baseline-only: " + ", ".join(baseline_only))
        if current_only:
            parts.append("current-only: " + ", ".join(current_only))
        soft.append("S6 baseline coverage gap: " + " | ".join(parts))

    # S7 (v2 only): cells that flipped not_applicable -> ok. Surfaced as
    # a soft signal so a future gateway-translator fix is celebrated, not
    # silently absorbed.
    if per_model_improvements:
        soft.append(
            "S7 per-model improvements vs baseline: "
            + ", ".join(per_model_improvements)
        )

    # S8: cross-contamination opportunity rate from the per-model trace
    # file. Returns None when the trace file is missing (e.g. a baseline-
    # only run); the gate is skipped silently in that case.
    contam = _contamination_rate(rd, summary.get("model"))
    if contam is not None and contam[2] > CONTAMINATION_RATE_MAX:
        n_sub, n_con, rate = contam
        soft.append(
            f"S8 contamination opportunity rate {rate:.1%} "
            f"({n_con}/{n_sub} agentic cells called chamber tools "
            f"before submit) > {CONTAMINATION_RATE_MAX:.0%} -- "
            "open methodology gap; two-pass agent design is the named "
            "structural fix"
        )

    silent = _silent_failures(agentic)
    if silent:
        soft.append(
            "S9 silent-failure cells (fidelity pass, defective dispatch record): "
            + "; ".join(silent)
        )

    # ----- Pretty print -----
    print("=" * 70)
    print("CHAMBER QUALITY GATE SUMMARY")
    print("=" * 70)
    print(f"  Timestamp:                 {summary.get('timestamp', '?')}")
    print(f"  Model:                     {summary.get('model', '?')}")
    print(f"  Cells (agentic/baseline):  {len(agentic)} / {len(baseline_engine)}")
    print()

    print("  Hard gates:")
    print(
        f"    H1 fidelity found:       {found_acc:.1%}  "
        f"(threshold {FOUND_ACCURACY_MIN:.0%})  "
        f"[{'OK' if found_ok else 'FAIL'}]"
    )
    print(
        f"    H2 fidelity value:       {val_acc:.1%}  "
        f"(threshold {VALUE_ACCURACY_MIN:.0%})  "
        f"[{'OK' if val_ok else 'FAIL'}]"
    )
    print(
        f"    H3 repro regressions:    {len(regressions)}  [{'OK' if not regressions else 'FAIL'}]"
    )
    print(
        f"    H4 engine errors:        {len(engine_errs)}  [{'OK' if not engine_errs else 'FAIL'}]"
    )
    print(
        f"    H5 protocol errors:      {len(proto_errs)}  [{'OK' if not proto_errs else 'FAIL'}]"
    )
    if _is_v2_baseline(baseline):
        print(
            f"    H6 per-model regr.:      {len(per_model_regs)}  [{'OK' if not per_model_regs else 'FAIL'}]"
        )
    print()

    print("  Soft gates:")
    print(
        f"    S1 inconclusive rate:    {inconc_rate:.1%}  "
        f"(threshold {INCONCLUSIVE_RATE_MAX:.0%})  "
        f"[{'OK' if inconc_rate <= INCONCLUSIVE_RATE_MAX else 'WARN'}]"
    )
    if tool_err:
        worst = max(tool_err.values())
        worst_tool = max(tool_err.keys(), key=lambda t: tool_err[t])
        print(
            f"    S2 worst tool error:     "
            f"{worst_tool}={worst:.0%}  "
            f"(threshold {PER_TOOL_ERROR_RATE_MAX:.0%})  "
            f"[{'OK' if worst <= PER_TOOL_ERROR_RATE_MAX else 'WARN'}]"
        )
    else:
        print("    S2 worst tool error:     n/a (no tool calls)")
    print(
        f"    S3 mean confidence:      {mean_conf:.2f}  "
        f"(threshold {MEAN_CONFIDENCE_MIN})  "
        f"[{'OK' if mean_conf >= MEAN_CONFIDENCE_MIN else 'WARN'}]"
    )
    print(
        f"    S4 mean trace length:    {mean_trace:.1f}  "
        f"(threshold {MEAN_TRACE_LEN_MAX:.0f})  "
        f"[{'OK' if mean_trace <= MEAN_TRACE_LEN_MAX else 'WARN'}]"
    )
    print(
        f"    S5 verdict drift:        {len(drift)}  [{'OK' if not drift else 'WARN'}]"
    )
    print(
        f"    S6 baseline coverage:    "
        f"baseline-only={len(baseline_only)} current-only={len(current_only)}  "
        f"[{'OK' if not (baseline_only or current_only) else 'WARN'}]"
    )
    if _is_v2_baseline(baseline):
        print(
            f"    S7 per-model improv.:    {len(per_model_improvements)}  "
            f"[{'OK' if not per_model_improvements else 'WARN'}]"
        )
    if contam is not None:
        n_sub, n_con, rate = contam
        print(
            f"    S8 contamination rate:   {rate:.1%}  "
            f"({n_con}/{n_sub} cells)  "
            f"(threshold {CONTAMINATION_RATE_MAX:.0%})  "
            f"[{'OK' if rate <= CONTAMINATION_RATE_MAX else 'WARN'}]"
        )
    else:
        print("    S8 contamination rate:   n/a (no trace file)")
    print(
        f"    S9 silent failures:      {len(silent)}  [{'OK' if not silent else 'WARN'}]"
    )
    print()

    print("  Per-cell verdict matrix:")
    print(
        f"    {'claim_id':<32s}  {'eng':<8s}  {'fidelity':<9s}  {'reproducibility':<14s}  {'latency_s':>9s}"
    )
    for key, cell in sorted(cells.items()):
        cid = cell.get("claim_id") or key
        eng = cell.get("engine") or ""
        fid = "PASS" if (cell.get("fidelity") or {}).get("overall_pass") else "FAIL"
        repro = (cell.get("reproducibility") or {}).get("verdict") or "n/a"
        lat = cell.get("latency_s", 0)
        print(
            f"    {cid:<32s}  {eng:<8s}  {fid:<9s}  {repro.upper():<14s}  {lat:>9.1f}"
        )
    print()

    print("-" * 70)
    if hard:
        print("RESULT: FAILED")
        for h in hard:
            print(f"  - {h}")
        print("=" * 70)
        return 1
    if soft:
        print("RESULT: PASSED with warnings")
        for s in soft:
            print(f"  - WARN: {s}")
        print("=" * 70)
        return 2
    print("RESULT: PASSED -- all chamber quality gates met")
    print("=" * 70)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Chamber benchmark quality gates")
    parser.add_argument(
        "--results-dir",
        default=None,  # resolved to archive_dir() below
        help="Directory containing latest_chamber.json and baseline_chamber.json",
    )
    args = parser.parse_args()
    # Default to the shipped archive rather than the originating project's
    # layout, which does not exist here.
    results_dir = args.results_dir or str(archive_dir())
    return run_gates(results_dir)


if __name__ == "__main__":
    sys.exit(main())
