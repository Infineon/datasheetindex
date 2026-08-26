"""Dispatch-record statistics requested during the EMNLP 2026 review.

Two numbers the reviewers asked for that the paper did not report. Both are
pure post-hoc analysis of stored results -- no LLM calls.

1. **Cross-check predicate distribution.** A reviewer could not determine what
   counts as a "cross-check" and inferred from Figure 3 (whose per-cell tool
   means are mostly below one call) that any such predicate would flag most
   clean cells. The predicate is ``search_text`` OR
   ``extract_table_markdown`` (see ``chamberbench/silent_failure.py``); this
   reports how often each fires across the clean fidelity-passing agentic
   cells, which is what makes the zero-false-positive result interpretable.
   Note the population is **207**, not the 280 the paper reports -- see
   ``_iter_clean_cells`` for the double count.

2. **Latency decomposition.** A reviewer asked whether tool-layer overhead is
   separable from provider queueing. Every dispatch step carries its own
   ``latency_ms``, so local tool execution can be summed per cell and compared
   against the cell's wall-clock ``latency_s``.

Run:
    uv run python scripts/dispatch_stats.py
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterator
from typing import Any

from chamberbench.claimsio import archive_dir
from chamberbench.silent_failure import (
    _DATASHEET_NAV_TOOLS,
    _DATASHEET_VERIFICATION_TOOLS,
)

RESULTS_DIR = archive_dir()
BASELINE_PATH = RESULTS_DIR / "baseline_chamber.json"
VARIANCE_PATH = RESULTS_DIR / "variance_chamber.json"

# Per-model agentic trace logs, used only for the latency decomposition.
TRACE_FILES = {
    "claudesonnet4.6": "latest_traces.claudesonnet4.6.jsonl",
    "gpt-5.1": "latest_traces.gpt-5.1.jsonl",
    "qwen3.6-27b": "latest_traces.qwen3.6-27b.jsonl",
}

# Locally executed tools. Their latency is our code (pymupdf, datasheetindex,
# the chamber package), not provider queueing.
_LOCAL_TOOLS = _DATASHEET_NAV_TOOLS | {
    "list_experiments",
    "query_dataset",
    "cross_sensor_check",
    "get_experiment_metadata",
    "get_ground_truth_graph",
}


def _iter_clean_cells() -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Yield (model, claim_id, cell) for every agentic cell, WITHOUT double-counting.

    The variance suite's repeat 1 is not an independent run: the aggregation
    helper's ``import_repeat_one`` copies the agentic cells of
    ``baseline_chamber.json`` into it, and every repeat-1 record carries
    ``source == "imported:baseline_chamber.json"``. Verified cell-by-cell:
    fidelity verdict, ``n_tool_calls_by_tool`` and ``latency_s`` are identical
    for all 25 cells in all three models.

    Iterating both files therefore counts the baseline run twice -- the error
    behind the published "280 clean cells", whose true unique population is 207.
    ``scripts/silent_failure_fp_scan.py`` has the same bug; its own output prints
    "73 + 207 = 280" where the 73 is a strict subset of the 207. We iterate the
    variance suite alone, which already contains the baseline as repeat 1.
    """
    variance = json.loads(VARIANCE_PATH.read_text(encoding="utf-8"))
    for model, repeats in (variance.get("runs") or {}).items():
        for run in repeats or []:
            for claim_id, cell in (run.get("cells") or {}).items():
                if isinstance(cell, dict):
                    yield model, claim_id, cell


def _is_clean(cell: dict[str, Any]) -> bool:
    fid = cell.get("fidelity") or {}
    return (
        bool(fid.get("overall_pass"))
        and not fid.get("engine_error")
        and not cell.get("engine_error")
    )


def report_predicate() -> None:
    """How often the cross-check predicate is satisfied on clean cells."""
    per_model: dict[str, Counter[str]] = defaultdict(Counter)
    totals: Counter[str] = Counter()
    calls: Counter[str] = Counter()
    n = 0

    for model, _claim_id, cell in _iter_clean_cells():
        if not _is_clean(cell):
            continue
        n += 1
        by_tool = cell.get("n_tool_calls_by_tool") or {}
        for name, count in by_tool.items():
            calls[name] += int(count)

        nav = sum(int(c) for t, c in by_tool.items() if t in _DATASHEET_NAV_TOOLS)
        satisfied = any(int(by_tool.get(t, 0)) for t in _DATASHEET_VERIFICATION_TOOLS)
        per_model[model]["cells"] += 1
        totals["cells"] += 1
        for tool in _DATASHEET_VERIFICATION_TOOLS:
            if int(by_tool.get(tool, 0)):
                per_model[model][tool] += 1
                totals[tool] += 1
        if nav:
            per_model[model]["nav"] += 1
            totals["nav"] += 1
        if satisfied:
            per_model[model]["predicate"] += 1
            totals["predicate"] += 1

    print("=" * 78)
    print("CROSS-CHECK PREDICATE OVER CLEAN FIDELITY-PASSING AGENTIC CELLS")
    print("=" * 78)
    print(f"predicate = at least one call to {sorted(_DATASHEET_VERIFICATION_TOOLS)}")
    print(f"clean cells: {n}")
    print()
    hdr = f"{'model':<20}{'cells':>7}{'nav>0':>8}{'search_text':>13}{'extract_tbl':>13}{'predicate':>11}"
    print(hdr)
    for model in sorted(per_model):
        m = per_model[model]
        print(
            f"{model:<20}{m['cells']:>7}{m['nav']:>8}{m['search_text']:>13}"
            f"{m['extract_table_markdown']:>13}{m['predicate']:>11}"
        )
    print(
        f"{'ALL':<20}{totals['cells']:>7}{totals['nav']:>8}{totals['search_text']:>13}"
        f"{totals['extract_table_markdown']:>13}{totals['predicate']:>11}"
    )
    print()
    # Printed as an explicit ratio because that is how the response states it, and
    # a claim should be checkable against the literal string it is written as.
    print(f"predicate satisfied on {totals['predicate']}/{totals['cells']} clean cells")
    print(f"navigation present on   {totals['nav']}/{totals['cells']}")
    print(
        f"extract_table_markdown present on {totals['extract_table_markdown']}/{totals['cells']}"
    )
    print()
    print("Mean calls per clean cell (the Figure 3 quantity):")
    for tool in sorted(_DATASHEET_NAV_TOOLS):
        print(f"    {tool:<26}{calls[tool] / n:>6.2f}")
    nav_total = sum(calls[t] for t in _DATASHEET_NAV_TOOLS)
    print(f"    {'(any navigation)':<26}{nav_total / n:>6.2f}")
    print()


def report_latency() -> None:
    """Split measured wall-clock cell latency into local tool time and the rest."""
    # A trace file can hold several attempts at the same claim: the suite runs
    # under `--reruns 1`, and older pre-two-pass runs linger in the same log.
    # Summing every attempt inflates tool time against a single cell's wall
    # clock, so we group by run_id and keep the attempt whose per-tool counts
    # match the cell's own `n_tool_calls_by_tool` -- i.e. the run that produced
    # the cell we are dividing into.
    per_run: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {"ms": 0.0, "counts": Counter()}
    )
    # Fail loudly on a missing trace log. Skipping silently printed a latency
    # header with zero rows and exit code 0, so a third party running this script
    # without the trace files got no numbers and no indication that the numbers
    # were missing -- the worst possible outcome for a script we cite as the
    # reproduction path.
    missing = [
        name for name in TRACE_FILES.values() if not (RESULTS_DIR / name).exists()
    ]
    if missing:
        raise SystemExit(
            "cannot decompose latency: missing trace log(s) "
            + ", ".join(missing)
            + f"\nexpected under {RESULTS_DIR}. These are the sole input to the latency"
            + " split; re-run the chamber suite or fetch the committed logs."
        )
    for model, filename in TRACE_FILES.items():
        path = RESULTS_DIR / filename
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                rec = json.loads(line)
                name = rec.get("tool_name")
                if name in _LOCAL_TOOLS and rec.get("latency_ms") is not None:
                    key = (model, rec.get("claim_id", ""), rec.get("run_id", ""))
                    per_run[key]["ms"] += float(rec["latency_ms"])
                    per_run[key]["counts"][name] += 1

    baseline_cells: dict[tuple[str, str], dict[str, Any]] = {}
    _base = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    for claim_id, by_engine in (_base.get("results") or {}).items():
        for model, cell in (by_engine.get("agentic") or {}).items():
            if isinstance(cell, dict):
                baseline_cells[(model, claim_id)] = cell

    tool_ms: dict[tuple[str, str], float] = {}
    unmatched = 0
    # The fingerprint match is not guaranteed unique: two attempts at the same
    # claim can produce identical per-tool counts, in which case picking one is an
    # arbitrary tie-break. Report how many cells that affects and how much time is
    # at stake, so the caveat is measured rather than asserted.
    ambiguous: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (model, claim_id, _run_id), rec in per_run.items():
        cell = baseline_cells.get((model, claim_id))
        if cell is None:
            continue
        want = {
            k: int(v)
            for k, v in (cell.get("n_tool_calls_by_tool") or {}).items()
            if k in _LOCAL_TOOLS
        }
        if dict(rec["counts"]) == want:
            tool_ms[(model, claim_id)] = rec["ms"]
            ambiguous[(model, claim_id)].append(rec["ms"])
    for key in baseline_cells:
        if key not in tool_ms and key[1]:
            unmatched += 1
    if unmatched:
        print(
            f"note: {unmatched} cells had no attempt matching their recorded tool counts; excluded."
        )
    ties = {k: v for k, v in ambiguous.items() if len(v) > 1}
    if ties:
        spread = max(max(v) - min(v) for v in ties.values()) / 1000.0
        print(
            f"note: {len(ties)} cell(s) had multiple attempts with identical tool counts;"
        )
        print(f"      the tie-break is arbitrary and worth at most {spread:.1f} s.")
        for (model, claim_id), values in ties.items():
            print(
                f"      {model} / {claim_id}: {', '.join(f'{v / 1000:.1f}s' for v in values)}"
            )

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    rows: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"wall": [], "tool": []}
    )
    for claim_id, by_engine in (baseline.get("results") or {}).items():
        for model, cell in (by_engine.get("agentic") or {}).items():
            if not isinstance(cell, dict):
                continue
            wall = cell.get("latency_s")
            key = (model, claim_id)
            if not wall or key not in tool_ms:
                continue
            rows[model]["wall"].append(float(wall))
            rows[model]["tool"].append(tool_ms[key] / 1000.0)

    print("=" * 78)
    print("LATENCY DECOMPOSITION (post-audit baseline run, agentic engine)")
    print("=" * 78)
    print(
        f"{'model':<20}{'cells':>7}{'wall (s)':>11}{'tool exec (s)':>15}{'tool share':>12}"
    )
    for model in sorted(rows):
        wall = rows[model]["wall"]
        tool = rows[model]["tool"]
        if not wall:
            continue
        mw = statistics.mean(wall)
        mt = statistics.mean(tool)
        print(f"{model:<20}{len(wall):>7}{mw:>11.1f}{mt:>15.1f}{100 * mt / mw:>11.1f}%")
    print()
    print("The remainder is model generation plus gateway queueing, which the")
    print("trace cannot separate further -- a single provider turn is one span.")
    print()


def main() -> int:
    report_predicate()
    report_latency()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
