"""Audit chamber-side tool use that precedes ``submit_claim_result``.

The chamber benchmark's agentic loop exposes two tool families to the
agent: datasheet navigation tools (``build_datasheet``, ``search_text``,
...) and chamber-side observation tools (``list_experiments``,
``query_dataset``, ``cross_sensor_check``, ``run_simulator``,
``get_experiment_metadata``, ``get_ground_truth_graph``). The system
prompt instructs the agent NOT to let chamber-side observations
influence what it records in the datasheet-side ``extracted`` field --
but nothing in the loop enforces this. An agent that calls chamber
tools before submitting could in principle let what it saw bias the
extracted value.

This module gives a static, post-hoc detection mechanism. For each
agentic cell it computes:

  - ``n_chamber_calls_before_submit``: count of chamber-tool calls
    that fired before the cell's ``submit_claim_result`` event.
  - ``first_chamber_tool_step``: the step index of the first such
    call (None if no chamber tool was used before submit).
  - ``chamber_tools_used``: the set of chamber tool names invoked
    before submit.

A cell is *contamination-opportunity-positive* if
``n_chamber_calls_before_submit > 0``. Whether the opportunity
actually became bias is undecidable from traces alone (the model's
reasoning is private); the metric only quantifies the surface area.

Reported as a soft signal: high prevalence does not invalidate
fidelity results, but it bounds how strong a methodology claim the
paper can make about datasheet-side independence. The named structural
fix is the two-pass agent design (datasheet phase freezes
``extracted`` before chamber tools are exposed) -- see the
methodology doc's "Cross-contamination audit" section.

Trace-schema aware: post-two-pass traces (schema v2) carry a ``phase``
field, so a chamber-tool call counts as contaminating iff it fired in
the ``extraction`` phase -- which the freeze makes structurally
impossible (0% by construction). Pre-two-pass traces (schema v1) have no
``phase``, so the audit falls back to step-ordering relative to the first
finalization tool. Running both lets one tool compute the before/after
delta.

Run:
    uv run python -m chamberbench.contamination_audit \\
        --traces archive/latest_traces.claudesonnet4.6.jsonl \\
        --out archive/contamination_audit.claudesonnet4.6.json

Or aggregate across all models on disk:
    uv run python -m chamberbench.contamination_audit \\
        --results-dir archive
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Chamber-side tool names that, if invoked before ``submit_claim_result``,
# create the opportunity for the agent's datasheet-side ``extracted``
# value to be biased by chamber observations. Mirrors the tool surface
# exposed by ``chamberbench.harness.chamber_tools._make_chamber_tools``.
CHAMBER_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "list_experiments",
        "get_experiment_metadata",
        "query_dataset",
        "cross_sensor_check",
        "run_simulator",
        "get_ground_truth_graph",
    }
)

# Finalization tools that mark the freeze boundary. ``submit_extraction`` is
# the two-pass freeze point; ``submit_claim_result`` is the pre-two-pass name,
# kept so archived (schema v1) traces still resolve a boundary.
SUBMIT_TOOL_NAMES = frozenset({"submit_extraction", "submit_claim_result"})


def _iter_trace_events(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL trace file, skipping malformed lines and session
    sentinels (which carry no claim_id / run_id pairing useful here).
    """
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("kind") == "session_start":
            continue
        out.append(ev)
    return out


def analyze_traces(traces_path: Path) -> dict[str, Any]:
    """Compute per-cell contamination metrics from one trace file.

    Returns a dict suitable for serialisation:

      {
        "traces_path": "...",
        "n_cells_total": int,
        "n_cells_with_submit": int,
        "n_cells_contaminated": int,
        "contamination_rate": float,  # contaminated / cells_with_submit
        "per_cell": [
          {
            "claim_id": str,
            "engine": str,
            "run_id": str,
            "n_chamber_calls_before_submit": int,
            "first_chamber_tool_step": int | None,
            "chamber_tools_used": [str, ...],
            "submit_step": int | None,
            "total_steps": int,
          },
          ...
        ]
      }
    """
    events = _iter_trace_events(traces_path)
    by_cell: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for ev in events:
        cid = ev.get("claim_id")
        eng = ev.get("engine")
        rid = ev.get("run_id")
        if cid and eng and rid:
            by_cell[(rid, cid, eng)].append(ev)

    per_cell: list[dict[str, Any]] = []
    n_with_submit = 0
    n_contaminated = 0
    for (rid, cid, eng), cell_events in sorted(by_cell.items()):
        cell_events.sort(key=lambda e: e.get("step", 0))
        # Freeze boundary: the step of the first finalization tool.
        submit_step: int | None = None
        for ev in cell_events:
            if ev.get("tool_name") in SUBMIT_TOOL_NAMES:
                submit_step = ev.get("step")
                break
        # schema v2 traces carry an explicit per-step phase; v1 traces do not,
        # and fall back to step-ordering relative to submit_step.
        has_phase = any("phase" in ev for ev in cell_events)

        chamber_calls_before: list[dict[str, Any]] = []
        # Denominator: cells that finalized the extraction phase (a submit tool
        # fired). A cell that errored before freezing has no assessable
        # extraction phase and is excluded, matching the pre-two-pass semantics.
        # `has_phase` only selects how the extraction phase is identified below.
        if submit_step is not None:
            n_with_submit += 1
            for ev in cell_events:
                if ev.get("kind") != "tool_call":
                    continue
                if ev.get("tool_name") not in CHAMBER_TOOL_NAMES:
                    continue
                if has_phase:
                    in_extraction = ev.get("phase", "extraction") == "extraction"
                else:
                    in_extraction = (
                        submit_step is not None and (ev.get("step") or 0) < submit_step
                    )
                if in_extraction:
                    chamber_calls_before.append(ev)
        if chamber_calls_before:
            n_contaminated += 1
        per_cell.append(
            {
                "claim_id": cid,
                "engine": eng,
                "run_id": rid,
                "n_chamber_calls_before_submit": len(chamber_calls_before),
                "first_chamber_tool_step": (
                    chamber_calls_before[0].get("step")
                    if chamber_calls_before
                    else None
                ),
                "chamber_tools_used": sorted(
                    {ev["tool_name"] for ev in chamber_calls_before}
                ),
                "submit_step": submit_step,
                "total_steps": len(cell_events),
            }
        )

    n_total = len(by_cell)
    rate = (n_contaminated / n_with_submit) if n_with_submit else 0.0
    return {
        "traces_path": str(traces_path),
        "n_cells_total": n_total,
        "n_cells_with_submit": n_with_submit,
        "n_cells_contaminated": n_contaminated,
        "contamination_rate": rate,
        "per_cell": per_cell,
    }


def write_summary(out_path: Path, summary: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _aggregate_dir(results_dir: Path) -> list[tuple[Path, Path]]:
    """Find every ``latest_traces.{model}.jsonl`` under results_dir and
    return ``(traces_path, summary_out_path)`` pairs.
    """
    pairs: list[tuple[Path, Path]] = []
    for tp in sorted(results_dir.glob("latest_traces.*.jsonl")):
        # Skip the canonical un-suffixed file (it's a duplicate of the
        # Sonnet per-model trace; including it would double-count).
        if tp.name == "latest_traces.jsonl":
            continue
        suffix = tp.name.removeprefix("latest_traces.").removesuffix(".jsonl")
        op = results_dir / f"contamination_audit.{suffix}.json"
        pairs.append((tp, op))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    ap.add_argument(
        "--traces", type=Path, default=None, help="Single trace JSONL to analyse."
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to write the summary JSON. Defaults next to --traces with name 'contamination_audit.{model}.json'.",
    )
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Process every latest_traces.{model}.jsonl in this directory (writes one summary per model).",
    )
    args = ap.parse_args()

    if (args.traces is None) == (args.results_dir is None):
        ap.error("provide exactly one of --traces or --results-dir")

    pairs: list[tuple[Path, Path]]
    if args.results_dir is not None:
        pairs = _aggregate_dir(args.results_dir)
        if not pairs:
            print(f"no latest_traces.*.jsonl under {args.results_dir}", file=sys.stderr)
            return 1
    else:
        out = args.out
        if out is None:
            suffix = (
                args.traces.name.removeprefix("latest_traces.").removesuffix(".jsonl")
                or "default"
            )
            out = args.traces.parent / f"contamination_audit.{suffix}.json"
        pairs = [(args.traces, out)]

    print("=" * 72)
    print("CHAMBER CROSS-CONTAMINATION AUDIT")
    print("=" * 72)
    print(f"  {'model':<22s} {'cells':>5s} {'submit':>7s} {'contam':>7s} {'rate':>7s}")
    print("-" * 72)
    grand_submit = 0
    grand_contam = 0
    for traces_path, out_path in pairs:
        summary = analyze_traces(traces_path)
        write_summary(out_path, summary)
        model = (
            traces_path.name.removeprefix("latest_traces.").removesuffix(".jsonl")
            or "?"
        )
        print(
            f"  {model:<22s} {summary['n_cells_total']:>5d} "
            f"{summary['n_cells_with_submit']:>7d} "
            f"{summary['n_cells_contaminated']:>7d} "
            f"{summary['contamination_rate'] * 100:>6.1f}%"
        )
        grand_submit += summary["n_cells_with_submit"]
        grand_contam += summary["n_cells_contaminated"]
    if grand_submit:
        print("-" * 72)
        print(
            f"  {'TOTAL':<22s} {'':>5s} {grand_submit:>7d} "
            f"{grand_contam:>7d} "
            f"{grand_contam / grand_submit * 100:>6.1f}%"
        )
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
