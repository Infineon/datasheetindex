"""Prepare the failure-attribution gold-labelling exercise.

Day 18 of the chamber paper plan. Produces three artifacts:

  1. ``archive/classifier_auto.{model}.json`` -- per-model
     classifier auto-labels (the existing manual-rule pass), computed
     from the union of snapshot + latest trace files. These are the
     "predicted" labels; the human-supplied gold labels in step 3 will
     be diffed against this.
  2. ``archive/sampled_cells.json`` -- the sampled
     (claim_id, engine, model) tuples plus the random seed for
     reproducibility.
  3. ``data/classifier_gold.yaml`` -- the blind labelling
     skeleton. For each sampled cell, every event is listed with
     ``gold_label:`` and ``notes:`` fields blank. The classifier's
     auto-label is *deliberately not included* in this YAML so the
     annotator labels blind.

Allocation strategy: 30 cells stratified across model × component.
All three models have 25 agentic cells with traces on disk, so the
sample is balanced 10/10/10. claudesonnet4.6 and gpt-5.1 are from the
2026-05-20 Responses-API-fork re-run; qwen3.6-27b is from the
2026-05-21 re-run on the prod gateway (qwen3.5-27b is retired).
Documented in the YAML metadata block.

Run:
    uv run python scripts/prepare_gold_labelling.py [--seed 0] [--n 30]
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from chamberbench.classifier import classify_run

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from chamberbench.claimsio import archive_dir, data_dir

RESULTS_DIR = archive_dir()
EVAL_DIR = data_dir()
SAMPLED_CELLS = RESULTS_DIR / "sampled_cells.json"
GOLD_YAML = EVAL_DIR / "classifier_gold.yaml"

MODELS = ("claudesonnet4.6", "gpt-5.1", "qwen3.6-27b")


def _component_of(claim_id: str) -> str:
    if claim_id.startswith("dps310-"):
        return "dps310"
    if claim_id.startswith("si115x-"):
        return "si115x"
    if claim_id.startswith("acs70331-"):
        return "acs70331"
    return "other"


def _load_trace_events(path: Path) -> list[dict[str, Any]]:
    """Load JSONL trace events from a single file, skipping malformed lines."""
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def merge_traces_per_model() -> dict[str, list[dict[str, Any]]]:
    """Merge snapshot + latest trace events per model, deduplicating by
    ``(run_id, step)``.

    Source-precedence: snapshot first, then latest. After the
    2026-05-20 re-run the per-model snapshot is refreshed to a copy of
    latest_traces, so for every model the two files share run_ids and
    dedup-by-(run_id, step) collapses them to one set. The two-source
    merge is retained for the case where a snapshot and a partial
    latest file carry disjoint sessions.

    Session-start sentinels carry no ``run_id`` field; they are kept
    unchanged on a first-seen-wins basis (won't be sampled anyway --
    they have no claim_id).
    """
    per_model_events: dict[str, list[dict[str, Any]]] = {}
    for m in MODELS:
        events: list[dict[str, Any]] = []
        seen: set[tuple[Any, Any]] = set()
        for src in (
            RESULTS_DIR / f"snapshot_layer2_traces.{m}.jsonl",
            RESULTS_DIR / f"latest_traces.{m}.jsonl",
        ):
            for ev in _load_trace_events(src):
                key = (ev.get("run_id"), ev.get("step"))
                if key[0] is not None and key in seen:
                    continue
                seen.add(key)
                events.append(ev)
        per_model_events[m] = events
    return per_model_events


def cells_with_traces(
    per_model_events: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[tuple[str, str], list[dict[str, Any]]]]:
    """Group events by (claim_id, engine) within each model.

    Returns ``{model: {(claim_id, engine): [events]}}``. Session-start
    sentinels are dropped; only events with a populated claim_id remain.
    """
    out: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = {}
    for m, events in per_model_events.items():
        cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for ev in events:
            if ev.get("kind") == "session_start":
                continue
            cid = ev.get("claim_id")
            eng = ev.get("engine")
            if not cid or not eng:
                continue
            cells[(cid, eng)].append(ev)
        out[m] = dict(cells)
    return out


def group_events_by_attempt(
    events: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Split a cell's events into per-attempt groups keyed by ``run_id``.

    A pytest ``--reruns`` retry (triggered when an agentic cell ends in
    ``terminal_without_submit`` -> ``SubmitToolNotCalledError``) produces a
    second full agentic session for the same ``(claim_id, engine)``, with its
    own ``run_id`` and ``turn_idx`` reset to 0. Grouping only by
    ``(claim_id, engine)`` and sorting by ``turn_idx`` interleaved the failed
    attempt and its rerun, so both ``build_datasheet`` calls landed adjacent at
    turn 0 -- reading as a within-turn double-call rather than a retry.

    Returns ``[(run_id, sorted_events), ...]`` with attempts in first-appearance
    (chronological) order and each attempt's events sorted by ``(turn_idx,
    kind)`` -- tool_calls before the final_output within a turn.
    """
    order: list[str] = []
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ev in events:
        run_id = str(ev.get("run_id") or "")
        if run_id not in by_run:
            order.append(run_id)
        by_run[run_id].append(ev)

    grouped: list[tuple[str, list[dict[str, Any]]]] = []
    for run_id in order:
        attempt = sorted(
            by_run[run_id],
            key=lambda e: (
                e.get("turn_idx") or 0,
                0 if e.get("kind") == "tool_call" else 1,
            ),
        )
        grouped.append((run_id, attempt))
    return grouped


def stratified_sample(
    cells_by_model: dict[str, dict[tuple[str, str], list[dict[str, Any]]]],
    n: int,
    seed: int,
) -> list[tuple[str, str, str]]:
    """Stratified sample over (model, component); agentic engine only.

    Returns ``[(claim_id, engine, model), ...]`` length ~n. Allocation:
    each model gets ``n // n_models`` cells (with the remainder going
    to the model with the largest pool to avoid biasing toward smaller
    pools); within each model, sampling is proportional to component
    pool size. Falls back to "take all available" when a model has
    fewer cells than its allocation.
    """
    rng = random.Random(seed)
    sample: list[tuple[str, str, str]] = []

    models_with_cells: list[tuple[str, list[tuple[str, str]]]] = []
    for m in MODELS:
        agentic = sorted(
            (cid, eng) for (cid, eng) in cells_by_model.get(m, {}) if eng == "agentic"
        )
        if agentic:
            models_with_cells.append((m, agentic))
    if not models_with_cells:
        return []

    n_models = len(models_with_cells)
    base = n // n_models
    extra = n - base * n_models
    # Give the extras to the models with the largest pools (avoids
    # over-sampling a tiny pool when n doesn't divide evenly).
    sorted_by_pool = sorted(models_with_cells, key=lambda mc: -len(mc[1]))
    per_model_target: dict[str, int] = {}
    for i, (m, cells) in enumerate(sorted_by_pool):
        target = base + (1 if i < extra else 0)
        per_model_target[m] = min(target, len(cells))  # cap at pool size

    for m, agentic in models_with_cells:
        take = per_model_target[m]
        if take == 0:
            continue
        # Stratify within model by component, proportional allocation.
        by_comp: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for cid, eng in agentic:
            by_comp[_component_of(cid)].append((cid, eng))
        comps_sorted = sorted(by_comp)
        sizes = [len(by_comp[c]) for c in comps_sorted]
        total = sum(sizes)
        if total == 0:
            continue
        alloc = [round(take * s / total) for s in sizes]
        while sum(alloc) > take:
            alloc[alloc.index(max(alloc))] -= 1
        while sum(alloc) < take:
            alloc[alloc.index(min(alloc))] += 1
        for comp, k in zip(comps_sorted, alloc, strict=True):
            pool = sorted(by_comp[comp])
            rng.shuffle(pool)
            for cid, eng in pool[:k]:
                sample.append((cid, eng, m))

    return sample


def run_classifier_per_model(
    per_model_events: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Run the manual-rule classifier per model and write per-model
    auto-label files.

    For each model we materialise a merged trace JSONL + a copy of the
    per-model summary JSON, then call ``classify_run`` in-process. The
    classifier modifies both files in place, so we use temp copies to
    avoid mutating the canonical artifacts.
    """
    summaries: dict[str, dict[str, Any]] = {}
    for m in MODELS:
        src_summary = RESULTS_DIR / f"latest_chamber.{m}.json"
        if not src_summary.exists():
            print(f"  {m}: skipping -- no latest_chamber.{m}.json on disk")
            continue
        # Materialise inputs for the classifier (writes back in place).
        merged_trace = RESULTS_DIR / f"classifier_auto.{m}.jsonl"
        merged_summary = RESULTS_DIR / f"classifier_auto.{m}.json"
        shutil.copyfile(src_summary, merged_summary)
        events = per_model_events[m]
        with merged_trace.open("w", encoding="utf-8") as fh:
            for ev in events:
                fh.write(json.dumps(ev, ensure_ascii=False) + "\n")

        result = classify_run(
            traces_path=merged_trace,
            summary_path=merged_summary,
            llm_assist=False,
        )
        # Per-claim attribution histogram
        per_claim_count = len(result.get("per_claim", {}))
        total = result.get("total", {})
        ok_n = total.get("ok", 0)
        non_ok = sum(v for k, v in total.items() if k != "ok")
        print(
            f"  {m}: classified {per_claim_count} cells; "
            f"ok={ok_n}, non-ok={non_ok} (across {sum(total.values())} events)"
        )
        summaries[m] = result
    return summaries


def _load_claims_gold() -> dict[str, dict[str, Any]]:
    """Return ``{claim_id: {parameter, value_contains, source_page}}`` from claims.yaml.

    Hand-parsed (no PyYAML at script-run-time required) -- we want the
    `parameter` string and the gold `value_contains` substrings the agent
    is supposed to surface, plus source page hints when present, to show
    next to each cell in the YAML so the annotator can judge "did the
    agent extract the right thing?".
    """
    import yaml as _yaml  # local: pyyaml is already a transitive dev dep

    path = EVAL_DIR / "claims.yaml"
    if not path.exists():
        return {}
    raw = _yaml.safe_load(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for c in raw.get("claims", []):
        cid = c.get("id")
        if not cid:
            continue
        out[cid] = {
            "parameter": c.get("parameter", ""),
            "value_contains": c.get("value_contains", []),
            "source_page": c.get("source_page"),
            "source_text": (c.get("source_text") or "")[:200],
        }
    return out


def _load_cells_extracted() -> dict[str, dict[str, Any]]:
    """Return ``{cell_id: extracted_summary}`` for every cell across all
    per-model latest_chamber files.

    Each value is a flat dict with the extracted parameter name, a short
    summary of the values list, and the fidelity verdict + confidence.
    Used to surface "what did the agent actually extract?" next to the
    cell's event list.
    """
    out: dict[str, dict[str, Any]] = {}
    for m in MODELS:
        path = RESULTS_DIR / f"latest_chamber.{m}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for key, cell in (data.get("results") or {}).items():
            cid = cell.get("claim_id") or key.split("|", 1)[0]
            eng = cell.get("engine") or (key.split("|", 1)[1] if "|" in key else "")
            cell_id = f"{cid}|{eng}|{m}"
            cr = cell.get("claim_result") or {}
            ext = cr.get("extracted") or {}
            values = ext.get("values") or []
            # Render values list compactly
            value_str = ""
            for v in values[:2]:
                parts = []
                for k in ("min_value", "typical_value", "max_value", "value"):
                    if v.get(k) is not None:
                        parts.append(f"{k}={v.get(k)}")
                if v.get("unit"):
                    parts.append(f"unit={v.get('unit')}")
                if v.get("conditions"):
                    parts.append(f"@ {v.get('conditions')[:60]}")
                value_str += ("; ".join(parts) + " || ") if parts else ""
            value_str = value_str.rstrip(" |") or "(no values)"
            fid = cell.get("fidelity") or {}
            out[cell_id] = {
                "parameter": ext.get("parameter", ""),
                "values_summary": value_str[:240],
                "fidelity_pass": fid.get("overall_pass"),
                "confidence": fid.get("confidence"),
                "repro_verdict": (
                    (cell.get("reproducibility") or {}).get("verdict") or ""
                ).lower(),
                "n_steps": cell.get("n_steps", 0),
            }
    return out


def _yaml_short(s: str, n: int = 140) -> str:
    """Compact a string for embedding in a YAML comment line.

    Collapses whitespace + newlines, truncates to n chars with ellipsis.
    """
    if not s:
        return ""
    s = " ".join(s.split())
    if len(s) > n:
        return s[: n - 3] + "..."
    return s


def build_gold_skeleton(
    sample: list[tuple[str, str, str]],
    cells_by_model: dict[str, dict[tuple[str, str], list[dict[str, Any]]]],
    seed: int,
    n: int,
    annotator: str = "",
) -> str:
    """Render the cell-level gold-labelling YAML.

    One label per cell. Each cell's events are shown as YAML comments
    including the agent's reasoning, tool inputs, tool outputs (all
    truncated for readability), plus cell-level context (gold value
    from claims.yaml, agent's final extracted value). Blind: classifier
    auto-labels are NOT included so the annotator labels without bias.
    """
    gold = _load_claims_gold()
    extracted = _load_cells_extracted()
    strata: dict[str, int] = defaultdict(int)
    for cid, _eng, model in sample:
        comp = _component_of(cid)
        strata[f"{model}|{comp}"] += 1

    lines: list[str] = []
    lines.append("# Chamber failure-attribution gold labels (CELL-LEVEL)")
    lines.append("#")
    lines.append("# INSTRUCTIONS: docs/chamber_annotator_guide.md -- read that first.")
    lines.append("# Label blind: do NOT open archive/classifier_auto.*.json")
    lines.append(
        "# (the classifier's own predictions) or another annotator's gold file."
    )
    lines.append("#")
    lines.append("# One gold_label per cell. Scan the events listed in the")
    lines.append("# comment block above each cell, then pick the label that")
    lines.append("# best describes whether the classifier got this cell right.")
    lines.append("#")
    lines.append("# Allowed gold_label values (see")
    lines.append("# docs/chamber_failure_attribution_protocol.md for definitions):")
    lines.append(
        "#   classifier_correct                 -- classifier's per-event labels look right"
    )
    lines.append(
        "#   classifier_missed_tool_output      -- a tool returned bad content; classifier said ok"
    )
    lines.append(
        "#   classifier_missed_tool_selection   -- agent took an inefficient path; not caught"
    )
    lines.append(
        "#   classifier_missed_condition_omission -- agent dropped a load-bearing condition"
    )
    lines.append(
        "#   classifier_missed_verification_skipped -- agent skipped a real verification check"
    )
    lines.append(
        "#   classifier_overflagged             -- classifier said non-ok but I'd say ok"
    )
    lines.append("#")
    lines.append(
        "# Multiple labels possible -- use a list for that cell (`gold_label:`"
    )
    lines.append("# can be a single string or a YAML list of strings).")
    lines.append("#")
    lines.append("# Leave `notes:` empty unless you want to explain a non-obvious")
    lines.append("# call or flag a case the rubric does not cover.")
    lines.append("metadata:")
    lines.append(f"  sampled_at: '{datetime.now(UTC).isoformat()}'")
    lines.append(f"  seed: {seed}")
    lines.append(f"  n_cells: {len(sample)}")
    lines.append(f"  target_n: {n}")
    lines.append("  labelling_mode: cell-level")
    if annotator:
        lines.append(f"  annotator: '{annotator}'")
    else:
        lines.append("  annotator: ''  # FILL IN: your name when you start labelling")
    lines.append("  rubric_version: 1")
    lines.append("  strata:")
    for k in sorted(strata):
        lines.append(f"    '{k}': {strata[k]}")
    lines.append("  caveats:")
    lines.append(
        "    - 'Claude and GPT cells are from the 2026-05-20 Responses-API-fork'"
    )
    lines.append(
        "    - 'agentic re-run. GPT cells route through the OpenAI Responses API:'"
    )
    lines.append(
        "    - 'agent reasoning is a real reasoning summary. Claude cells use'"
    )
    lines.append(
        "    - 'extended thinking at effort=medium; reasoning is thinking-block'"
    )
    lines.append("    - 'text. qwen3.6-27b cells are from the prod LiteLLM gateway'")
    lines.append("    - '(qwen3.5-27b is retired; the qwen baseline renders datasheet'")
    lines.append("    - 'pages to images). qwen runs through the Anthropic-shape'")
    lines.append("    - 'passthrough to vLLM with reasoning enabled via the'")
    lines.append(
        "    - 'enable_thinking chat-template kwarg, so qwen reasoning is also'"
    )
    lines.append("    - 'thinking-block text. All three models have 25 agentic cells,'")
    lines.append("    - 'so the sample stays balanced 10/10/10.'")
    lines.append("    - ''")
    lines.append(
        "    - 'Blind labelling: classifier auto-labels are NOT in this file.'"
    )
    lines.append(
        "    - 'Scan the event comments and pick the label(s) you would assign.'"
    )
    lines.append("cells:")
    for cid, eng, model in sample:
        events = cells_by_model.get(model, {}).get((cid, eng), [])
        # Split into per-attempt groups so a pytest --reruns retry renders as
        # distinct ATTEMPT blocks instead of interleaving at turn 0 (which
        # reads as a within-turn double build_datasheet rather than a rerun).
        attempts = group_events_by_attempt(events)
        cell_id = f"{cid}|{eng}|{model}"
        gold_info = gold.get(cid, {})
        ext_info = extracted.get(cell_id, {})

        lines.append("")
        lines.append(
            "  # ========================================================================="
        )
        lines.append(f"  # CELL: {cell_id}")
        lines.append(
            "  # ========================================================================="
        )
        # Gold (the right answer from claims.yaml)
        if gold_info:
            lines.append(
                f"  # CLAIM:     {_yaml_short(gold_info.get('parameter', ''), 100)}"
            )
            vc = gold_info.get("value_contains") or []
            if vc:
                lines.append(f"  # GOLD must-contain: {vc}")
            if gold_info.get("source_page"):
                lines.append(f"  # GOLD source: page {gold_info['source_page']}")
            if gold_info.get("source_text"):
                lines.append(
                    f"  # GOLD text:  {_yaml_short(gold_info.get('source_text', ''), 140)}"
                )
        # Agent extracted (what the agent submitted)
        if ext_info:
            lines.append(
                f"  # AGENT extracted: {_yaml_short(ext_info.get('values_summary', ''), 200)}"
            )
            lines.append(
                f"  # AGENT fidelity_pass={ext_info.get('fidelity_pass')} "
                f"conf={ext_info.get('confidence')} "
                f"repro={ext_info.get('repro_verdict')!r} "
                f"n_steps={ext_info.get('n_steps')}"
            )
        n_events_total = sum(len(evs) for _, evs in attempts)
        if len(attempts) > 1:
            lines.append(
                f"  # TRACE: n_events={n_events_total} "
                f"attempts={len(attempts)} (pytest --reruns retry; each attempt "
                f"is a fresh agentic session, turn_idx resets to 0)"
            )
        else:
            lines.append(f"  # TRACE: n_events={n_events_total}")
        # Per-attempt, per-event detail: input, output excerpt, reasoning, error
        for a_idx, (run_id, attempt_events) in enumerate(attempts, start=1):
            if len(attempts) > 1:
                fo = [e for e in attempt_events if e.get("kind") == "final_output"]
                outcome = (fo[-1].get("error") or "ok") if fo else "no final_output"
                lines.append(
                    f"  #   --- ATTEMPT {a_idx}/{len(attempts)} (run_id={run_id}, outcome={outcome}) ---"
                )
            for idx, ev in enumerate(attempt_events):
                tool = ev.get("tool_name") or "-"
                kind = ev.get("kind") or "?"
                turn = ev.get("turn_idx")
                err = _yaml_short(ev.get("error") or "", 120)
                in_summary = _yaml_short(ev.get("tool_input_summary") or "", 100)
                out_summary = _yaml_short(ev.get("tool_output_summary") or "", 180)
                reasoning = _yaml_short(ev.get("agent_reasoning") or "", 140)
                latency = ev.get("latency_ms")
                lat_str = f"{latency}ms" if isinstance(latency, (int, float)) else ""
                lines.append(
                    f"  # [{idx:>2d}] turn={turn} {kind:<13s} tool={tool:<24s} ({lat_str})"
                )
                if reasoning:
                    lines.append(f"  #      reasoning: {reasoning}")
                if in_summary:
                    lines.append(f"  #      input:     {in_summary}")
                if err:
                    lines.append(f"  #      ERROR:     {err}")
                elif out_summary:
                    lines.append(f"  #      output:    {out_summary}")
        lines.append(f"  - cell_id: '{cell_id}'")
        lines.append(f"    claim_id: '{cid}'")
        lines.append(f"    model: '{model}'")
        lines.append(f"    engine: '{eng}'")
        lines.append(f"    n_events: {n_events_total}")
        lines.append(
            "    gold_label: ''  # FILL IN  (string or list of strings; see protocol doc)"
        )
        lines.append("    notes: ''")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0, help="Sampling seed (deterministic)")
    ap.add_argument("--n", type=int, default=30, help="Target sample size")
    ap.add_argument(
        "--out",
        type=Path,
        default=GOLD_YAML,
        help="Where to write the labelling skeleton. Use a new path for a second annotator; "
        "the same --seed reproduces the same cells, which is what makes the two sets comparable.",
    )
    ap.add_argument(
        "--annotator",
        default="",
        help="Pre-fill the annotator name in the metadata block",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an --out file that already carries labels (destroys annotation work)",
    )
    args = ap.parse_args()
    args.out = (args.out if args.out.is_absolute() else Path.cwd() / args.out).resolve()

    # This script used to write one hardcoded path unconditionally. That path
    # holds the annotation behind the paper's classifier-agreement figure, so a
    # re-run to prepare a second annotator would have silently destroyed it.
    if args.out.exists() and not args.force:
        existing = yaml.safe_load(args.out.read_text(encoding="utf-8")) or {}
        labelled = [
            c
            for c in (existing.get("cells") or [])
            if str(c.get("gold_label") or "").strip()
        ]
        if labelled:
            who = (existing.get("metadata") or {}).get("annotator") or "unknown"
            print(
                f"REFUSING to overwrite {args.out}: {len(labelled)} labels already filled in by {who!r}."
            )
            print(
                "Pass --out <new-path> for a second annotator, or --force to discard that work."
            )
            return 1

    print("== Step 1: merging snapshot + latest trace events per model ==")
    per_model_events = merge_traces_per_model()
    for m in MODELS:
        n_evt = len(per_model_events[m])
        print(f"  {m}: {n_evt} merged trace events")

    print()
    print("== Step 2: indexing cells with multi-event traces ==")
    cells_by_model = cells_with_traces(per_model_events)
    for m in MODELS:
        agentic = [(c, e) for (c, e) in cells_by_model.get(m, {}) if e == "agentic"]
        counts = [len(cells_by_model[m][(c, e)]) for (c, e) in agentic]
        if counts:
            print(
                f"  {m}: {len(agentic)} agentic cells, "
                f"events/cell median={statistics.median(counts):.0f} "
                f"min={min(counts)} max={max(counts)}"
            )
        else:
            print(f"  {m}: 0 agentic cells")

    print()
    print(f"== Step 3: stratified sampling n={args.n} (seed={args.seed}) ==")
    sample = stratified_sample(cells_by_model, args.n, args.seed)
    by_strat: dict[str, int] = defaultdict(int)
    for cid, _eng, model in sample:
        by_strat[f"{model}|{_component_of(cid)}"] += 1
    print(f"  sampled {len(sample)} cells:")
    for k in sorted(by_strat):
        print(f"    {k:<32s} {by_strat[k]}")
    SAMPLED_CELLS.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "n": args.n,
                "sample": [
                    {"claim_id": c, "engine": e, "model": m} for c, e, m in sample
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"  wrote {SAMPLED_CELLS.relative_to(PROJECT_ROOT)}")

    print()
    print("== Step 4: running classifier per model (manual rule pass) ==")
    run_classifier_per_model(per_model_events)

    print()
    print("== Step 5: writing gold-labelling YAML skeleton (blind) ==")
    yaml_text = build_gold_skeleton(
        sample, cells_by_model, args.seed, args.n, args.annotator
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml_text, encoding="utf-8")
    try:
        shown = args.out.relative_to(PROJECT_ROOT)
    except ValueError:
        shown = args.out
    print(f"  wrote {shown}")

    total_events = 0
    for cid, eng, model in sample:
        total_events += len(cells_by_model.get(model, {}).get((cid, eng), []))
    print()
    print("== Ready to label ==")
    print(
        f"   {len(sample)} cells (cell-level labels), {total_events} events shown as context."
    )
    print(
        f"   Estimated annotation time: ~{max(15, len(sample) * 2)} minutes at 2 min/cell (scan events, pick label)."
    )
    print("   See docs/chamber_failure_attribution_protocol.md for the rubric.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
