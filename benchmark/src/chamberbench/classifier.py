"""Failure-attribution classifier for the chamber benchmark.

Reads a per-run trace JSONL plus the aggregate `latest_chamber.json`,
assigns each tool_call step (and the final_output, if it was a failure)
into one of four rubric categories from the methodology doc:

  tool_output           -- the tool itself returned wrong content / errored
  tool_selection        -- the agent called the wrong tool, or picked the
                           right tool with the wrong inputs
  condition_omission    -- the agent extracted a number but dropped or
                           misread its operating conditions
  verification_skipped  -- the agent finalized without exercising any
                           cross-check tool (search_text /
                           cross_sensor_check / extract_table_markdown)

Plus two non-failure slots:
  ok                    -- step is in a successful agentic chain
  unclassified          -- residual; rubric did not apply

Manual deterministic rules run first; LLM-assisted classification fills
in only the steps the manual rules left as `unclassified`. Manual always
overrides LLM.

CLI:
    uv run python -m chamber.classifier \\
        --traces archive/latest_traces.claudesonnet4.6.jsonl \\
        --summary archive/latest_chamber.claudesonnet4.6.json \\
        [--llm-assist]

Idempotent: re-running the classifier on the same trace + summary
produces identical attribution slots.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from chamberbench.claimsio import BENCHMARK_ROOT, archive_dir

logger = logging.getLogger(__name__)


Attribution = Literal[
    "tool_output",
    "tool_selection",
    "condition_omission",
    "verification_skipped",
    "engine_error",
    "ok",
    "unclassified",
]


# Tools that constitute "verification" -- using one of these counts as
# the cross-check the system prompt asks for.
_VERIFICATION_TOOLS = frozenset(
    {
        "search_text",
        "extract_table_markdown",
        "cross_sensor_check",
    }
)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        out.append(d)
    return out


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    tmp.replace(path)


def _write_json_atomic(path: Path, obj: Any) -> None:
    """Write a JSON file atomically (.part + os.replace).

    The summary JSON is multi-KB and a SIGINT mid-write would otherwise
    leave a truncated file that the next run cannot parse.
    """
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _read_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Manual rule pass
# ---------------------------------------------------------------------------


_CLASSIFIABLE_KINDS = frozenset({"tool_call", "final_output"})


def _classify_manual(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> tuple[
    list[dict[str, Any]], dict[str, list[Attribution]], dict[str, dict[str, int]]
]:
    """Apply deterministic rules. Returns (rows, per-claim labels,
    per-claim tool_output-by-tool counts).

    Rules applied per step:
      - kind=tool_call AND error != ""              -> tool_output
      - engine_error cell                           -> engine_error
        (covers retry-exhausted, timeouts, model crashes; H4 also catches it)
      - kind=final_output AND error != ""           -> verification_skipped
        (engine ended without structured output and was not engine-error)

    Per-claim rules applied to the final_output row:
      - fidelity FAIL & no verification tool called -> verification_skipped
      - fidelity FAIL & at least one load-bearing condition missing
                                                    -> condition_omission
      - fidelity FAIL otherwise                     -> tool_selection

    Per-claim tail rule:
      - fidelity PASS & repro PASS|INCONCLUSIVE     -> any remaining
        unclassified steps become "ok".
    """
    by_claim_engine: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        if row.get("kind") not in _CLASSIFIABLE_KINDS:
            # Tightened to an allowlist: session_start, future protocol_event,
            # etc. are skipped so they don't get classified by accident.
            continue
        cid = row.get("claim_id")
        eng = row.get("engine")
        if not cid or not eng:
            continue
        by_claim_engine[(cid, eng)].append(i)

    per_claim: dict[str, list[Attribution]] = defaultdict(list)
    tool_output_by_tool: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    cells = summary.get("results", {}) if summary else {}

    for (cid, eng), idxs in by_claim_engine.items():
        cell = cells.get(f"{cid}|{eng}", {})
        fidelity = cell.get("fidelity") or {}
        repro = cell.get("reproducibility") or {}

        fid_pass = bool(fidelity.get("overall_pass"))
        repro_verdict = (repro.get("verdict") or "").lower()
        cell_engine_error = bool(fidelity.get("engine_error"))

        used_verification_tool = any(
            rows[i].get("tool_name") in _VERIFICATION_TOOLS
            for i in idxs
            if rows[i].get("kind") == "tool_call"
        )
        condition_omitted = _has_condition_omission(cell)

        for i in idxs:
            row = rows[i]
            kind = row.get("kind")
            err = row.get("error", "")

            if kind == "tool_call" and err:
                row["attribution"] = "tool_output"
                row["attribution_note"] = f"tool returned error: {err[:200]}"
                if row.get("tool_name"):
                    tool_output_by_tool[f"{cid}|{eng}"][row["tool_name"]] += 1
                continue

            if cell_engine_error:
                # Engine-level failure: route every otherwise-unclassified step
                # to the engine_error slot. The four rubric categories don't
                # apply to a transport-level failure; H4 also catches the cell.
                if row.get("attribution", "unclassified") == "unclassified":
                    row["attribution"] = "engine_error"
                    row["attribution_note"] = (
                        "cell-level engine_error; step preserved for context"
                    )
                continue

            if kind == "final_output" and err:
                row["attribution"] = "verification_skipped"
                row["attribution_note"] = (
                    f"engine ended with error and no structured output: {err[:200]}"
                )

        # Per-claim final_output rule (only when the engine succeeded
        # but fidelity FAILed; engine_error path already handled above).
        final_idxs = [i for i in idxs if rows[i].get("kind") == "final_output"]
        if not fid_pass and not cell_engine_error:
            label: Attribution = "tool_selection"
            note = "fidelity FAIL"
            if not used_verification_tool:
                label = "verification_skipped"
                note = "fidelity FAIL and no verification tool was called"
            elif condition_omitted:
                label = "condition_omission"
                note = "fidelity FAIL with load-bearing condition(s) missing"
            for i in final_idxs:
                rows[i]["attribution"] = label
                rows[i]["attribution_note"] = note

        # Successful chain tail rule.
        if fid_pass and repro_verdict in ("pass", "inconclusive"):
            for i in idxs:
                if rows[i].get("attribution", "unclassified") == "unclassified":
                    rows[i]["attribution"] = "ok"

        for i in idxs:
            per_claim[f"{cid}|{eng}"].append(rows[i].get("attribution", "unclassified"))

    return rows, per_claim, {k: dict(v) for k, v in tool_output_by_tool.items()}


def _has_condition_omission(cell: dict[str, Any]) -> bool:
    """True when at least one load-bearing claim condition is absent
    from the agent's `extracted_conditions`.

    Requires `cell["load_bearing_condition_names"]` (curator-supplied,
    populated by ChamberResultsCollector.record). When that field is
    missing or empty -- e.g. a claim with no load-bearing conditions --
    we fall back to the conservative "no extracted conditions at all"
    signal so the rubric still says something useful.
    """
    cr = cell.get("claim_result") or {}
    extracted_names_lower = {
        (c.get("name") or "").lower() for c in (cr.get("extracted_conditions") or [])
    }
    expected = [
        n.lower() for n in (cell.get("load_bearing_condition_names") or []) if n
    ]
    if expected:
        return any(name not in extracted_names_lower for name in expected)
    # Degenerate case: curator didn't mark any load-bearing conditions.
    # The strongest signal we have left is that the agent extracted
    # nothing structured at all.
    return not extracted_names_lower


# ---------------------------------------------------------------------------
# LLM-assisted pass (optional)
# ---------------------------------------------------------------------------


_LLM_SYSTEM_PROMPT = """You are classifying one step from a datasheet-extraction agent's trace
into the chamber-benchmark failure-attribution rubric.

Slots (return exactly one):

  tool_output           the tool itself returned wrong content or errored
  tool_selection        the agent picked the wrong tool, or the right tool
                        with wrong inputs (reasoning error)
  condition_omission    the agent extracted a number but dropped or
                        misread its operating conditions
  verification_skipped  the agent finalized without exercising any
                        cross-check (search_text / extract_table_markdown
                        / cross_sensor_check)
  engine_error          transport-level failure (timeout, retry exhausted,
                        gateway crash) that's not an agent reasoning error
  ok                    step is part of a successful chain

Manual deterministic rules already labelled the obvious cases. You
only see steps that fell through to "unclassified" and the cell-level
context for the claim. Pick the single slot whose rubric definition
best fits.

Return strictly valid JSON: {"label": "<slot>", "note": "<<=200 chars>"}.
"""


def _build_llm_user_prompt(row: dict[str, Any], cell: dict[str, Any]) -> str:
    """Compact context: cell fidelity + repro + the step itself."""
    fidelity = cell.get("fidelity") or {}
    repro = cell.get("reproducibility") or {}
    return (
        f"Claim: {cell.get('claim_id', '?')}  engine: {cell.get('engine', '?')}\n"
        f"Fidelity overall_pass: {fidelity.get('overall_pass')}  "
        f"reproducibility verdict: {repro.get('verdict')}\n"
        f"Step kind: {row.get('kind')}  tool: {row.get('tool_name', '')}\n"
        f"Step error: {row.get('error', '') or '(none)'}\n"
        f"Agent reasoning (truncated): {(row.get('agent_reasoning') or '')[:500]}\n"
        f"Tool input summary: {(row.get('tool_input_summary') or '')[:300]}\n"
        f"Tool output summary: {(row.get('tool_output_summary') or '')[:300]}\n"
    )


_VALID_LLM_LABELS: frozenset[str] = frozenset(
    {
        "tool_output",
        "tool_selection",
        "condition_omission",
        "verification_skipped",
        "engine_error",
        "ok",
    }
)


def _classify_llm(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    model: str = "claudesonnet4.6",
) -> list[dict[str, Any]]:
    """For every still-`unclassified` step, ask the model to choose a slot.

    The prompt is constrained to the rubric. Manual rules always win;
    the LLM never overrides an existing label other than `unclassified`.
    Anthropic SDK errors are non-fatal -- the step keeps its
    "unclassified" label rather than corrupting the rubric histogram.
    """
    try:
        from anthropic import (
            Anthropic,
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            RateLimitError,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("anthropic SDK not importable; skipping LLM-assist (%s)", exc)
        return rows

    cells = summary.get("results", {}) if summary else {}
    unresolved_idxs = [
        i
        for i, r in enumerate(rows)
        if r.get("attribution", "unclassified") == "unclassified"
        and r.get("kind") in ("tool_call", "final_output")
    ]
    if not unresolved_idxs:
        logger.info("LLM-assist: no unclassified steps; skipping")
        return rows

    # Set up the SDK env (TLS / base_url) the same way the engines do.
    from chamberbench.credentials import setup_credentials, tls_verify_disabled

    setup_credentials()

    import os

    kwargs: dict[str, Any] = {"api_key": os.environ["ANTHROPIC_API_KEY"]}
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    if tls_verify_disabled():
        import httpx

        kwargs["http_client"] = httpx.Client(verify=False)

    client = Anthropic(**kwargs)
    n_classified = 0

    try:
        for i in unresolved_idxs:
            row = rows[i]
            cid = row.get("claim_id", "")
            eng = row.get("engine", "")
            cell = cells.get(f"{cid}|{eng}", {})
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=512,
                    system=_LLM_SYSTEM_PROMPT,
                    messages=[
                        {"role": "user", "content": _build_llm_user_prompt(row, cell)},
                    ],
                )
            except (
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
                RateLimitError,
            ) as exc:
                logger.warning("LLM-assist transient failure on step %d: %s", i, exc)
                continue
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM-assist failed on step %d: %s", i, exc)
                continue

            text = "".join(getattr(b, "text", "") for b in response.content).strip()
            label, note = _parse_llm_response(text)
            if label is None:
                logger.warning(
                    "LLM-assist returned unparseable response on step %d: %r",
                    i,
                    text[:200],
                )
                continue
            row["attribution"] = label
            row["attribution_note"] = (note or "LLM-assist")[:200]
            n_classified += 1
    finally:
        try:
            client.close()
        except Exception:
            logger.debug("error closing classifier client", exc_info=True)

    logger.info(
        "LLM-assist classified %d / %d unresolved steps",
        n_classified,
        len(unresolved_idxs),
    )
    return rows


def _parse_llm_response(text: str) -> tuple[str | None, str]:
    """Parse the model's `{"label": ..., "note": ...}` response.

    Returns (label, note) where label is None if the response can't be
    parsed or doesn't match the rubric. Tolerates markdown fences.
    """
    s = text.strip()
    for marker in ("```json", "```"):
        idx = s.find(marker)
        if idx != -1:
            start = idx + len(marker)
            end = s.find("```", start)
            if end != -1:
                s = s[start:end].strip()
                break
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return None, ""
    label = obj.get("label")
    note = str(obj.get("note") or "")
    if label not in _VALID_LLM_LABELS:
        return None, note
    return label, note


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def classify_run(
    traces_path: Path,
    summary_path: Path,
    *,
    llm_assist: bool = False,
    model: str = "claudesonnet4.6",
) -> dict[str, Any]:
    """Apply the rubric in place. Idempotent.

    Returns a small dict with `per_claim` attribution histograms and a
    global `total` counter, suitable for embedding in the aggregate
    summary or feeding to quality_gates.
    """
    rows = _read_jsonl(traces_path)
    summary = _read_summary(summary_path) or {}

    # Reset every prior attribution to "unclassified" so re-runs are
    # idempotent against earlier classifier passes (manual rules always
    # win, but they only fire when the right preconditions hold).
    for r in rows:
        if r.get("kind") in ("tool_call", "final_output"):
            r["attribution"] = "unclassified"
            r["attribution_note"] = ""

    rows, per_claim, tool_output_by_tool = _classify_manual(rows, summary)
    if llm_assist:
        rows = _classify_llm(rows, summary, model=model)

    _write_jsonl(traces_path, rows)

    total: dict[str, int] = defaultdict(int)
    for labels in per_claim.values():
        for label in labels:
            total[label] += 1

    # Embed per-claim attribution into the summary cells for the gates
    # to consume. tool_output_by_tool gives quality_gates a precise
    # numerator for the per-tool error rate (S2) instead of the old
    # proportional-allocation hack.
    if summary and "results" in summary:
        for key, labels in per_claim.items():
            cell = summary["results"].get(key)
            if cell is None:
                continue
            cell["attribution"] = list(labels)
            counts: dict[str, int] = defaultdict(int)
            for label in labels:
                counts[label] += 1
            cell["attribution_counts"] = dict(counts)
            cell["tool_output_by_tool"] = tool_output_by_tool.get(key, {})
        # Stamp a top-level schema_version so future gate consumers can
        # detect frozen baselines that predate a schema bump.
        summary["schema_version"] = 1
        _write_json_atomic(summary_path, summary)

    return {
        "per_claim": dict(per_claim),
        "total": dict(total),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument(
        "--traces",
        default=None,  # resolved to archive_dir() below
        type=Path,
    )
    parser.add_argument(
        "--summary",
        default=None,  # resolved to archive_dir() below
        type=Path,
    )
    parser.add_argument("--llm-assist", action="store_true")
    parser.add_argument("--model", default="claudesonnet4.6")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    # Default to the shipped archive. The model-qualified names are the ones
    # that exist here; the originating project's unqualified `latest_traces.jsonl`
    # has no counterpart in this release.
    if args.traces is None:
        args.traces = archive_dir() / "latest_traces.claudesonnet4.6.jsonl"
    if args.summary is None:
        args.summary = archive_dir() / "latest_chamber.claudesonnet4.6.json"

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # Load .env and set the same TLS-disable flag the chamber_collector
    # fixture uses, so `chamber-classify --llm-assist` works against the
    # internal LiteLLM gateway without external configuration.
    if args.llm_assist:
        try:
            from dotenv import load_dotenv

            for env_path in (
                Path(".env"),
                BENCHMARK_ROOT / ".env",
            ):
                if env_path.exists():
                    load_dotenv(env_path)
                    break
        except Exception:
            logger.debug("dotenv load failed; relying on existing env", exc_info=True)
        import os as _os

        _os.environ.setdefault("DISABLE_TLS_VERIFY", "true")

    out = classify_run(
        traces_path=args.traces,
        summary_path=args.summary,
        llm_assist=args.llm_assist,
        model=args.model,
    )

    print("=" * 60)
    print("CHAMBER FAILURE-ATTRIBUTION CLASSIFIER")
    print("=" * 60)
    print(f"  traces:  {args.traces}")
    print(f"  summary: {args.summary}")
    print(f"  cells:   {len(out['per_claim'])}")
    print("  totals:")
    for label in (
        "ok",
        "tool_output",
        "tool_selection",
        "condition_omission",
        "verification_skipped",
        "engine_error",
        "unclassified",
    ):
        print(f"    {label:<22s} {out['total'].get(label, 0):>4d}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
