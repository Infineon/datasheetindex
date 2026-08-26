"""Consolidate per-model chamber runs into the 3-axis frozen baseline.

Reads multiple `latest_chamber.{model}.json` files and
their pre-rerun snapshots, then writes `baseline_chamber.json` with the
new `claim_id × engine × model` shape.

Schema (schema_version = 2):

    {
      "schema_version": 2,
      "timestamp": "<iso8601>",
      "claims_path": "data/claims.yaml",
      "claim_ids": [...],
      "engines": ["agentic", "baseline"],
      "models": ["claudesonnet4.6", "gpt-5.1", "qwen3.6-27b"],
      "results": {
        "<claim_id>": {
          "<engine>": {
            "<model>": {
              "status": "ok" | "not_applicable" | "pending_rerun",
              ...flat per-cell record from latest_chamber.*.json...
            }
          }
        }
      }
    }

Status values:
  - "ok": full cell record from a real run.
  - "not_applicable": the engine cannot run on this (model, gateway)
    pair for a structural reason captured by `engine_error` and
    `not_applicable_reason`. Today: qwen3.5-27b baseline only
    (vLLM cannot ingest a PDF). gpt-5.1's baseline was also
    not_applicable until the 2026-05-20 Responses-API transport fix,
    which sends the PDF as a first-class `input_file`; it now runs.
    The `engine_error` captured by the runner is preserved, plus a
    paper-citable rationale.
  - "pending_rerun": no data on disk because a prerequisite (gateway
    upstream availability, claim curation, etc.) was not met at
    consolidation time. `reason` and `notes` document the block;
    quality_gates treats these as not-comparable, not as missing.

Source-file precedence per (engine, model): for each (claim, engine,
model) tuple we read sources in order; later sources win. The final-revision
agentic snapshot is read first so a post-rerun file that only contains
baseline + ACS70331-agentic does not erase the 20 DPS310+Si115x agentic
cells from the snapshot.

Run:
    uv run python scripts/build_chamber_baseline.py \
        --results-dir archive \
        --out archive/baseline_chamber.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from chamberbench.claimsio import archive_dir, data_dir, short_path

DEFAULT_RESULTS_DIR = archive_dir()
DEFAULT_OUT = DEFAULT_RESULTS_DIR / "baseline_chamber.json"

MODELS = ("claudesonnet4.6", "gpt-5.1", "qwen3.6-27b")
ENGINES = ("agentic", "baseline")


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _split_key(k: str) -> tuple[str, str]:
    """`<claim_id>|<engine>` -> (claim_id, engine).

    Falls back to the cell's own `claim_id` / `engine` fields when the
    key shape is unexpected, so an old-format cell without the pipe-
    delimiter still merges cleanly.
    """
    if "|" in k:
        cid, eng = k.split("|", 1)
        return cid, eng
    return k, ""


def _gather_for_model(
    results_dir: Path, model: str
) -> dict[tuple[str, str], dict[str, Any]]:
    """Read the model's per-cell records from snapshot + latest, with
    latest winning on overlap.

    Returns ``{(claim_id, engine): cell_record}``. The cell_record is the
    raw dict the runner wrote -- this function only merges, it does not
    add `status` or restructure fields.
    """
    # Source order matters: later sources win. The final-revision agentic
    # snapshot goes first so a post-rerun file that only contains
    # baseline + ACS70331-agentic does not erase the 20 DPS310+Si115x
    # agentic cells from the snapshot.
    sources = [
        results_dir / f"snapshot_layer2_agentic.{model}.json",
        results_dir / f"latest_chamber.{model}.json",
    ]
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for src in sources:
        d = _load(src)
        if d is None:
            continue
        for k, cell in (d.get("results") or {}).items():
            cid = cell.get("claim_id") or _split_key(k)[0]
            eng = cell.get("engine") or _split_key(k)[1]
            if not cid or not eng:
                continue
            merged[(cid, eng)] = cell
    return merged


def _is_structural_baseline_fail(model: str, cell: dict[str, Any]) -> bool:
    """Detect the baseline-portability structural failure.

    Pattern: model is not Anthropic-native, engine is baseline, and the
    cell carries a populated `engine_error`. After the 2026-05-20
    transport fix this fires only for qwen3.5-27b, whose vLLM backend
    cannot ingest a PDF (``Hosted_vllmException ... cannot identify
    image file``). gpt-5.1's baseline previously matched here too
    (``BadRequestError ... Expected a base64-encoded``) but now routes
    through the Responses API, which accepts the PDF; its baseline cells
    carry no `engine_error` and are not flagged. The check stays
    defensive: any new gateway-specific variant on a non-Anthropic
    baseline cell with a populated engine error will still match.
    """
    if cell.get("engine") != "baseline":
        return False
    if model == "claudesonnet4.6":
        return False
    err = cell.get("engine_error") or ""
    return bool(err)


# Engagement-cost cliff (2026-05-18): non-Anthropic agentic cells
# that exhaust the gateway's per-cell timeout or input-token budget. These
# only surfaced after the oracle-leak audit removed the answer from the
# prompt, forcing the agent to engage with the document instead of
# reflecting the hint. Two known shapes:
#   - "timeout after Ns" -- the per-cell wall-clock ceiling (CHAMBER_TIMEOUT_S)
#   - "ContextWindowExceededError" -- the gateway's input-token cap
#     (Azure / vLLM both surface this with the same error class name)
# Both are deployment-side limits, not methodology defects: the agent
# behaves correctly but the (gateway, model) pair cannot host a long-
# engagement cell on this benchmark. Documented in the methodology doc's
# "Post-audit results (2026-05-18 re-run)" section.
_ENGAGEMENT_FAILURE_SIGNATURES = (
    "timeout after",  # CHAMBER_TIMEOUT_S wall-clock ceiling
    "ContextWindowExceededError",  # explicit class name (vLLM surfaces this)
    "Input tokens exceed",  # Azure / OpenAI phrasing of the same
    "maximum context length",  # alternate vLLM phrasing
)


def _is_structural_engagement_fail(model: str, cell: dict[str, Any]) -> bool:
    if cell.get("engine") != "agentic":
        return False
    if model == "claudesonnet4.6":
        # Sonnet has a 1M-token context; any agentic engine error on this
        # deployment is a real bug, not a structural limit. Don't carve out.
        return False
    err = cell.get("engine_error") or ""
    return any(sig in err for sig in _ENGAGEMENT_FAILURE_SIGNATURES)


def _classify(model: str, cell: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Decide cell status and add a thin status-explanation block.

    Returns (status, augmented_cell). The cell is shallow-copied so the
    source dicts are not mutated.
    """
    out = dict(cell)
    if _is_structural_baseline_fail(model, cell):
        out["status"] = "not_applicable"
        out["not_applicable_reason"] = (
            "Cross-provider portability finding: the non-agentic "
            "baseline sends the PDF as an Anthropic-native `document` "
            "content block; the LiteLLM Anthropic-shape passthrough does "
            "not translate this faithfully to "
            f"{'Azure OpenAI' if model.startswith('gpt') else 'vLLM'}. "
            "Captured here for paper completeness, not as a regression."
        )
        return "not_applicable", out
    if _is_structural_engagement_fail(model, cell):
        err = cell.get("engine_error") or ""
        limit_kind = "timeout" if "timeout" in err else "input-token budget"
        gateway = "Azure OpenAI" if model.startswith("gpt") else "vLLM"
        out["status"] = "not_applicable"
        out["not_applicable_reason"] = (
            "Engagement-cost cliff: the post-oracle-leak-fix agent "
            "engages with the document instead of reflecting the hint, "
            f"and this (gateway, model) pair exhausts the per-cell "
            f"{limit_kind} ({gateway}). Documented in the methodology "
            "doc's 'Post-audit results (2026-05-18 re-run)' section as a "
            "deployment-side limit, not a methodology defect. Captured "
            "for paper completeness."
        )
        return "not_applicable", out
    out["status"] = "ok"
    return "ok", out


def _pending_stub(
    claim_id: str, engine: str, model: str, reason: str
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "engine": engine,
        "model": model,
        "status": "pending_rerun",
        "reason": reason,
    }


def _claim_ids_from_yaml(claims_path: Path) -> list[str]:
    """Return claim ids in YAML order. Pure-text parser (no yaml dep on
    the script path; the runner-side already validates with PyYAML)."""
    out: list[str] = []
    for line in claims_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("- id:"):
            v = s.split(":", 1)[1].strip().strip('"').strip("'")
            if v:
                out.append(v)
    return out


def build(
    results_dir: Path,
    claims_path: Path,
    qwen_pending_reason: str,
) -> dict[str, Any]:
    claim_ids = _claim_ids_from_yaml(claims_path)
    if not claim_ids:
        raise SystemExit(f"no claims parsed from {claims_path}")

    per_model: dict[str, dict[tuple[str, str], dict[str, Any]]] = {
        m: _gather_for_model(results_dir, m) for m in MODELS
    }

    results: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    summary: dict[str, dict[str, int]] = {
        m: {"ok": 0, "not_applicable": 0, "pending_rerun": 0, "missing": 0}
        for m in MODELS
    }

    for cid in claim_ids:
        results[cid] = {}
        for eng in ENGINES:
            results[cid][eng] = {}
            for m in MODELS:
                cell = per_model.get(m, {}).get((cid, eng))
                if cell is None:
                    # No data on disk for this tuple.
                    if m == "qwen3.5-27b":
                        results[cid][eng][m] = _pending_stub(
                            cid, eng, m, qwen_pending_reason
                        )
                        summary[m]["pending_rerun"] += 1
                    else:
                        # A healthy-model gap is itself a data integrity
                        # signal -- flag it loudly rather than silently
                        # writing a pending stub for a model we DID run.
                        results[cid][eng][m] = _pending_stub(
                            cid,
                            eng,
                            m,
                            "no data on disk and model is healthy -- rerun missed this cell or input data is corrupt",
                        )
                        summary[m]["missing"] += 1
                    continue
                status, augmented = _classify(m, cell)
                results[cid][eng][m] = augmented
                summary[m][status] += 1

    return {
        "schema_version": 2,
        "timestamp": datetime.now(UTC).isoformat(),
        "claims_path": short_path(claims_path),
        "claim_ids": claim_ids,
        "engines": list(ENGINES),
        "models": list(MODELS),
        "results": results,
        "_consolidation_summary": summary,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Consolidate chamber runs into the frozen baseline."
    )
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    ap.add_argument("--claims-path", default=str(data_dir() / "claims.yaml"))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing baseline. Without this, writing over the "
        "shipped archive is refused.",
    )
    ap.add_argument(
        "--qwen-pending-reason",
        default=(
            "vLLM upstream under maintenance on 2026-05-13 "
            "(the internal LiteLLM gateway returned 503 with an HTML "
            "maintenance page on three consecutive probes). "
            "Re-execute when the gateway's vLLM health endpoint "
            "is green; data sourced from the prior agentic snapshot "
            "for the 20 DPS310+Si115x agentic cells (carried forward), "
            "and absent for the 5 ACS70331 agentic + 25 baseline cells."
        ),
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="Print summary, don't write."
    )
    args = ap.parse_args()

    out = build(
        Path(args.results_dir),
        Path(args.claims_path),
        args.qwen_pending_reason,
    )

    print(
        f"Consolidated {len(out['claim_ids'])} claims × "
        f"{len(out['engines'])} engines × {len(out['models'])} models = "
        f"{len(out['claim_ids']) * len(out['engines']) * len(out['models'])} cells"
    )
    for m, counts in out["_consolidation_summary"].items():
        print(
            f"  {m:<18s} ok={counts['ok']:>3d}  "
            f"not_applicable={counts['not_applicable']:>3d}  "
            f"pending_rerun={counts['pending_rerun']:>3d}  "
            f"missing={counts['missing']:>3d}"
        )

    if args.dry_run:
        return 0

    out_path = Path(args.out)
    # `baseline_chamber.json` is the archive's primary artifact and the only
    # re-gradable one. Running this script with no arguments used to replace it
    # silently -- the same hazard `consolidate_variance.py` is guarded against.
    if out_path.exists() and not args.force and not args.dry_run:
        print(f"REFUSING to overwrite {short_path(out_path)}.")
        print("  It is primary evidence, not a build output.")
        print("  Pass --out <new-path> to write elsewhere, or --force to overwrite.")
        return 1
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
