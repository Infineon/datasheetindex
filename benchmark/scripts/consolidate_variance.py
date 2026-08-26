"""Consolidate the chamber variance runs into the final variance_chamber.json.

The variance experiment (revision item 3) ran in three pieces because two
harness/infra issues surfaced mid-run:

  - GPT-5.1's 15-turn budget was a stale artifact (no real context
    constraint -- measured ~30k tokens at turn 30, far under its window).
    A fair cross-model comparison needs a uniform 30-turn budget, so the
    GPT-5.1 leg was re-run at max_turns=30 -> variance_gpt_rerun.json.
  - qwen3.6-27b's self-hosted vLLM pod dropped mid-run twice, returning
    HTTP 503 "all pods are down" for a batch of repeat-3 cells each time.
    The two attempts 503-failed on *disjoint* claim sets, so a clean,
    fully-503-free repeat 3 is rebuilt by patching the main run's
    repeat 3 with the retry's result (variance_qwen_r3.json) for exactly
    the cells the main run's backend outage killed. Every resulting cell
    is a genuine qwen agentic run -- this is a retry of infra-failed
    cells, not synthetic data.

Inputs (archive/):
  - variance_chamber.mainrun.bak.json -- the main 3-model run
  - variance_gpt_rerun.json           -- GPT-5.1 re-run at 30 turns
  - variance_qwen_r3.json             -- qwen repeat-3 retry

Output: variance_chamber.json -- Claude (3 repeats), GPT-5.1 (3 repeats
at 30 turns), qwen3.6-27b (3 repeats, repeat 3 503-free after patching).

Run:
    uv run python scripts/consolidate_variance.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from chamberbench.claimsio import archive_dir

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from chamberbench.variance import aggregate_variance

CHAMBER = archive_dir()
MAIN = CHAMBER / "variance_chamber.mainrun.bak.json"
GPT_RERUN = CHAMBER / "variance_gpt_rerun.json"
QWEN_RERUN = CHAMBER / "variance_qwen_r3.json"
OUT = CHAMBER / "variance_chamber.json"


def _is_503(cell: dict[str, Any]) -> bool:
    """True if the cell failed because the (qwen) backend pod was down."""
    e = cell.get("engine_error", "") or ""
    return "not serving requests" in e or "all pods are down" in e


def _live_repeat(data: dict[str, Any], model: str) -> dict[str, Any]:
    """The single live repeat from a `--fresh-repeats 1` side-file."""
    live = [r for r in data["runs"][model] if r["source"] == "live"]
    assert len(live) == 1, f"expected 1 live repeat for {model}, got {len(live)}"
    return live[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=OUT,
        help="Where to write the consolidated file (default: the shipped archive path).",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the shipped variance_chamber.json anyway.",
    )
    args = ap.parse_args()

    # The shipped archive/variance_chamber.json is a PRIMARY artifact, not this
    # script's output. It is dated 2026-06-05 and carries no `_consolidation`
    # key, while the inputs below are the superseded 2026-05-22 run: the Claude
    # and GPT-5.1 legs of the published results come from a later re-run that is
    # not in the archive. Running this script therefore does not reconstruct the
    # published file -- it REPLACES two of three model legs with older data, and
    # silently moves the paper's Table 1 (GPT-5.1 mean latency 236s -> 133s).
    # It is kept because it documents how the May consolidation was performed.
    if args.out.exists() and not args.force:
        print(f"REFUSING to overwrite {args.out}.")
        print(
            "  It is the published artifact and was NOT produced by this script "
            "from these inputs;"
        )
        print(
            "  regenerating would revert the Claude and GPT-5.1 legs to the "
            "superseded 2026-05-22 run."
        )
        print("  Pass --out <new-path> to write elsewhere, or --force to overwrite.")
        return 1

    main_data = json.loads(MAIN.read_text(encoding="utf-8"))
    gpt_data = json.loads(GPT_RERUN.read_text(encoding="utf-8"))
    qwen_data = json.loads(QWEN_RERUN.read_text(encoding="utf-8"))

    # Claude -- unchanged from the main run.
    claude = main_data["runs"]["claudesonnet4.6"]

    # GPT-5.1 -- the 30-turn re-run supersedes the main run's 15-turn leg.
    gpt = gpt_data["runs"]["gpt-5.1"]

    # qwen -- repeats 1-2 from the main run (clean: every error is qwen-side,
    # no 503). Repeat 3 = the main run's repeat 3 with its backend-outage
    # 503 cells patched from the retry.
    qwen_runs = main_data["runs"]["qwen3.6-27b"]
    qwen_r1 = next(r for r in qwen_runs if r["repeat"] == 1)
    qwen_r2 = next(r for r in qwen_runs if r["repeat"] == 2)
    qwen_r3_main = next(r for r in qwen_runs if r["repeat"] == 3)
    qwen_retry = _live_repeat(qwen_data, "qwen3.6-27b")["cells"]

    patched: dict[str, Any] = {}
    n_patched = 0
    for cid, cell in qwen_r3_main["cells"].items():
        if _is_503(cell):
            sub = qwen_retry[cid]
            assert not _is_503(sub), f"claim {cid} is 503 in both attempts"
            patched[cid] = sub
            n_patched += 1
        else:
            patched[cid] = cell
    assert not any(_is_503(c) for c in patched.values()), (
        "503 remains in patched repeat 3"
    )
    qwen_r3 = {
        "repeat": 3,
        "source": f"live; {n_patched} cells retried after a qwen backend outage",
        "started": qwen_r3_main["started"],
        "cells": patched,
    }

    runs = {
        "claudesonnet4.6": claude,
        "gpt-5.1": gpt,
        "qwen3.6-27b": [qwen_r1, qwen_r2, qwen_r3],
    }
    for model, reps in runs.items():
        assert len(reps) == 3, f"{model}: expected 3 repeats, got {len(reps)}"
        for rep in reps:
            n = len(rep["cells"])
            assert n == 25, f"{model} repeat {rep['repeat']}: {n} cells, expected 25"

    payload = {
        "schema_version": 1,
        "timestamp": datetime.now(UTC).isoformat(),
        "n_repeats": 3,
        "engine": "agentic",
        "models": ["claudesonnet4.6", "gpt-5.1", "qwen3.6-27b"],
        "claim_ids": main_data["claim_ids"],
        "runs": runs,
        "aggregate": aggregate_variance(runs),
        "_consolidation": {
            "built_by": "scripts/consolidate_variance.py",
            "claudesonnet4.6": "main run",
            "gpt-5.1": "re-run at max_turns=30 (variance_gpt_rerun.json)",
            "qwen3.6-27b": (
                f"repeats 1-2 from the main run; repeat 3 = main run's "
                f"repeat 3 with {n_patched} backend-outage 503 cells patched "
                f"from variance_qwen_r3.json"
            ),
        },
    }
    args.out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"wrote {args.out}")
    agg = payload["aggregate"]
    for model in payload["models"]:
        a = agg[model]
        f = a["fidelity"]
        std = "n/a" if f["std"] is None else f"{f['std']:.2f}"
        print(
            f"  {model:18s} fidelity per_run={f['per_run']}  "
            f"mean={f['mean']:.2f}  std={std}  "
            f"engine_errors={a['engine_errors']['per_run']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
