"""Silent-failure detector false-positive scan across models and repeats.

Extends the false-positive control of the fault-injection experiment
(``scripts/fault_injection.py``, Arm B) from the single
post-audit baseline to the full three-repeat variance suite, so the
detector's zero-false-positive property is measured across all three model
families and all repeated runs rather than one run.

The detector (``chamberbench.silent_failure``) reads only
the emergent per-tool dispatch record, never the extracted value, so this
script runs no LLM. It scores every fidelity-passing, non-engine-error
agentic cell -- the population in which a false positive is even possible
-- and reports the flag rate per model and per source.

A flagged clean cell is either a false positive (bad for the detector) or
a genuine silent failure hiding in the real runs (interesting); the script
prints every flagged cell so the distinction can be adjudicated by hand.

The two sources are reported separately and NEVER pooled. The variance
suite already contains the post-audit baseline as its repeat 1, so adding
the two totals double-counts that run -- this script printed the resulting
"0 of 280" as a ready-to-paste paper line long after the paper had
corrected it to 207, and a regeneration would have restored it.

Run:
    uv run python scripts/silent_failure_fp_scan.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from chamberbench.silent_failure import detect_silent_failure

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from chamberbench.claimsio import archive_dir

RESULTS_DIR = archive_dir()
BASELINE_PATH = RESULTS_DIR / "baseline_chamber.json"
VARIANCE_PATH = RESULTS_DIR / "variance_chamber.json"


def _iter_baseline_cells(
    doc: dict[str, Any],
) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Yield (model, claim_id, cell) for every agentic cell in the baseline run."""
    for claim_id, by_engine in (doc.get("results") or {}).items():
        for model, cell in (by_engine.get("agentic") or {}).items():
            if isinstance(cell, dict):
                yield model, claim_id, cell


def _iter_variance_cells(
    doc: dict[str, Any],
) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Yield (model, claim_id, cell) for every agentic cell across all repeats."""
    for model, repeats in (doc.get("runs") or {}).items():
        for run in repeats or []:
            for claim_id, cell in (run.get("cells") or {}).items():
                if isinstance(cell, dict):
                    yield model, claim_id, cell


def _engaged(cell: dict[str, Any]) -> bool:
    """True for the cells the detector can even consider: fidelity-pass, no engine error."""
    fid = cell.get("fidelity") or {}
    return (
        bool(fid.get("overall_pass"))
        and not fid.get("engine_error")
        and not cell.get("engine_error")
    )


def _scan(
    cells: Iterator[tuple[str, str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Score the detector over a stream of cells, broken down by model."""
    by_model: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"evaluated": 0, "flagged": 0, "flagged_cells": []}
    )
    for model, claim_id, cell in cells:
        if not _engaged(cell):
            continue
        report = detect_silent_failure(cell)
        by_model[model]["evaluated"] += 1
        if report.flagged:
            by_model[model]["flagged"] += 1
            by_model[model]["flagged_cells"].append(
                {"claim_id": claim_id, "rules": report.rules}
            )
    return dict(by_model)


def _totals(by_model: dict[str, dict[str, Any]]) -> tuple[int, int]:
    evaluated = sum(m["evaluated"] for m in by_model.values())
    flagged = sum(m["flagged"] for m in by_model.values())
    return evaluated, flagged


def _report_source(name: str, by_model: dict[str, dict[str, Any]]) -> None:
    evaluated, flagged = _totals(by_model)
    print()
    print("-" * 72)
    rate = (flagged / evaluated) if evaluated else 0.0
    print(f"{name}: {flagged}/{evaluated} flagged  (fp/flag rate {rate:.1%})")
    for model in sorted(by_model):
        m = by_model[model]
        print(f"    {model:<20s} {m['flagged']}/{m['evaluated']} flagged")
        for fc in m["flagged_cells"]:
            print(f"        FLAGGED {fc['claim_id']}  rules={fc['rules']}")


def main() -> int:
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    variance = json.loads(VARIANCE_PATH.read_text(encoding="utf-8"))

    base_by_model = _scan(_iter_baseline_cells(baseline))
    var_by_model = _scan(_iter_variance_cells(variance))

    print("=" * 72)
    print("SILENT-FAILURE DETECTOR -- FALSE-POSITIVE SCAN")
    print("=" * 72)
    _report_source("post-audit baseline (single run)", base_by_model)
    _report_source("variance suite (3 repeats x 3 models)", var_by_model)

    base_eval, _base_flag = _totals(base_by_model)
    var_eval, var_flag = _totals(var_by_model)
    # Name the models that actually contributed to the reported denominator.
    # The union with the baseline would describe a model family the variance
    # suite lacks -- and this line is written to be pasted into the paper.
    models = sorted(var_by_model)

    print()
    print("=" * 72)
    print(
        f"DISTINCT POPULATION: {var_flag}/{var_eval} flagged across {len(models)} models ({', '.join(models)})"
    )
    print(
        f"  paper line: detector flags {var_flag} of {var_eval} fidelity-passing "
        f"agentic cells\n  across all {len(models)} model families and three repeats."
    )
    print()
    print(
        f"  NOT {base_eval} + {var_eval} = {base_eval + var_eval}. The variance suite's repeat 1 IS\n"
        "  the post-audit baseline -- chamberbench.variance.import_repeat_one\n"
        "  copies those cells in, and every repeat-1 record carries source ==\n"
        '  "imported:baseline_chamber.json". Verified cell by cell: fidelity\n'
        "  verdict, n_tool_calls_by_tool and latency_s are identical for all 25\n"
        "  cells in all three models. Pooling the two files counts that run twice,\n"
        "  and it is the error behind the retracted 280. The baseline is reported\n"
        "  separately above because it is a strict subset, never as an addend."
    )
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
