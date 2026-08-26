"""Chamber benchmark -- confidence behavior analysis.

Day 17 of the chamber paper plan was originally scoped as a traditional
calibration analysis (reliability diagram, ECE, Brier score) of the
agent's self-reported ``ClaimResult.confidence`` against the matrix
verdicts. That scope is *degenerate on this benchmark*: across the 95
``status=ok`` cells in the v2 frozen baseline, **all 95 cells PASS
fidelity** (Sonnet 50/50, gpt-5.1 25/25, qwen3.6-27b 20/20) and 0 cells
have a definitive ``fail`` reproducibility verdict (12 pass, 83
inconclusive). A reliability diagram becomes a flat line at y=1.0, ECE
collapses to ``mean(1.0 - confidence)``, and Brier becomes
``mean((conf - 1)^2)``. None of those measure calibration in the usual
sense -- they measure "is the agent confident enough given it's always
right", the inverse of what calibration is supposed to detect.

This is a paper-richer finding than a bad calibration curve would have
been. The methodology doc already records four known negative-class
cases (1 Day-11 si115x-linearity drop + 3 Day-15 ACS70331 curator
errors); all four were caught by the fidelity scorer surfacing
high-confidence-not-found verdicts during *curation*, before the matrix
proper ran. They live in commit history and the methodology doc, not in
``baseline_chamber.json``.

What this module produces instead:

  1. **Confidence-distribution per model** (figure: ``confidence_
     distribution.png``). Each model's confidence reads as a strip plot
     with quartile marks; reveals model-personality differences (Sonnet
     concentrates near 0.99, gpt-5.1 has the widest range, qwen3.6-27b
     occupies a narrow middle band).
  2. **Confidence vs. effort** (figure: ``confidence_vs_effort.png``).
     Per-cell scatter of confidence against latency (and a secondary
     axis for navigation tool count), coloured by model. Tests whether
     the agent spends more effort when it ends up less confident -- a
     self-awareness signal independent of calibration.
  3. **Brier-loss table** (stdout). Brier with explicit framing: on an
     all-positive benchmark this measures "distance from full
     confidence", not calibration. Useful as a model-comparison
     quantity (lower = more confident) but should not be reported as a
     calibration metric in the paper.

The "Day-17 = calibration analysis" plan line is therefore *executed
in spirit* (we look at confidence behavior) but *re-scoped in letter*
(no reliability diagram or ECE, because the data forbids it). The
methodology doc's "Open questions" entry on confidence calibration is
updated to reflect this: getting a real reliability diagram on this
benchmark would require either re-running the four documented
negative-class cases, or curating an explicitly-adversarial negative
claim set -- both named future work.

Run:
    uv run --group chamber python -m chamberbench.calibration
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Resolved through claimsio so that CHAMBERBENCH_ARCHIVE_DIR reaches these
# too. Counting `.parents[n]` from the module's own location -- what this
# did before the benchmark was extracted -- silently pointed one directory
# level above the repository once the file moved.
from chamberbench.claimsio import archive_dir

# Only used to shorten paths in printed output.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = archive_dir()
FIGURES_DIR = RESULTS_DIR / "figures"
BASELINE_PATH = RESULTS_DIR / "baseline_chamber.json"

# Mirrored from analysis.py so the two plots stay visually coherent.
MODEL_ORDER: list[str] = ["claudesonnet4.6", "gpt-5.1", "qwen3.6-27b"]
MODEL_COLOURS: dict[str, str] = {
    "claudesonnet4.6": "#cc7028",
    "gpt-5.1": "#10a37f",
    "qwen3.6-27b": "#615ced",
}
MODEL_LABELS: dict[str, str] = {
    "claudesonnet4.6": "Claude Sonnet 4.6",
    "gpt-5.1": "GPT-5.1",
    "qwen3.6-27b": "Qwen3.6-27B",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellRow:
    """One ``status=ok`` cell from the v2 baseline, flattened for analysis."""

    model: str
    engine: str
    claim_id: str
    confidence: float
    fidelity_pass: bool
    repro_verdict: str  # "pass" | "fail" | "inconclusive" | ""
    latency_s: float
    n_nav_tool_calls: int  # tool dispatches excluding submit_claim_result


# Finalization tools (both two-pass submit tools, plus the pre-two-pass name):
# real tool dispatches in the trace but not navigation work, so excluded from
# the "real work done" count.
_SUBMIT_TOOLS = frozenset(
    {"submit_extraction", "submit_chamber_outcome", "submit_claim_result"}
)


def _n_nav_tool_calls(cell: dict[str, Any]) -> int:
    """Tool dispatches excluding the finalization (submit) tools.

    Mirrors ``analysis._nav_tools_per_cell``: the submit tools are real tool
    dispatches in the trace but not *navigation* tools; the "real work done"
    number is the others.
    """
    counts = cell.get("n_tool_calls_by_tool") or {}
    return sum(v for tool, v in counts.items() if tool not in _SUBMIT_TOOLS)


def load_ok_cells(baseline_path: Path = BASELINE_PATH) -> list[CellRow]:
    """Flatten ``baseline_chamber.json`` (v2) to a list of OK cells.

    Cells with status != "ok" are skipped: ``not_applicable`` cells
    carry no real confidence (engine_error before any extraction),
    ``pending_rerun`` cells have no data at all. A status-"ok" cell that
    still carries an ``engine_error`` (e.g. a turn that ended without the
    submit call) is skipped too -- it produced no extraction to
    calibrate. Cells without a numeric confidence are also skipped
    (defensive; shouldn't happen on the current frozen baseline).
    """
    if not baseline_path.exists():
        raise FileNotFoundError(baseline_path)
    data = json.loads(baseline_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 2:
        raise ValueError(
            f"baseline_chamber.json schema_version is {data.get('schema_version')!r}, expected 2"
        )

    rows: list[CellRow] = []
    for cid, by_eng in (data.get("results") or {}).items():
        if not isinstance(by_eng, dict):
            continue
        for eng, by_model in by_eng.items():
            if not isinstance(by_model, dict):
                continue
            for model, cell in by_model.items():
                if (
                    not isinstance(cell, dict)
                    or cell.get("status") != "ok"
                    or cell.get("engine_error")
                ):
                    continue
                fid = cell.get("fidelity") or {}
                conf = fid.get("confidence")
                if not isinstance(conf, (int, float)):
                    continue
                rows.append(
                    CellRow(
                        model=model,
                        engine=eng,
                        claim_id=cid,
                        confidence=float(conf),
                        fidelity_pass=bool(fid.get("overall_pass")),
                        repro_verdict=(
                            (cell.get("reproducibility") or {}).get("verdict") or ""
                        ).lower(),
                        latency_s=float(cell.get("latency_s") or 0.0),
                        n_nav_tool_calls=_n_nav_tool_calls(cell),
                    )
                )
    return rows


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def brier_score(confidences: list[float], outcomes: list[bool]) -> float:
    """Mean squared error between confidence and binary outcome.

    On an all-positive class (every outcome True), this reduces to
    ``mean((c - 1)**2)``: a "distance from full confidence" measure,
    not a calibration metric. We report it anyway for cross-model
    comparison but caption it honestly.
    """
    if not confidences:
        return 0.0
    if len(confidences) != len(outcomes):
        raise ValueError(
            f"length mismatch: {len(confidences)} confidences vs {len(outcomes)} outcomes"
        )
    return sum(
        (c - (1.0 if o else 0.0)) ** 2
        for c, o in zip(confidences, outcomes, strict=False)
    ) / len(confidences)


def per_model_summary(rows: list[CellRow]) -> dict[str, dict[str, float]]:
    """Headline stats per model: n, mean / median / min / max confidence,
    Brier loss, mean latency, mean nav tool calls. Returned as a flat
    dict for stdout pretty-printing and for the commit-message table.
    """
    out: dict[str, dict[str, float]] = {}
    for model in MODEL_ORDER:
        sub = [r for r in rows if r.model == model]
        if not sub:
            continue
        confs = [r.confidence for r in sub]
        outs = [r.fidelity_pass for r in sub]
        out[model] = {
            "n": float(len(sub)),
            "conf_mean": statistics.fmean(confs),
            "conf_median": statistics.median(confs),
            "conf_min": min(confs),
            "conf_max": max(confs),
            "conf_stdev": statistics.stdev(confs) if len(confs) > 1 else 0.0,
            "brier": brier_score(confs, outs),
            "latency_mean": statistics.fmean(r.latency_s for r in sub),
            "nav_tools_mean": statistics.fmean(r.n_nav_tool_calls for r in sub),
        }
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def confidence_distribution_figure(rows: list[CellRow]) -> Path:
    """Strip plot of per-cell confidence by model, with quartile marks.

    Strip plot (jittered scatter) is chosen over violin because n is
    small (20-50 per model) and a smoothed density would over-claim.
    Quartile bars (Q1, median, Q3) drawn on top so the eye can compare
    distributions without reading individual dots.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    rng = np.random.default_rng(0)  # reproducible jitter
    for i, model in enumerate(MODEL_ORDER):
        confs = sorted(r.confidence for r in rows if r.model == model)
        if not confs:
            continue
        x = i + (rng.random(len(confs)) - 0.5) * 0.32
        ax.scatter(
            x,
            confs,
            s=22,
            alpha=0.55,
            color=MODEL_COLOURS[model],
            edgecolors="none",
            zorder=2,
        )
        q1, med, q3 = np.percentile(confs, [25, 50, 75])
        ax.hlines(
            [q1, med, q3],
            i - 0.30,
            i + 0.30,
            colors=[MODEL_COLOURS[model]] * 3,
            linewidths=[1.0, 2.5, 1.0],
            zorder=3,
        )
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels(
        [
            f"{MODEL_LABELS[m]}\n(n={sum(1 for r in rows if r.model == m)})"
            for m in MODEL_ORDER
        ],
        fontsize=9,
    )
    ax.set_ylabel("Agent self-reported confidence")
    ax.set_ylim(0.78, 1.005)
    ax.set_title(
        "Per-model confidence distribution (status=ok cells from frozen baseline)",
        fontsize=11,
    )
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    # Quartile-line legend
    ax.plot([], [], color="grey", linewidth=2.5, label="median")
    ax.plot([], [], color="grey", linewidth=1.0, label="Q1 / Q3")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    out = FIGURES_DIR / "confidence_distribution.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def confidence_vs_effort_figure(rows: list[CellRow]) -> Path:
    """Two-panel scatter: confidence vs. latency, confidence vs. nav tools.

    Tests "does the agent spend more effort when it ends up less
    confident?" -- a self-awareness signal independent of calibration.
    Each model is plotted in its accent colour; baseline and agentic
    engines distinguished by marker shape (baseline = square,
    agentic = circle).
    """
    fig, (ax_latency, ax_tools) = plt.subplots(1, 2, figsize=(10.0, 4.5))
    for ax, y_attr, y_label in (
        (ax_latency, "latency_s", "Cell latency (s)"),
        (ax_tools, "n_nav_tool_calls", "Navigation tool calls per cell"),
    ):
        for model in MODEL_ORDER:
            for eng, marker in (("agentic", "o"), ("baseline", "s")):
                sub = [r for r in rows if r.model == model and r.engine == eng]
                if not sub:
                    continue
                ax.scatter(
                    [r.confidence for r in sub],
                    [getattr(r, y_attr) for r in sub],
                    s=28,
                    alpha=0.55,
                    color=MODEL_COLOURS[model],
                    marker=marker,
                    edgecolors="none",
                    label=f"{MODEL_LABELS[model]} ({eng})"
                    if ax is ax_latency
                    else None,
                )
        ax.set_xlabel("Confidence")
        ax.set_ylabel(y_label)
        ax.grid(linestyle=":", alpha=0.4)
        ax.set_xlim(0.78, 1.005)
    ax_latency.legend(loc="upper left", fontsize=7, frameon=False)
    fig.suptitle(
        "Confidence vs. effort (per-cell, frozen baseline)",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIGURES_DIR / "confidence_vs_effort.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_summary(summary: dict[str, dict[str, float]]) -> None:
    print()
    print("Per-model confidence behavior (status=ok cells from frozen baseline)")
    print("=" * 78)
    header = (
        f"{'Model':<20s} {'n':>4s} {'mean':>6s} {'med':>6s} "
        f"{'min':>6s} {'max':>6s} {'sd':>6s} "
        f"{'Brier':>7s} {'lat_s':>7s} {'tools':>6s}"
    )
    print(header)
    print("-" * 78)
    for model, s in summary.items():
        print(
            f"{MODEL_LABELS[model]:<20s} "
            f"{int(s['n']):>4d} "
            f"{s['conf_mean']:>6.3f} {s['conf_median']:>6.3f} "
            f"{s['conf_min']:>6.2f} {s['conf_max']:>6.2f} "
            f"{s['conf_stdev']:>6.3f} "
            f"{s['brier']:>7.5f} "
            f"{s['latency_mean']:>7.1f} "
            f"{s['nav_tools_mean']:>6.2f}"
        )
    print()
    print("Brier loss caption: on this all-positive-class benchmark, Brier")
    print("reduces to mean((conf - 1)^2). It measures 'distance from full")
    print("confidence', not calibration. Lower = more confident; should NOT")
    print("be reported as a calibration metric in the paper.")
    print()


def main() -> int:
    rows = load_ok_cells()
    if not rows:
        print("no ok cells in baseline_chamber.json -- nothing to plot")
        return 1
    summary = per_model_summary(rows)
    _print_summary(summary)
    p1 = confidence_distribution_figure(rows)
    p2 = confidence_vs_effort_figure(rows)
    print(f"wrote {p1.relative_to(PROJECT_ROOT)}")
    print(f"wrote {p2.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
