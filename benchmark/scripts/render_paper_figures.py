"""Regenerate the data-bearing paper figures from the post-audit baseline.

The figures emitted by the project's analysis.py were found to be stale
(cost_latency_scatter still showed a pre-audit 60-cell matrix). This
script rebuilds the three data-bearing figures directly from
archive/baseline_chamber.json so that, like the numbers in
scripts/render_paper_tables.py, they come from the canonical post-audit
source. Run:

    uv run python scripts/render_paper_figures.py

Confidence figures (confidence_distribution, confidence_vs_effort) are
produced post-audit by calibration.py and are not rebuilt here.
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from chamberbench.claimsio import archive_dir

BASELINE = archive_dir() / "baseline_chamber.json"
# Cost/latency figure carries per-model error bars from the repeated runs;
# the tool-dispatch and fidelity heatmaps stay single-reference (baseline).
# Env override so the figure can be rebuilt from a snapshot while a live
# variance run is rewriting the canonical artifact.
VARIANCE = Path(
    os.environ.get(
        "VARIANCE_JSON",
        str(archive_dir() / "variance_chamber.json"),
    )
)
# Figures land in the benchmark's own `figures/` by default. The paper build
# points this at its LaTeX tree instead, which is why it is an override rather
# than a constant: the same script has to serve a reproduction that has no
# paper checkout and a paper build that does.
FIGDIR = Path(os.environ.get("CHAMBERBENCH_FIGURE_DIR", PROJECT_ROOT / "figures"))
FIGDIR.mkdir(parents=True, exist_ok=True)

# (input $/M, output $/M, cache_read $/M) -- matches scripts/chamber_cost_summary.py.
PRICING = {
    "claudesonnet4.6": (3.00, 15.00, 0.30),
    "gpt-5.1": (1.25, 10.00, 0.125),
    "qwen3.6-27b": (0.20, 0.60, 0.20),
}
MODELS = ["claudesonnet4.6", "gpt-5.1", "qwen3.6-27b"]
DISPLAY = {
    "claudesonnet4.6": "Claude Sonnet 4.6",
    "gpt-5.1": "GPT-5.1",
    "qwen3.6-27b": "Qwen3.6-27B",
}
COLOR = {
    "claudesonnet4.6": "#e0883c",
    "gpt-5.1": "#4fa784",
    "qwen3.6-27b": "#7b73d1",
}
DOC_TOOLS = [
    "build_datasheet",
    "get_section_text",
    "search_text",
    "extract_table_markdown",
    "inspect_page",
]
CHAMBER_TOOLS = [
    "list_experiments",
    "get_experiment_metadata",
    "query_dataset",
    "cross_sensor_check",
    "get_ground_truth_graph",
]
NAV_TOOLS = DOC_TOOLS + CHAMBER_TOOLS


def cost_dollars(usage: dict, model: str) -> float:
    in_m, out_m, cr_m = PRICING[model]
    return (
        usage.get("input_tokens", 0) * in_m
        + usage.get("output_tokens", 0) * out_m
        + usage.get("cache_read_tokens", 0) * cr_m
    ) / 1e6


def agentic_cells(results: dict, claim_ids: list, model: str, ok_only: bool = True):
    out = []
    for cid in claim_ids:
        c = results.get(cid, {}).get("agentic", {}).get(model)
        if c and (not ok_only or c["status"] == "ok"):
            out.append(c)
    return out


def fig_tool_dispatch(results, claim_ids):
    """Heatmap: mean calls per cell, per navigation tool, per model."""
    grid = []
    for tool in NAV_TOOLS:
        row = []
        for model in MODELS:
            cells = agentic_cells(results, claim_ids, model)
            calls = [(c.get("n_tool_calls_by_tool") or {}).get(tool, 0) for c in cells]
            row.append(statistics.mean(calls) if calls else 0.0)
        grid.append(row)

    fig, ax = plt.subplots(figsize=(6.0, 6.4))
    im = ax.imshow(grid, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([DISPLAY[m] for m in MODELS])
    ax.set_yticks(range(len(NAV_TOOLS)))
    ax.set_yticklabels(NAV_TOOLS)
    for i in range(len(NAV_TOOLS)):
        for j in range(len(MODELS)):
            val = grid[i][j]
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color="white" if val < (max(max(r) for r in grid) * 0.6) else "black",
                fontsize=8,
            )
    ax.set_title(
        "Per-tool dispatch by model (agentic, post-audit;\nsubmit tools excluded)"
    )
    fig.colorbar(im, ax=ax, label="mean calls per cell")
    fig.tight_layout()
    out = FIGDIR / "tool_dispatch_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_cost_latency(variance):
    """Scatter: per-cell cost vs latency (agentic), per-model means with
    +/- std error bars over the repeated runs.

    Dots pool the non-engine-error cells across every repeat. The per-model
    mean marker is the mean of the per-run means; the error bars are the
    std of those per-run means (so they match Table 1's latency +/- std).
    """
    runs = variance["runs"]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for model in MODELS:
        repeats = runs.get(model, [])
        xs: list[float] = []
        ys: list[float] = []
        per_run_cost: list[float] = []
        per_run_lat: list[float] = []
        for rep in repeats:
            cells = [c for c in rep["cells"].values() if not c.get("engine_error")]
            if not cells:
                continue
            rc = [cost_dollars(c["usage"], model) for c in cells]
            rl = [c["latency_s"] for c in cells]
            xs += rc
            ys += rl
            per_run_cost.append(statistics.mean(rc))
            per_run_lat.append(statistics.mean(rl))
        if not xs:
            continue
        ax.scatter(
            xs,
            ys,
            c=COLOR[model],
            label=DISPLAY[model],
            alpha=0.6,
            edgecolors="none",
            s=40,
        )
        mx = statistics.mean(per_run_cost)
        my = statistics.mean(per_run_lat)
        xerr = statistics.stdev(per_run_cost) if len(per_run_cost) >= 2 else 0.0
        yerr = statistics.stdev(per_run_lat) if len(per_run_lat) >= 2 else 0.0
        ax.errorbar(
            mx,
            my,
            xerr=xerr,
            yerr=yerr,
            fmt="X",
            c=COLOR[model],
            markersize=15,
            markeredgecolor="black",
            markeredgewidth=1.5,
            ecolor="black",
            elinewidth=1.2,
            capsize=4,
            zorder=5,
        )
        ax.annotate(
            f"mean: ${mx:.3f}, {my:.0f}s",
            (mx, my),
            textcoords="offset points",
            xytext=(10, 6),
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_xlabel("cost per cell (USD, public-list pricing; log scale)")
    ax.set_ylabel("latency per cell (s)")
    ax.set_title(
        "Cost vs. latency per cell (agentic; per-model mean "
        r"$\pm$ std over repeated runs)"
    )
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = FIGDIR / "cost_latency_scatter.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_fidelity(results, claim_ids):
    """Heatmap: per-claim agentic fidelity by model; confidence shown in cell."""
    fig, ax = plt.subplots(figsize=(6.4, 8.4))
    n_rows, n_cols = len(claim_ids), len(MODELS)
    for i, cid in enumerate(claim_ids):
        for j, model in enumerate(MODELS):
            c = results.get(cid, {}).get("agentic", {}).get(model)
            if not c or c.get("engine_error") or c["status"] != "ok":
                color, label = "#bdbdbd", "err"
            elif c["fidelity"]["overall_pass"]:
                color = "#4f9d5d"
                label = f"{c['claim_result']['extracted']['confidence']:.2f}"
            else:
                color = "#c0612f"
                label = f"{c['claim_result']['extracted']['confidence']:.2f}"
            ax.add_patch(
                Rectangle((j, n_rows - 1 - i), 1, 1, facecolor=color, edgecolor="white")
            )
            ax.text(
                j + 0.5,
                n_rows - 1 - i + 0.5,
                label,
                ha="center",
                va="center",
                color="white",
                fontsize=7,
            )
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks([j + 0.5 for j in range(n_cols)])
    ax.set_xticklabels([DISPLAY[m] for m in MODELS])
    ax.set_yticks([n_rows - 1 - i + 0.5 for i in range(n_rows)])
    ax.set_yticklabels(claim_ids, fontsize=7)
    ax.set_title("Cross-model agentic fidelity (post-audit)")
    ax.tick_params(length=0)
    fig.tight_layout()
    out = FIGDIR / "fidelity_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_reproducibility_perturbation(sweep: dict) -> Path:
    """Verdict vs document-vs-reality divergence, with the inconclusive band
    (combined uncertainty) shaded and fidelity flat at pass."""
    rows = sweep["rows"]
    combined = sweep["measured_sigma"]
    vmap = {"pass": 0, "inconclusive": 1, "fail": 2}
    cmap = {"pass": "#3aaa6b", "inconclusive": "#e8b14f", "fail": "#c0392b"}
    xs = [r["divergence"] for r in rows]
    ys = [vmap[r["verdict"]] for r in rows]
    colors = [cmap[r["verdict"]] for r in rows]

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.axvspan(
        0.0,
        combined,
        color="#e8b14f",
        alpha=0.18,
        label="combined uncertainty (" + format(combined, ".3f") + " hPa)",
    )
    ax.axvline(0.0, color="grey", ls="--", lw=0.8, alpha=0.7)
    ax.scatter(xs, ys, c=colors, s=45, zorder=5, edgecolors="white", linewidths=0.6)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["pass", "inconclusive", "fail"])
    ax.set_ylim(-0.5, 2.5)
    ax.set_xlabel("divergence: measured pressure - claimed upper bound (hPa)")
    ax.set_title(
        "Reproducibility verdict under controlled perturbation\n"
        "(DPS310 operating-range claim; fidelity = pass at every point)"
    )
    ax.text(
        0.02,
        0.92,
        "fidelity: pass (all points)",
        transform=ax.transAxes,
        fontsize=8,
        color="#3aaa6b",
        fontweight="bold",
    )
    ax.text(
        -0.085,
        0.55,
        "measurement below bound\n(inside stated range)",
        fontsize=7.5,
        color="#555555",
        ha="center",
        va="bottom",
    )
    ax.text(
        0.165,
        1.45,
        "measurement above bound\n(outside stated range)",
        fontsize=7.5,
        color="#555555",
        ha="center",
        va="top",
    )
    ax.grid(True, axis="x", ls="--", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = FIGDIR / "reproducibility_perturbation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    d = json.loads(BASELINE.read_text(encoding="utf-8"))
    results, claim_ids = d["results"], d["claim_ids"]
    variance = json.loads(VARIANCE.read_text(encoding="utf-8"))
    print("Rebuilding figures:")
    print(
        f"  tool dispatch + fidelity: {BASELINE.name} (timestamp {d.get('timestamp')})"
    )
    print(
        f"  cost / latency:           {VARIANCE.name} (timestamp {variance.get('timestamp')})"
    )
    for out in (
        fig_tool_dispatch(results, claim_ids),
        fig_cost_latency(variance),
        fig_fidelity(results, claim_ids),
    ):
        print("  wrote", out.relative_to(PROJECT_ROOT))
    sweep = json.loads(
        (archive_dir() / "perturbation_sweep.json").read_text(encoding="utf-8")
    )
    print("  perturbation:             " + str(fig_reproducibility_perturbation(sweep)))


if __name__ == "__main__":
    main()
