"""Chamber benchmark analysis -- aggregate cells and produce paper figures.

Reads:
  - ``archive/latest_chamber.{model}.json`` for each tracked
    model. Cells filtered to the canonical 20 (DPS310 + Si115x); the
    5 ACS70331 engagement-diagnostic cells are excluded by default
    because their reproducibility verdict is inconclusive by design
    (see ``protocols/engagement_diagnostic.py``).
  - ``archive/latest_traces.{model}.jsonl`` for per-tool
    dispatch counts when the result file's per-cell ``n_tool_calls_
    by_tool`` is not granular enough.

Writes ``archive/figures/`` containing the four paper
figures for the chamber paper (docs/reproducing.md):

  1. fidelity_heatmap.png             -- 20 claims x N models, agentic
  2. tool_dispatch_heatmap.png        -- tool name x model, mean calls/cell
  3. cost_latency_scatter.png         -- one point per cell, coloured by model
  4. engagement_over_revisions.png    -- per-model mean nav_tools/cell across
                                         revision 1, revision 2, revision 3

The four-plot bundle is the paper's quantitative spine. Each figure
caption (rendered in the paper, not here) names the takeaway in one
sentence.

Run:
    uv run --group chamber python -m chamberbench.analysis
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Resolved through claimsio so that CHAMBERBENCH_ARCHIVE_DIR reaches these
# too. Counting `.parents[n]` from the module's own location -- what this
# did before the benchmark was extracted -- silently pointed one directory
# level above the repository once the file moved.
from chamberbench.claimsio import archive_dir, short_path

# Only used to shorten paths in printed output.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = archive_dir()
FIGURES_DIR = RESULTS_DIR / "figures"

# Public-list pricing per 1M tokens (input, output). Mirrors
# ``scripts/chamber_cost_summary.py`` so the two artefacts agree.
PRICING: dict[str, tuple[float, float]] = {
    "claudesonnet4.6": (3.00, 15.00),
    "gpt-5.1": (1.25, 10.00),
    "qwen3.6-27b": (0.20, 0.60),
}

RESULT_FILES: dict[str, str] = {
    "claudesonnet4.6": "latest_chamber.claudesonnet4.6.json",
    "gpt-5.1": "latest_chamber.gpt-5.1.json",
    "qwen3.6-27b": "latest_chamber.qwen3.6-27b.json",
}

# Models in canonical plot order (left-to-right, top-to-bottom).
MODEL_ORDER: list[str] = ["claudesonnet4.6", "gpt-5.1", "qwen3.6-27b"]
MODEL_COLOURS: dict[str, str] = {
    "claudesonnet4.6": "#cc7028",  # Anthropic accent
    "gpt-5.1": "#10a37f",  # OpenAI accent
    "qwen3.6-27b": "#615ced",  # Qwen accent
}
MODEL_LABELS: dict[str, str] = {
    "claudesonnet4.6": "Claude Sonnet 4.6",
    "gpt-5.1": "GPT-5.1",
    "qwen3.6-27b": "Qwen3.6-27B",
}

# Claims in the headline matrix (DPS310 + Si115x). ACS70331 cells are
# engagement-diagnostic-only and reproducibility-inconclusive by
# design, so they would distort agreement / verdict plots; they get
# their own subset chart elsewhere if a follow-up wants it.
HEADLINE_COMPONENTS = ("dps310-", "si115x-")


def _load_cells(model: str) -> dict[str, dict[str, Any]]:
    """Return the agentic cells for ``model`` keyed by claim_id.

    Filters to the headline DPS310+Si115x set; drops engine_error
    cells (structural baseline failures keep their original
    `engine_error` field). Each value is the raw cell dict.
    """
    path = RESULTS_DIR / RESULT_FILES[model]
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    out: dict[str, dict[str, Any]] = {}
    for key, rec in data["results"].items():
        if not key.endswith("|agentic"):
            continue
        cid = rec.get("claim_id", "")
        if not any(cid.startswith(p) for p in HEADLINE_COMPONENTS):
            continue
        if rec.get("engine_error"):
            continue
        out[cid] = rec
    return out


def _cell_cost(rec: dict[str, Any], model: str) -> float:
    """Dollar cost for one cell, public-list pricing per the table."""
    usage = rec.get("usage") or {}
    in_per_m, out_per_m = PRICING[model]
    return (
        usage.get("input_tokens", 0) * in_per_m
        + usage.get("output_tokens", 0) * out_per_m
    ) / 1e6


# Finalization tools (both two-pass submit tools, plus the pre-two-pass name):
# real dispatches in the trace but not navigation work.
_SUBMIT_TOOLS = frozenset(
    {"submit_extraction", "submit_chamber_outcome", "submit_claim_result"}
)


def _nav_tools_per_cell(rec: dict[str, Any]) -> int:
    """Tool calls excluding the finalization (submit) tools.

    The submit tools are real tool dispatches in the trace but not
    *navigation* tools, so excluding them gives the "real work done" number.
    """
    tc = rec.get("n_tool_calls_by_tool") or {}
    return sum(v for tool, v in tc.items() if tool not in _SUBMIT_TOOLS)


def _claim_order() -> list[str]:
    """Stable claim order across all plots: DPS310 first, then Si115x."""
    seen: list[str] = []
    for model in MODEL_ORDER:
        for cid in _load_cells(model):
            if cid not in seen:
                seen.append(cid)
    seen.sort(key=lambda c: (0 if c.startswith("dps310-") else 1, c))
    return seen


# ---------------------------------------------------------------------------
# Figure 1: cross-model fidelity heatmap
# ---------------------------------------------------------------------------


def fidelity_heatmap() -> Path:
    """Claim x model grid coloured by ``fidelity.overall_pass``.

    Green = pass, amber = pass-with-low-confidence (<0.85), red = fail.
    Empty cells (missing data for a model) shown grey.
    """
    claims = _claim_order()
    models = MODEL_ORDER
    grid = np.full((len(claims), len(models)), np.nan)
    text_grid: list[list[str]] = []
    for i, cid in enumerate(claims):
        row: list[str] = []
        for j, model in enumerate(models):
            cells = _load_cells(model)
            rec = cells.get(cid)
            if rec is None:
                row.append("—")
                continue
            fid = rec.get("fidelity", {}) or {}
            conf = fid.get("confidence", 0.0)
            if fid.get("overall_pass"):
                grid[i, j] = 1.0 if conf >= 0.85 else 0.5
            else:
                grid[i, j] = 0.0
            row.append(f"{conf:.2f}")
        text_grid.append(row)

    fig_height = max(6.0, 0.32 * len(claims))
    fig, ax = plt.subplots(figsize=(7.5, fig_height))
    cmap = plt.matplotlib.colors.ListedColormap(["#c0392b", "#e8b14f", "#3aaa6b"])
    bounds = [-0.5, 0.25, 0.75, 1.5]
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    masked = np.ma.masked_invalid(grid)
    ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=20, ha="right")
    ax.set_yticks(range(len(claims)))
    ax.set_yticklabels(claims, fontsize=8)
    for i in range(len(claims)):
        for j in range(len(models)):
            ax.text(
                j,
                i,
                text_grid[i][j],
                ha="center",
                va="center",
                color="white",
                fontsize=7,
                fontweight="bold",
            )
    ax.set_title("Cross-model agentic fidelity (confidence shown in cell)")
    fig.tight_layout()
    out = FIGURES_DIR / "fidelity_heatmap.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2: per-tool dispatch heatmap
# ---------------------------------------------------------------------------


def tool_dispatch_heatmap() -> Path:
    """Tool name x model, coloured by mean calls per cell.

    Visualises the apples-to-apples engagement claim: all three
    models exercise the tool surface comparably.
    """
    by_model_tool: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    all_tools: set[str] = set()
    for model in MODEL_ORDER:
        for rec in _load_cells(model).values():
            tc = rec.get("n_tool_calls_by_tool") or {}
            for tool, count in tc.items():
                if tool in _SUBMIT_TOOLS:
                    continue
                by_model_tool[model][tool].append(count)
                all_tools.add(tool)
            # tools never called this cell get a 0
            for tool in all_tools - set(tc):
                if tool in _SUBMIT_TOOLS:
                    continue
                by_model_tool[model][tool].append(0)

    # Stable tool order: datasheet-side first, then chamber-side
    datasheet_tools = [
        "build_datasheet",
        "get_section_text",
        "search_text",
        "extract_table_markdown",
        "inspect_page",
    ]
    chamber_tools = [
        "list_experiments",
        "get_experiment_metadata",
        "query_dataset",
        "cross_sensor_check",
        "run_simulator",
        "get_ground_truth_graph",
    ]
    ordered_tools = [t for t in datasheet_tools + chamber_tools if t in all_tools]
    # Anything else surfaced (unexpected) goes last alphabetically
    extras = sorted(all_tools - set(ordered_tools))
    ordered_tools.extend(extras)

    grid = np.zeros((len(ordered_tools), len(MODEL_ORDER)))
    for j, model in enumerate(MODEL_ORDER):
        for i, tool in enumerate(ordered_tools):
            calls = by_model_tool[model].get(tool, [])
            grid[i, j] = statistics.mean(calls) if calls else 0.0

    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(ordered_tools) + 2))
    im = ax.imshow(grid, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], rotation=20, ha="right")
    ax.set_yticks(range(len(ordered_tools)))
    ax.set_yticklabels(ordered_tools)
    for i in range(len(ordered_tools)):
        for j in range(len(MODEL_ORDER)):
            v = grid[i, j]
            colour = "white" if v < grid.max() * 0.55 else "black"
            ax.text(
                j, i, f"{v:.2f}", ha="center", va="center", color=colour, fontsize=8
            )
    fig.colorbar(im, ax=ax, label="mean calls per cell")
    ax.set_title("Per-tool dispatch by model (submit tools excluded)")
    fig.tight_layout()
    out = FIGURES_DIR / "tool_dispatch_heatmap.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 3: cost-vs-latency scatter
# ---------------------------------------------------------------------------


def cost_latency_scatter() -> Path:
    """One point per cell on (cost, latency) coloured by model.

    The Pareto frontier visualised: Sonnet faster-but-expensive,
    qwen cheap-but-slow-and-flaky, gpt-5.1 the middle compromise.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for model in MODEL_ORDER:
        xs, ys = [], []
        for rec in _load_cells(model).values():
            xs.append(_cell_cost(rec, model))
            ys.append(rec.get("latency_s", 0.0))
        ax.scatter(
            xs,
            ys,
            s=42,
            alpha=0.75,
            color=MODEL_COLOURS[model],
            edgecolor="white",
            linewidth=0.7,
            label=MODEL_LABELS[model],
        )
        if xs:
            mx = statistics.mean(xs)
            my = statistics.mean(ys)
            ax.scatter(
                [mx],
                [my],
                s=240,
                marker="X",
                color=MODEL_COLOURS[model],
                edgecolor="black",
                linewidth=1.4,
                zorder=5,
            )
            ax.annotate(
                f"mean: ${mx:.3f}, {my:.0f}s",
                (mx, my),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=8,
                color="black",
            )
    ax.set_xscale("log")
    ax.set_xlabel("cost per cell (USD, public-list pricing; log scale)")
    ax.set_ylabel("latency per cell (s)")
    ax.set_title("Cost vs. latency, agentic 60-cell matrix")
    ax.grid(True, which="both", alpha=0.25, linestyle="--")
    ax.legend(loc="best")
    fig.tight_layout()
    out = FIGURES_DIR / "cost_latency_scatter.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 4: engagement over revisions
# ---------------------------------------------------------------------------


# Manually curated historical data. Revisions 1 and 2 come from earlier
# commits' aggregates and are not recomputable from the shipped archive,
# which holds only the final revision; revision 3 is computed live from the
# current data files. Revisions 1-2 delivered structured output through an
# output_format constraint, which masked Qwen's tool tokens to zero;
# revision 3 switched to the submit_claim_result tool, and the jump from
# 0.0 to 11.6 nav tools/cell is that change, not a model change.
HISTORICAL_NAV_TOOLS: dict[str, dict[str, float | None]] = {
    "claudesonnet4.6": {
        "revision1_output_format": 9.3,
        "revision2_output_format": 9.3,
        "revision3_submit_tool": None,
    },
    "gpt-5.1": {
        "revision1_output_format": 9.95,
        "revision2_output_format": 9.95,
        "revision3_submit_tool": None,
    },
    "qwen3.6-27b": {
        "revision1_output_format": 0.0,
        "revision2_output_format": 0.0,
        "revision3_submit_tool": None,
    },
}


def engagement_over_revisions() -> Path:
    """Bar chart: per-model mean nav tools/cell over revisions.

    Tells the qwen 0.0 -> 11.6 story visually in one figure. The
    Revisions 1-2 are historical (curated constants); revision 3 is
    computed live so the figure stays honest when the matrix is
    regenerated.
    """
    revisions = [
        "revision1_output_format",
        "revision2_output_format",
        "revision3_submit_tool",
    ]
    # Labelled by the structured-output mechanism each revision used.
    rev_labels = [
        "Revision 1\n(output_format)",
        "Revision 2\n(output_format)",
        "Revision 3\n(submit_claim_result)",
    ]
    # Compute the final revision live
    for model in MODEL_ORDER:
        cells = list(_load_cells(model).values())
        nav = [
            _nav_tools_per_cell(r)
            for r in cells
            if r.get("fidelity", {}).get("overall_pass")
        ]
        HISTORICAL_NAV_TOOLS[model]["revision3_submit_tool"] = (
            statistics.mean(nav) if nav else 0.0
        )

    width = 0.25
    x = np.arange(len(revisions))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, model in enumerate(MODEL_ORDER):
        ys = [HISTORICAL_NAV_TOOLS[model][r] or 0.0 for r in revisions]
        bars = ax.bar(
            x + (i - 1) * width,
            ys,
            width,
            color=MODEL_COLOURS[model],
            label=MODEL_LABELS[model],
            edgecolor="white",
            linewidth=0.6,
        )
        for bar, y in zip(bars, ys, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y + 0.15,
                f"{y:.2f}" if y > 0 else "0",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(rev_labels)
    ax.set_ylabel("mean navigation tools per cell")
    ax.set_title(
        "Agentic tool engagement across three harness revisions\n"
        "(mean navigation tool calls per cell; submit tools excluded)"
    )
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.set_ylim(
        0,
        max(
            13.0,
            max(
                v
                for m in MODEL_ORDER
                for v in HISTORICAL_NAV_TOOLS[m].values()
                if v is not None
            )
            * 1.2,
        ),
    )
    fig.tight_layout()
    out = FIGURES_DIR / "engagement_over_revisions.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"writing figures to {short_path(FIGURES_DIR)}")
    out1 = fidelity_heatmap()
    print(f"  [1/4] {out1.name}")
    out2 = tool_dispatch_heatmap()
    print(f"  [2/4] {out2.name}")
    out3 = cost_latency_scatter()
    print(f"  [3/4] {out3.name}")
    out4 = engagement_over_revisions()
    print(f"  [4/4] {out4.name}")
    print("done.")


if __name__ == "__main__":
    main()
