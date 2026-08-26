"""Compute classifier-vs-gold agreement at cell level.

Reads:
  - ``data/classifier_gold.yaml`` (hand-labelled)
  - ``archive/classifier_auto.{model}.json`` (auto-labelled,
    produced by prepare_gold_labelling.py)

Produces:
  - ``archive/classifier_agreement.md`` -- human-readable
    summary with overall agreement, per-label breakdown, and a
    "what this means for the paper's per-tool plot" paragraph ready to
    drop into the methodology doc.

The cell-level gold labels live in a controlled vocabulary
(see docs/reproducing.md):

  classifier_correct
  classifier_missed_tool_output
  classifier_missed_tool_selection
  classifier_missed_condition_omission
  classifier_missed_verification_skipped
  classifier_overflagged

For each cell, gold = classifier_correct means the auto-label is
trusted; any other label means the classifier missed something. We
report:
  - cell-level agreement rate (= fraction labelled classifier_correct)
  - per-classifier-missed-label histogram
  - per-tool implication: which tools the missed events touched

Run:
    uv run python scripts/compute_classifier_agreement.py
    uv run python scripts/compute_classifier_agreement.py --gold <path>

With two --gold files it also reports INTER-ANNOTATOR agreement between
them. That is the number the paper's Limitations calls for by name: the
classifier is currently validated against a single annotator, so a second
one labelling the same 30 cells says whether the rubric means the same
thing to two people, not just whether one person agrees with the code.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from chamberbench.claimsio import archive_dir, data_dir, short_path

RESULTS_DIR = archive_dir()
GOLD_YAML = data_dir() / "classifier_gold.yaml"
AGREEMENT_MD = RESULTS_DIR / "classifier_agreement.md"

LABELS = (
    "classifier_correct",
    "classifier_missed_tool_output",
    "classifier_missed_tool_selection",
    "classifier_missed_condition_omission",
    "classifier_missed_verification_skipped",
    "classifier_overflagged",
)


def _normalize_label(raw: Any) -> list[str]:
    """Coerce gold_label field to a list of strings.

    Accepts string, list-of-string, empty string (abstain), or None.
    """
    if raw is None or raw == "":
        return []
    if isinstance(raw, str):
        return [raw.strip()]
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    return []


def _load_auto_labels() -> dict[str, dict[str, Any]]:
    """Load per-cell auto-labels from classifier_auto.{model}.json files.

    Returns ``{cell_id: cell_record}`` keyed by ``<cid>|<eng>|<model>``
    to match the gold YAML's cell_id format.
    """
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(RESULTS_DIR.glob("classifier_auto.*.json")):
        model = path.stem.removeprefix("classifier_auto.")
        data = json.loads(path.read_text(encoding="utf-8"))
        for key, cell in (data.get("results") or {}).items():
            cid = cell.get("claim_id") or key.split("|", 1)[0]
            eng = cell.get("engine") or (key.split("|", 1)[1] if "|" in key else "")
            cell_id = f"{cid}|{eng}|{model}"
            out[cell_id] = cell
    return out


def _tools_touched(cell: dict[str, Any]) -> list[str]:
    """Tools the cell exercised (from n_tool_calls_by_tool)."""
    counts = cell.get("n_tool_calls_by_tool") or {}
    return sorted(t for t, n in counts.items() if n > 0)


def _classifier_flagged(cell: dict[str, Any]) -> bool:
    """The classifier's cell-level call: did it raise any non-ok event?

    The classifier emits a per-event ``attribution`` list; a cell is
    PROBLEMATIC if any event is non-ok or the run hit an engine error,
    else CLEAN.
    """
    counts = cell.get("attribution_counts") or {}
    non_ok = sum(n for lab, n in counts.items() if lab != "ok")
    if cell.get("engine_error"):
        non_ok += 1
    return non_ok > 0


def _cohens_kappa(
    labelled: dict[str, list[str]], auto_cells: dict[str, dict[str, Any]]
) -> tuple[float, float, float, dict[str, dict[str, int]], int]:
    """Chance-corrected agreement on the binary CLEAN/PROBLEMATIC call.

    Two raters per adjudicated cell:
      - classifier: PROBLEMATIC iff it raised a non-ok event (see
        ``_classifier_flagged``), else CLEAN.
      - human (from gold_label): ``classifier_correct`` means the human
        agrees with the classifier's call (human = classifier);
        ``classifier_missed_*`` means a real problem the classifier
        missed (human = PROBLEMATIC); ``classifier_overflagged`` means no
        real problem (human = CLEAN).

    Raw agreement is inflated by the high base rate of clean cells, so we
    report Cohen's kappa alongside it. Returns
    ``(kappa, p_observed, p_expected, confusion, n)``.
    """
    cats = ("CLEAN", "PROBLEMATIC")
    cm = {a: {b: 0 for b in cats} for a in cats}
    for cell_id, ls in labelled.items():
        auto = auto_cells.get(cell_id)
        if auto is None:
            continue  # no auto-label to compare against; skip from kappa
        clf = "PROBLEMATIC" if _classifier_flagged(auto) else "CLEAN"
        if ls == ["classifier_correct"]:
            human = clf
        elif ls == ["classifier_overflagged"]:
            human = "CLEAN"
        else:  # any classifier_missed_* (or a mixed list) => problem exists
            human = "PROBLEMATIC"
        cm[clf][human] += 1
    n = sum(cm[a][b] for a in cats for b in cats)
    if n == 0:
        return float("nan"), 0.0, 0.0, cm, 0
    po = sum(cm[a][a] for a in cats) / n
    pe = sum(
        (sum(cm[a][b] for b in cats) / n) * (sum(cm[x][a] for x in cats) / n)
        for a in cats
    )
    kappa = (po - pe) / (1 - pe) if (1 - pe) else float("nan")
    return kappa, po, pe, cm, n


def _inter_annotator(first: Path, second: Path) -> int:
    """Cell-level agreement between two annotators over the cells both labelled."""
    docs = []
    for path in (first, second):
        if not path.exists():
            print(f"ERROR: gold labels file not found: {path}")
            return 1
        docs.append(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    names = [
        (d.get("metadata") or {}).get("annotator") or path.name
        for d, path in zip(docs, (first, second), strict=True)
    ]
    labels = [
        {
            c["cell_id"]: _normalize_label(c.get("gold_label"))
            for c in (d.get("cells") or [])
        }
        for d in docs
    ]
    shared = [cid for cid in labels[0] if labels[0][cid] and labels[1].get(cid)]
    if not shared:
        print("No cell is labelled in both files yet -- nothing to compare.")
        return 0

    exact = [cid for cid in shared if sorted(labels[0][cid]) == sorted(labels[1][cid])]

    # The binary reduction the paper reports: did each annotator call the cell
    # clean, or did they flag something? Two people can disagree on WHICH miss
    # a cell shows and still agree that it is not clean.
    def clean(ls: list[str]) -> bool:
        return ls == ["classifier_correct"]

    binary = [cid for cid in shared if clean(labels[0][cid]) == clean(labels[1][cid])]
    print("=" * 70)
    print("INTER-ANNOTATOR AGREEMENT")
    print("=" * 70)
    print(f"  Annotators: {names[0]!r} vs {names[1]!r}")
    print(f"  Cells labelled by both:  {len(shared)}")
    print(
        f"  Exact label match:       {len(exact)}/{len(shared)} = {len(exact) / len(shared):.1%}"
    )
    print(
        f"  Clean/flagged match:     {len(binary)}/{len(shared)} = {len(binary) / len(shared):.1%}"
    )
    disagreements = [
        cid for cid in shared if sorted(labels[0][cid]) != sorted(labels[1][cid])
    ]
    if disagreements:
        print()
        print("  Disagreements (report every one, whatever it shows):")
        for cid in disagreements:
            print(f"    {cid}")
            print(f"      {names[0]}: {labels[0][cid]}")
            print(f"      {names[1]}: {labels[1][cid]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Classifier-vs-gold agreement (and inter-annotator agreement)."
    )
    ap.add_argument(
        "--gold",
        type=Path,
        action="append",
        help="Gold-label YAML. Repeat once to add a second annotator and get inter-annotator agreement.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to write the agreement report (default: beside the shipped one).",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the shipped archive/classifier_agreement.md in place.",
    )
    args = ap.parse_args()
    golds = args.gold or [GOLD_YAML]

    if len(golds) > 2:
        print("ERROR: pass at most two --gold files.")
        return 1
    if len(golds) == 2:
        rc = _inter_annotator(golds[0], golds[1])
        print()
        # Then fall through and score each against the classifier.
        for path in golds:
            print()
            rc = max(rc, _score_one(path, args.out, args.force))
        return rc
    return _score_one(golds[0])


def _score_one(gold_path: Path, out_md: Path | None = None, force: bool = False) -> int:
    global GOLD_YAML
    GOLD_YAML = gold_path
    if not GOLD_YAML.exists():
        print(f"ERROR: gold labels file not found: {GOLD_YAML}")
        print("Run scripts/prepare_gold_labelling.py first, then label cells.")
        return 1

    gold_doc = yaml.safe_load(GOLD_YAML.read_text(encoding="utf-8"))
    auto_cells = _load_auto_labels()

    sample = gold_doc.get("cells") or []
    if not sample:
        print("ERROR: no `cells:` block in gold YAML.")
        return 1

    # Status counters
    abstain: list[str] = []
    invalid: list[tuple[str, str]] = []  # (cell_id, unknown_label)
    labelled: dict[str, list[str]] = {}
    cell_lookup: dict[str, dict[str, Any]] = {c["cell_id"]: c for c in sample}

    for cell in sample:
        cell_id = cell["cell_id"]
        labels = _normalize_label(cell.get("gold_label"))
        if not labels:
            abstain.append(cell_id)
            continue
        bad = [lbl for lbl in labels if lbl not in LABELS]
        if bad:
            for b in bad:
                invalid.append((cell_id, b))
            # Keep the valid ones if any; if none, abstain.
            labels = [lbl for lbl in labels if lbl in LABELS]
            if not labels:
                abstain.append(cell_id)
                continue
        labelled[cell_id] = labels

    total_labelled = len(labelled)
    correct_n = sum(1 for ls in labelled.values() if ls == ["classifier_correct"])
    agreement_rate = (correct_n / total_labelled) if total_labelled else 0.0

    # Per-label histogram (excluding classifier_correct, which is the
    # "no miss" bucket reported separately)
    miss_hist: Counter[str] = Counter()
    miss_cells_by_label: dict[str, list[str]] = defaultdict(list)
    for cell_id, ls in labelled.items():
        for label in ls:
            if label == "classifier_correct":
                continue
            miss_hist[label] += 1
            miss_cells_by_label[label].append(cell_id)

    # Per-tool implication: aggregate tools touched by miss-labelled cells
    tools_in_miss_cells: Counter[str] = Counter()
    for cell_id, ls in labelled.items():
        if ls == ["classifier_correct"]:
            continue
        cell_record = auto_cells.get(cell_id) or cell_lookup.get(cell_id, {})
        for t in _tools_touched(cell_record):
            tools_in_miss_cells[t] += 1

    # Pretty-print to stdout
    print("=" * 70)
    print("CHAMBER CLASSIFIER-VS-GOLD AGREEMENT")
    print("=" * 70)
    print(f"  Annotator: {gold_doc.get('metadata', {}).get('annotator') or 'unknown'}")
    print(f"  Total cells in sample:   {len(sample)}")
    print(f"  Labelled:                {total_labelled}")
    print(f"  Abstained (no label):    {len(abstain)}")
    print(f"  Invalid label entries:   {len(invalid)}")
    print()
    print(
        f"  Cell-level agreement: {correct_n}/{total_labelled} = {agreement_rate:.1%}"
    )

    kappa, k_po, k_pe, k_cm, k_n = _cohens_kappa(labelled, auto_cells)
    if k_n:
        print(
            f"  Cohen's kappa (CLEAN/PROBLEMATIC, n={k_n}): {kappa:.3f} (p_o={k_po:.3f}, p_e={k_pe:.3f})"
        )
        print(
            "    confusion [classifier x human]: "
            f"CC={k_cm['CLEAN']['CLEAN']} "
            f"CP={k_cm['CLEAN']['PROBLEMATIC']} "
            f"PC={k_cm['PROBLEMATIC']['CLEAN']} "
            f"PP={k_cm['PROBLEMATIC']['PROBLEMATIC']}"
        )
    print()
    if miss_hist:
        print("  Classifier-missed labels (cell counts):")
        for label in LABELS:
            if label == "classifier_correct":
                continue
            n = miss_hist.get(label, 0)
            if n:
                print(f"    {label:<42s} {n}")
    if invalid:
        print()
        print("  Invalid label entries (typos? not in rubric):")
        for cell_id, bad_label in invalid:
            print(f"    {cell_id}: {bad_label!r}")
    if abstain:
        print()
        print(f"  Abstained cells ({len(abstain)}):")
        for cell_id in abstain:
            print(f"    {cell_id}")
    print()

    # Threshold interpretation from the paper plan
    if total_labelled == 0:
        verdict = "no labels yet -- run after labelling cells"
    elif agreement_rate >= 0.85:
        verdict = (
            "Strong: the classifier's per-event labels are well-defended at "
            f"{agreement_rate:.0%}. The per-tool error-rate plot can be "
            "reported with confidence; bound uncertainty at "
            f"~±{(1 - agreement_rate) * 100:.0f}% per cell."
        )
    elif agreement_rate >= 0.70:
        verdict = (
            f"Moderate: {agreement_rate:.0%} agreement. The per-tool "
            "error-rate plot is reportable but with widened uncertainty "
            "bounds (~±15% per cell). The Limitations section should "
            "discuss the disagreement pattern explicitly."
        )
    else:
        verdict = (
            f"Below threshold: {agreement_rate:.0%} agreement. Per the plan, "
            "this becomes a paper-section unto itself -- the classifier as "
            "currently implemented does not reliably attribute errors. The "
            "per-tool plot becomes descriptive only; classifier improvement "
            "is named future work."
        )
    print(f"  Verdict: {verdict}")
    print("=" * 70)

    # Write markdown report
    md_lines: list[str] = []
    md_lines.append("# Chamber classifier-vs-gold agreement")
    md_lines.append("")
    md_lines.append(f"Generated: {datetime.now(UTC).isoformat()}")
    # No annotator name is written here, and none is held anywhere else either.
    # The premise this originally rested on -- "leave the name in the gold file's
    # metadata, which is not part of a release" -- was wrong: `data/*.yaml` ships
    # with the benchmark, which is how two real names reached a public release.
    # Gold files now carry a non-identifying label (`annotator-1`, `annotator-2`),
    # enforced by `tests/test_annotator_privacy.py`. What matters for independence
    # is the *relationship* between annotator and surface, not the name.
    md_lines.append(
        "Annotator: [redacted for review; identity recorded in the gold file metadata]"
    )
    md_lines.append("")
    md_lines.append("## Headline")
    md_lines.append("")
    md_lines.append(
        f"Cell-level agreement: **{correct_n}/{total_labelled} = "
        f"{agreement_rate:.1%}** "
        f"(n_sample={len(sample)}, abstained={len(abstain)}, "
        f"invalid={len(invalid)})"
    )
    md_lines.append("")
    if k_n:
        md_lines.append(
            f"Cohen's kappa (binary CLEAN vs PROBLEMATIC, n={k_n}): "
            f"**{kappa:.3f}** (observed agreement {k_po:.1%}, chance "
            f"{k_pe:.1%}). Confusion [classifier x human]: "
            f"clean/clean={k_cm['CLEAN']['CLEAN']}, "
            f"clean/problematic={k_cm['CLEAN']['PROBLEMATIC']}, "
            f"problematic/clean={k_cm['PROBLEMATIC']['CLEAN']}, "
            f"problematic/problematic={k_cm['PROBLEMATIC']['PROBLEMATIC']}."
        )
        md_lines.append("")
    md_lines.append(f"**Verdict**: {verdict}")
    md_lines.append("")
    if miss_hist:
        md_lines.append("## Classifier-missed findings")
        md_lines.append("")
        md_lines.append("| Label | Cell count |")
        md_lines.append("|---|---:|")
        for label in LABELS:
            if label == "classifier_correct":
                continue
            n = miss_hist.get(label, 0)
            if n:
                md_lines.append(f"| `{label}` | {n} |")
        md_lines.append("")
        md_lines.append("Cell breakdown:")
        for label, cells in miss_cells_by_label.items():
            md_lines.append("")
            md_lines.append(f"### `{label}` ({len(cells)} cells)")
            md_lines.append("")
            for c in cells:
                md_lines.append(f"- `{c}`")
    if tools_in_miss_cells:
        md_lines.append("")
        md_lines.append("## Tools touched by miss-labelled cells")
        md_lines.append("")
        md_lines.append(
            "These are the tools whose per-tool error rates should be reported with widened uncertainty bounds:"
        )
        md_lines.append("")
        md_lines.append("| Tool | Miss-labelled cells |")
        md_lines.append("|---|---:|")
        for t, n in sorted(tools_in_miss_cells.items(), key=lambda kv: -kv[1]):
            md_lines.append(f"| `{t}` | {n} |")
    md_lines.append("")
    md_lines.append("## Methodology-doc paragraph (drop-in)")
    md_lines.append("")
    md_lines.append(
        "> *Across a hand-labelled sample of "
        + str(total_labelled)
        + " cells (stratified across model and component), the auto-classifier "
        "agreed with cell-level human judgment in " + str(correct_n) + " cells, "
        f"for an agreement rate of {agreement_rate:.0%}. "
        + (
            "The remaining disagreements were concentrated in the `"
            + (max(miss_hist, key=lambda k: miss_hist[k]) if miss_hist else "n/a")
            + "` category, primarily on cells exercising "
            + (
                max(tools_in_miss_cells, key=lambda k: tools_in_miss_cells[k])
                if tools_in_miss_cells
                else "the tool surface broadly"
            )
            + ". "
            if miss_hist
            else ""
        )
        + "The per-tool error-rate plot in this paper should be read with this "
        + f"~{(1 - agreement_rate) * 100:.0f}% classifier-disagreement bound in mind.*"
    )
    # `classifier_agreement.md` is tracked evidence, and this script is a
    # DOCUMENTED read-only analysis -- so writing it by default meant the
    # documented command churned the archive on every run. Regenerable, so the
    # default now writes beside it rather than refusing.
    out_md = out_md or AGREEMENT_MD
    if out_md == AGREEMENT_MD and AGREEMENT_MD.exists() and not force:
        out_md = PROJECT_ROOT / "classifier_agreement.regenerated.md"
        print(f"  (archive copy left intact; writing {short_path(out_md)})")
        print("   pass --force to overwrite the shipped report, or --out <path>)")
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"  wrote {short_path(out_md)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
