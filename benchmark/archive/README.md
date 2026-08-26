# The archive

Archived model outputs. Every number the paper reports is computed from these
files, and they are the reason the results can be checked without re-running a
model.

**Treat this directory as evidence.** It is excluded from the repository's
whitespace hooks so that the released bytes are the ones the numbers came from,
and the scripts that write here refuse to overwrite without `--force`.

## Two things to know before parsing

**1. Nine files are not valid RFC 8259 JSON.** They contain bare `NaN` tokens,
which Python's `json` accepts and `jq`, JavaScript, Go and R's `jsonlite` all
reject:

    baseline_chamber.json, latest_chamber*.json, classifier_auto.*.json

`NaN` is deliberate rather than accidental — it marks a measurement that was
never taken, and the accompanying `measured_sigma_basis == "stub"` says the same thing.
It should have been serialised as `null`. Rewriting it now would change the
bytes of the archive, so it stays, documented. To read these outside Python:

```bash
sed 's/: NaN/: null/g' baseline_chamber.json | jq .
```

Match the colon, not the word. `NaN` also appears **inside string values** here
(262 of the 661 occurrences in `baseline_chamber.json` are prose such as
`"measured_value=NaN by design"`), and a `\bNaN\b` replacement rewrites those
too — silently altering the data while appearing to fix only the syntax.

**2. Naming is not fully consistent.** Most files are
`<experiment>.<model-alias>[.<variant>].json`, but the `fault_injection_*`
files use underscores inside the model alias (`gpt-5_1`, `qwen3_6-27b`) where
every other family uses dots (`gpt-5.1`, `qwen3.6-27b`). Historical; the
contents are unaffected.

Model aliases are the internal gateway's names, not provider snapshot IDs —
see [`../docs/reproducing.md`](../docs/reproducing.md).

## What each file is for

### Primary results

| File | What it holds | Used by |
|---|---|---|
| `baseline_chamber.json` | The full post-audit matrix: per claim x engine x model, including `claim_result` (the raw extraction). The re-gradable file: `regrade_archive.py` reads its 149 stored extractions. The `latest_chamber.{model}.json` copies carry the same 149 cells byte-identically; `latest_chamber.qwen3.5-27b.json` and `a4988_fidelity_rerun.*` hold further extractions outside the published matrix. | most analyses; `regrade_archive.py` |
| `variance_chamber.json` | Three repeats per model, reduced to Table 1. Stores verdicts **without** extractions, so it cannot be re-graded. A primary artifact: it was NOT produced by `consolidate_variance.py` from the inputs below — see that script's guard. | `render_paper_tables.py` |
| `latest_chamber.{model}.json` | Per-model single-run detail behind the cost figures. `latest_chamber.json` is a byte-identical copy of the Sonnet file, kept because scripts referenced both names. | `chamber_cost_summary.py` |
| `latest_traces.{model}.jsonl` | Per-step trace events: tool calls, tokens, latency. Per-step detail behind the classifier and the cost summary. (The paper's dispatch figure is built from `baseline_chamber.json`, not from these.) | `chamber_cost_summary.py`, classifier |

### Corrupt-success arms (the detector evidence)

These are the experiments behind the paper's central claim — that a model can
pass a fidelity check without reading the document, and that a dispatch-level
rule catches it.

| File | Arm |
|---|---|
| `closed_book.{model}.json` | The model is denied document access entirely. Any fidelity pass here is a corrupt success by construction. |
| `null_tool_injection.{model}.json` | Document tools are present but return nothing. `.thinking_on` is the Qwen reasoning-mode variant. |
| `wrong_content.{model}.json` | Tools return a *different* document. |
| `fault_injection{,_gpt-5_1,_qwen3_6-27b}.json` | Planted-fault runs behind the detector's false-positive scan. |

Each cell records the detector decision made at run time
(`detector_flagged`, `detector_rules`); `tests/test_silent_failure.py`
recomputes it from the shipped rules and fails if the two disagree.

### Supporting

| File | What it holds |
|---|---|
| `classifier_auto.{model}.json` | Auto-applied failure-attribution labels. With the two gold files in `../data/`, these produce Cohen's kappa. |
| `classifier_agreement.md` | The generated agreement report (regenerable). |
| `sampled_cells.json` | The cells sampled for blind gold labelling. |
| `a4988_fidelity*.json` | The off-corpus fourth component. `_rerun` is the later re-run on a changed tool surface. |
| `perturbation_sweep.json` | The controlled-perturbation sweep behind the reproducibility figure. |
| `variance_chamber.mainrun.bak.json`, `variance_gpt_rerun.json`, `variance_qwen_r3.json` | Superseded inputs to the May consolidation. Kept for provenance; **not** the published run. |
| `variance_qwen_no_think.json` | Qwen with reasoning disabled — the 25/24/24 result that shows the instability is a serving-stack artifact. |

## Regenerating

Only `classifier_agreement.md` and `figures/` are outputs. Everything else is
primary evidence: if a script offers to write it, it is offering to replace the
record, and will ask for `--force` first.
