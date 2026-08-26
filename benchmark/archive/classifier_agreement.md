# Chamber classifier-vs-gold agreement

Generated: 2026-08-26T09:00:59.324424+00:00
Annotator: [redacted for review; identity recorded in the gold file metadata]

## Headline

Cell-level agreement: **24/27 = 88.9%** (n_sample=30, abstained=3, invalid=0)

Cohen's kappa (binary CLEAN vs PROBLEMATIC, n=27): **0.609** (observed agreement 88.9%, chance 71.6%). Confusion [classifier x human]: clean/clean=21, clean/problematic=3, problematic/clean=0, problematic/problematic=3.

**Verdict**: Strong: the classifier's per-event labels are well-defended at 89%. The per-tool error-rate plot can be reported with confidence; bound uncertainty at ~±11% per cell.

## Classifier-missed findings

| Label | Cell count |
|---|---:|
| `classifier_missed_tool_selection` | 2 |
| `classifier_missed_verification_skipped` | 1 |

Cell breakdown:

### `classifier_missed_tool_selection` (2 cells)

- `acs70331-primary-resistance-qfn|agentic|claudesonnet4.6`
- `si115x-standby-current-1v8|agentic|claudesonnet4.6`

### `classifier_missed_verification_skipped` (1 cells)

- `acs70331-rise-time|agentic|qwen3.6-27b`

## Tools touched by miss-labelled cells

These are the tools whose per-tool error rates should be reported with widened uncertainty bounds:

| Tool | Miss-labelled cells |
|---|---:|
| `build_datasheet` | 3 |
| `get_section_text` | 3 |
| `search_text` | 3 |
| `list_experiments` | 2 |
| `submit_chamber_outcome` | 2 |
| `submit_extraction` | 2 |
| `extract_table_markdown` | 1 |
| `submit_claim_result` | 1 |

## Methodology-doc paragraph (drop-in)

> *Across a hand-labelled sample of 27 cells (stratified across model and component), the auto-classifier agreed with cell-level human judgment in 24 cells, for an agreement rate of 89%. The remaining disagreements were concentrated in the `classifier_missed_tool_selection` category, primarily on cells exercising build_datasheet. The per-tool error-rate plot in this paper should be read with this ~11% classifier-disagreement bound in mind.*
