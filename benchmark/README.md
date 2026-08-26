# Chamber-grounded benchmark

The grading surface for a datasheet-extraction agent, and the archived model
outputs it graded.

An extraction agent can be wrong in two different ways, and most evaluations
only see one of them. It can misread the document — call that **fidelity**.
Or it can faithfully report a number that is not physically true — call that
**reproducibility**. This benchmark grades both, using
[`datasheetindex`](https://github.com/Infineon/datasheetindex) for the
document side and the [Causal Chambers](https://causalchamber.org) for the
physical side.

The finding that motivated it: **a model passed our fidelity check without
ever opening the datasheet.** Fidelity alone cannot see that, which is why
this repository ships the dispatch-level detector rules alongside the scorer.

> **Paper.** *Fidelity Is Not Enough: Dispatch-Level Instrumentation for
> Datasheet-Extraction Agents.* EMNLP 2026 Industry Track.
> <!-- arXiv link goes here once posted; see docs/reproducing.md for which
>      numbers in the paper each artifact below regenerates. -->

## What is here, and what is not

This is the **scoring** half of the benchmark. It contains everything needed
to re-derive a published number from archived model output:

| | |
|---|---|
| `src/chamberbench/` | the grading surface — fidelity, reproducibility, the detector rules, the quality gates |
| `data/` | the 25-claim set, the 12-claim off-corpus set, and the human annotation files |
| `archive/` | the archived model outputs every published number is computed from |
| `scripts/` | the analyses that turn the archive into the paper's tables and figures |

It does **not** contain the agent harness that produced the archive, or the
datasheet corpus. Neither is a licensing problem so much as a scoping one, and
both are addressed in [`docs/reproducing.md`](docs/reproducing.md):

- **The corpus** is four manufacturer datasheets, publicly downloadable but not
  ours to redistribute. Three of the four are third-party (Silicon
  Laboratories, Allegro MicroSystems); the fourth is Infineon's own DPS310.
  They are identified by part number and revision, with checksums.
- **The harness** is the agent under test. Releasing it is tracked separately;
  it is not needed to check any number reported in the paper.

## Reproduce the published numbers, offline

No API key. No PDFs. No network — with one exception, marked in the table in
[`docs/reproducing.md`](docs/reproducing.md): `scan_natural_divergence.py`
fetches the public Causal Chambers dataset (~1 MB, cached under
`CHAMBER_CACHE_ROOT`, default `/tmp/cc_data`).

```bash
cd benchmark
uv venv && uv pip install -e '.[test]'
uv run python scripts/render_paper_tables.py    # the paper's model-comparison tables
uv run python scripts/render_paper_figures.py   # its figures
uv run pytest -q                                # 141 tests, incl. the paper's numbers
```

`uv.lock` pins the numeric stack (numpy, pandas, matplotlib) the published
numbers were computed with; `uv sync` uses it. An unpinned numpy is a real
hazard for an artifact like this — a future change to a percentile or
summation default would move a table with nothing failing.

Everything under `scripts/` runs against `archive/` alone. That is the point
of shipping the archive: the derivations are checkable without re-running a
model, and without trusting that we ran one.

To re-score the archive under a *different* grading surface — a re-derived
claim set, stricter substrings, a different confidence floor — use
`regrade_archive.py`, which runs the archived extractions back through the
same grader that produced the published verdicts:

```bash
CHAMBERBENCH_DATA_DIR=/path/to/your/claims uv run python scripts/regrade_archive.py
```

This is the honest way to attack the results: the grading surface is
hand-written, and a disagreement about it is a finding rather than a bug.

**Two limits on that, stated up front.** Only `baseline_chamber.json` keeps the
raw extractions, so only its 149 cells can be re-graded; the variance repeats
store verdicts alone, and Table 1's spread therefore has to be taken on trust.
And the env var reaches the *re-scoring* scripts — `regrade_archive.py`,
`score_rederivation.py`, `strict_fidelity_rescore.py` — not the renderers,
which print verdicts the archive already holds.

## The two axes, and why they are kept apart

**Fidelity** (`chamberbench.grading`) asks whether the agent reported what the
datasheet says. `evaluate_case` is a pure function of the extracted value and
the claim's `value_contains` / `confidence_min` fields.

**Reproducibility** (`chamberbench.reproducibility`) asks whether the claimed
value is physically true. `verdict()` takes a `ClaimSpec` and a
`ChamberMeasurement` — **it never sees agent output**. That independence is
structural rather than a convention, and it is what lets the two verdicts
disagree informatively.

**The detector rules** (`chamberbench.silent_failure`) are the dispatch-level
signals: `tool_bypass` (a document-grounded answer submitted without reading
the document) and `tool_read_failed` (every read errored). Both are predicates
over the `datasheetindex` tool surface rather than over anything private,
which is why the benchmark lives in this repository — the surface they read
can be stood up from the library itself.

## Version pinning

**The benchmark is pinned to a `datasheetindex` version, and this matters.**
The tool surface changed between 0.31 and 0.34: `build_datasheet` now nudges
`search_text` on LLM-reconstructed tables of contents, and running-headers are
stripped from page-matched text. **Tool-call counts are not comparable across
that boundary.** The published numbers were produced against the version
recorded in [`docs/reproducing.md`](docs/reproducing.md); re-running at repository
HEAD is a valid experiment but not a reproduction of the paper.

## Licence and attribution

MIT for everything we wrote — the code, the docs, the claim sets, and the
archived model outputs. **That licence does not extend to everything in
`archive/`.** The archived traces record what tools returned to the model, and
those tool outputs contain short excerpts of third-party datasheets (Silicon
Laboratories, Allegro MicroSystems) which remain their authors' copyright and
are reproduced only so that published results can be verified. The datasheets
themselves are not included here. [`NOTICE`](./NOTICE) states the scope
precisely and gives a takedown contact.

Physical measurements come from the Causal Chambers dataset, distributed under
CC BY 4.0, which requires attribution:

```bibtex
@article{gamella2025chambers,
  title   = {Causal Chambers as a Real-World Physical Testbed for {AI} Methodology},
  author  = {Gamella, Juan L. and Peters, Jonas and B{\"u}hlmann, Peter},
  journal = {Nature Machine Intelligence},
  volume  = {7},
  number  = {1},
  pages   = {107--118},
  year    = {2025},
  doi     = {10.1038/s42256-024-00964-x},
}
```
