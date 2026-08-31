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
> Agentic Datasheet Extraction.* EMNLP 2026 Industry Track.
> [arXiv:2608.28439](https://arxiv.org/abs/2608.28439)
> <!-- See docs/reproducing.md for which numbers in the paper each artifact
>      below regenerates. -->

## What is here, and what is not

Everything needed to re-derive a published number from archived model output,
and the harness that produced that output:

| | |
|---|---|
| `src/chamberbench/` | the grading surface — fidelity, reproducibility, the detector rules, the quality gates |
| `src/chamberbench/harness/` | the agent under test: the two engines, the tool surface, the runner |
| `data/` | the 25-claim set, the 12-claim off-corpus set, and the human annotation files |
| `archive/` | the archived model outputs every published number is computed from |
| `scripts/` | the analyses that turn the archive into the paper's tables and figures, plus the live experiment producers |
| `gateway/` | a reference LiteLLM config for the proxy the harness calls through |

The **agent harness** that produced the archive ships too, under
`src/chamberbench/harness/` — see [Running the harness](#running-the-harness)
below. It is a separate install tier: none of the offline reproduction needs
it, so Tier 1 pulls in no model client at all.

What is **not** here is the **datasheet corpus** — four manufacturer
datasheets, publicly downloadable but not ours to redistribute. Three of the
four are third-party (Silicon Laboratories, Allegro MicroSystems); the fourth
is Infineon's own DPS310. They are identified by part number and revision, with
checksums, in [`docs/reproducing.md`](docs/reproducing.md).

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
uv run pytest -q                                # 222 tests, incl. the paper's numbers
```

That install is Tier 1: the grading surface, the archive, and nothing that can
call a model. It reports **222 passed, 37 skipped**, and the 37 are not one
group but two:

- **29** cover the harness and skip because the `harness` extra is not
  installed. They run once it is — see below.
- **8** are `test_every_corrupt_success_is_caught`, parametrised over the ten
  archived injection arms. They skip in **both** tiers, on the data rather
  than the install: eight of the ten arms hold no *corrupt success* at all —
  no cell that passed fidelity without an engine error — so there is nothing
  for the detector to have been given a chance to miss. That is a real limit
  on the recall evidence rather than a gap in the suite: the recall claim
  rests on the two arms that do hold such cells,
  `closed_book.claudesonnet4.6` (7) and `closed_book.gpt-5.1` (1), eight
  cells in all.

With the `harness` extra the suite reports **261 passed, 8 skipped, 1
deselected**; the deselected one is the network-marked test, which
`pyproject.toml` deselects by default and which is invisible on Tier 1
because its module is import-skipped.

`uv.lock` pins the numeric stack (numpy, pandas, matplotlib) so that a future
release cannot silently move a table — a changed percentile or summation
default would otherwise do exactly that, with nothing failing. Note `uv sync`
reads the lock; the `uv pip install` above does not. The pin records **today's**
resolution, not the versions in use on the run dates (numpy 2.4.5/2.4.6, pandas
3.0.3, matplotlib 3.10.9); it freezes the analysis stack going forward rather
than reconstructing the original one.

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

## Running the harness

Re-running the agent, rather than re-deriving a number from the archive, is a
second install tier. Tier 1 above deliberately pulls in no model client at all,
so the harness's dependencies live behind an extra:

```bash
uv pip install -e '.[harness]'   # anthropic, openai, requests, tenacity, datasheetindex
uv run chamber-run --model claudesonnet4.6 --engine agentic --out results/
```

Without it, every Tier 2 command fails with a bare
`ModuleNotFoundError: No module named 'anthropic'`.

You will also need the corpus (see
[`docs/reproducing.md`](docs/reproducing.md#the-corpus) for parts, revisions
and checksums) and credentials for a gateway. Two documents cover the rest:

- **[`docs/regenerating.md`](docs/regenerating.md)** — the per-artifact
  manifest. Which producer wrote each file in `archive/`, the exact invocation,
  and every caveat where a recorded command cannot be replayed as written.
- **[`gateway/README.md`](gateway/README.md)** — the harness never calls a
  provider directly. The reference LiteLLM config, the three surfaces a gateway
  must expose, the TLS posture, and the one fidelity-critical check: that
  `extra_body` really reaches the Qwen backend.

Nothing in this section is needed to check a published number. That is the
point of the split, and [`docs/reproducing.md`](docs/reproducing.md) keeps the
two questions apart deliberately.

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
signals: `tool_bypass` (a document-grounded answer submitted without any
navigation call) and `verification_skipped` (the agent navigated but never
cross-checked what it found). Both are predicates
over the `datasheetindex` tool surface rather than over anything private,
which is why the benchmark lives in this repository — the surface they read
can be stood up from the library itself.

## Version pinning

**The scoring half of this benchmark does not import `datasheetindex`, and
that is worth being precise about.** The code that re-derives a published
number is pure Python over archived outputs, so nothing in it depends on the
library version. (The harness does import the library — that is what it drives.
So does exactly one script, `grounding_wrong_document.py`, which re-runs
`locate_text` over the corpus PDFs; it is a side analysis that regenerates no
shipped artifact and backs no published number, and it prints an install note
rather than failing when the library or the corpus is absent.) What *did*
depend on the library version was the agent run that produced the archive — and
the tool surface has moved a long way since. The published runs used 0.13.0 and 0.14.0; this repository's `main` is
0.34.0. Between 0.31 and 0.34 alone: `build_datasheet` now nudges
`search_text` on LLM-reconstructed tables of contents, and running-headers are
stripped from page-matched text. **Tool-call counts are not comparable across
that boundary.** The published numbers were produced against the version
recorded in [`docs/reproducing.md`](docs/reproducing.md); re-running at repository
HEAD is a valid experiment but not a reproduction of the paper.

## Licence and attribution

MIT for everything we wrote — the code, the docs, the claim sets, and the
archived model outputs. **That licence does not extend to everything in
`archive/`.** The archived traces record what tools returned to the model, and
those outputs — and the model's own reasoning about them — quote third-party
datasheets (Silicon Laboratories, Allegro MicroSystems) at length — for the
three heavily-navigated documents, between a quarter and half of the body text
appears verbatim somewhere in the archive, and individual pages can be partly
reconstructed. That text remains its authors' copyright and is reproduced only so
that published results can be verified. The datasheets themselves are not
included here. [`NOTICE`](./NOTICE) states the extent and scope precisely, with
measurements and a takedown contact.

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
