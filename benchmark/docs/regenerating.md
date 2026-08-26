# Regenerating the archive

This is a per-artifact manifest: for every file under `archive/` (except
`archive/README.md`, which is prose, not evidence), which command produced
it. `docs/reproducing.md` answers "can I check the paper's arithmetic" from
the archive alone, offline, with no model calls. This document answers a
different question -- "which command made this specific file" -- and it
covers both kinds of producer:

- **Tier 1** -- offline scripts under `scripts/` and `src/chamberbench/`
  that read `archive/` and `data/` and write a derived report or figure.
  No model call, no gateway credentials.
- **Tier 2** -- the live agent harness (`src/chamberbench/harness/`, the
  `chamber-run` CLI, and the scripts that call `extract_chamber_agentic` /
  `extract_chamber_baseline` directly: `scripts/variance.py`,
  `scripts/null_tool.py`, `scripts/fault_injection*.py`,
  `scripts/fourth_component.py`, `scripts/perturbation.py`). Every Tier 2
  command makes real, billable calls against a model gateway and needs
  credentials resolved by `chamberbench.credentials.setup_credentials()` /
  `chamberbench.harness.setup_gateway_credentials()`: `ANTHROPIC_API_KEY` +
  optional `ANTHROPIC_BASE_URL`, or `LITELLM_MASTER_KEY` + `LITELLM_BASE_URL`
  for a gateway, or a `.env` file. Install them with the `harness` extra:
  `uv pip install -e '.[harness]'`. Tier 1's `.[test]` install deliberately
  pulls in no model client, so without it every Tier 2 command below fails on
  a missing `anthropic`. See `../README.md`, "Running the harness", and
  `../gateway/README.md` for the proxy these calls go through.

**Do not build this manifest by grepping filenames.** Two reasons, both
verified while writing it: (1) several producers construct their output
filename at runtime -- `scripts/fault_injection_multimodel.py` writes
`f"fault_injection_{args.model.replace('.', '_')}.json"`, so a search for the
literal archived name `fault_injection_gpt-5_1.json` finds nothing in that
script; (2) a name match cannot distinguish a producer from a consumer --
`scripts/chamber_cost_summary.py` mentions `latest_chamber.*` because it
*reads* it, not because it writes it. Every row below comes from reading the
candidate producer's actual output-path logic, not from a filename search.

## Regeneration safety

`archive/` is read-only published evidence; nothing below should be run with
its output pointed directly at the shipped files unless you mean to replace
them.

- **The consolidation scripts default their `--out` into `archive/` and
  refuse to overwrite an existing file without `--force`:**
  `scripts/build_chamber_baseline.py`, `scripts/consolidate_variance.py`,
  `scripts/compute_classifier_agreement.py`,
  `scripts/prepare_gold_labelling.py`. Their commands below are written as
  shipped-default invocations plus the `--force` (or, for
  `prepare_gold_labelling.py`, `--rebuild-auto-labels`) flag that overwriting
  the archive in place requires.
- **The live single-run scripts take a required `--out` with no archive
  default, and their own `--help` text says the archive is never a valid
  target:** `scripts/variance.py`, `scripts/null_tool.py`,
  `scripts/fault_injection.py`, `scripts/fault_injection_multimodel.py`,
  `scripts/fourth_component.py`, `scripts/perturbation.py`. Commands below
  point `--out` at a scratch path (`results/<name>`) standing in for
  wherever the original run actually landed before its output was copied
  into `archive/` by hand; nothing records that intermediate path today.
- **`chamber-run` goes further and hard-refuses `archive/` as a target.**
  `ChamberResultsCollector.__init__` raises `ValueError` if `results_dir`
  resolves to `chamberbench.claimsio.archive_dir()` -- there is no `--force`
  override for this one. `--out` must be a scratch directory.
- **Figure scripts default their output to `benchmark/figures/` (gitignored
  scratch), not `archive/figures/`.** `scripts/render_paper_figures.py` and
  `python -m chamberbench.analysis` both read `CHAMBERBENCH_FIGURE_DIR`,
  defaulting to `PROJECT_ROOT / "figures"`; landing a figure in
  `archive/figures/` needs `CHAMBERBENCH_FIGURE_DIR=archive/figures` set
  explicitly. `chamberbench.calibration` is the exception -- its
  `FIGURES_DIR` is hardcoded to `archive_dir() / "figures"`, no override
  needed.

## Two things a reproduction will hit immediately

### 1. The archive records no library version, and it predates a tool-surface change

Every summary file in the archive (`baseline_chamber.json`,
every `latest_chamber*.json`) has `datasheetindex_version: null`.
`chamber-run` (`src/chamberbench/harness/run.py`,
`_datasheetindex_version()`) now stamps the resolved `datasheetindex`
distribution version into every fresh run's summary for exactly this reason
-- the field exists *because* the archive it is documenting doesn't have it.

`docs/reproducing.md` recovered the actual versions from lockfile history:
**0.13.0** for the Qwen legs (2026-05-21/22) and **0.14.0** for Claude and
GPT-5.1 (2026-06-05). This repository's `main` is well past both. The tool
surface changed between 0.31 and 0.34: `build_datasheet` began nudging
`search_text` on LLM-reconstructed tables of contents, and running headers
and footers are now stripped from page-matched text. **A fresh run's
`n_tool_calls_by_tool` is not comparable to the archive's** -- the dispatch
counts the paper reports are counts of exactly the kind that boundary
changed. This is disclosed in `docs/reproducing.md`'s "Version pinning"
section; repeated here because it is the single fact most likely to make a
reproduction attempt look like a regression when it isn't one.

### 2. Reproduction is not byte-identical, and "not comparable" differs sharply by arm

From `archive/variance_chamber.json` (3 repeats per model, verified directly
against the file, not copied from prose elsewhere):

| Arm | Fidelity (mean +/- std, out of 25) | Claims stable / flipped | Engine errors |
|---|---|---|---|
| claudesonnet4.6 | 25.0 +/- 0.0 | 25 / 0 | 0 |
| gpt-5.1 | 25.0 +/- 0.0 | 25 / 0 | 0 |
| qwen3.6-27b | 19.0 +/- 4.0 | 13 / 12 | 17 |

For Claude and GPT-5.1 the outcome is exact across all 3 repeats -- fidelity
25/25/25, zero flipped claims, zero engine errors. For Qwen, 12 of 25 claims
flip their pass/fail outcome between repeats **on the authors' own gateway**,
with 17 engine errors across the 3 repeats. A reader attempting to reproduce
the Qwen arm and failing to match it exactly has learned very little by
itself; `docs/reproducing.md` and `archive/variance_qwen_no_think.json`
attribute the instability to a documented vLLM/reasoning-mode interaction
([QwenLM/Qwen3#1817](https://github.com/QwenLM/Qwen3/issues/1817)) rather
than to the model weights -- with reasoning disabled the same model scores
much more stably (see that row below). Treat the Claude and GPT-5.1 arms as
the ones a reproduction should match closely, and the Qwen arm as one whose
*instability itself* is the reported result, not a target to hit.

## The manifest

One row per file actually shipped under `archive/` (37 total: 33 `.json`, 3
`.jsonl`, 1 `.md`). Where the exact invocation could not be recovered, the row
says so rather than guessing -- see `variance_qwen_no_think.json`.

The six rendered figures below are **not** part of that 37 and are documented
separately; see the Figures section for why.

### A4988 fourth-component fidelity (off-corpus generalisation)

| Artifact | Producer | Command | Arm | Notes |
|---|---|---|---|---|
| `a4988_fidelity.claudesonnet4.6.json` | `scripts/fourth_component.py` | `uv run python scripts/fourth_component.py --model claudesonnet4.6 --pdf <path-to-A4988-motor-driver.pdf> --out results/a4988_fidelity.claudesonnet4.6.json` | agentic, chamber tools un-registered (no chamber protocol for a motor driver) | 12-claim `data/claims_a4988.yaml` (Allegro A4988). Fidelity only, both matchers reported. Timestamp 2026-07-26T10:37Z. **The archived commands as originally recorded did not run:** the claim file's `pdf_source` is the bare part label `"A4988"`, which is neither a URL nor a path, so every claim raised `FileNotFoundError` and was swallowed into `engine_error`. `--pdf` is now required and an unresolvable `pdf_source` fails before any billable call, the same disclosure the `wrong_content.*` rows carry for `--decoy`. Point `--pdf` at a local copy of the A4988 datasheet -- mirrored in the Causal Chambers repository as `hardware/datasheets/motor_driver.pdf`, with the revision and checksum in `docs/reproducing.md`'s corpus table. |
| `a4988_fidelity.gpt-5.1.json` | `scripts/fourth_component.py` | `uv run python scripts/fourth_component.py --model gpt-5.1 --pdf <path-to-A4988-motor-driver.pdf> --out results/a4988_fidelity.gpt-5.1.json` | agentic, chamber tools un-registered | Timestamp 2026-07-26T10:41Z. **The archived commands as originally recorded did not run:** the claim file's `pdf_source` is the bare part label `"A4988"`, which is neither a URL nor a path, so every claim raised `FileNotFoundError` and was swallowed into `engine_error`. `--pdf` is now required and an unresolvable `pdf_source` fails before any billable call, the same disclosure the `wrong_content.*` rows carry for `--decoy`. Point `--pdf` at a local copy of the A4988 datasheet -- mirrored in the Causal Chambers repository as `hardware/datasheets/motor_driver.pdf`, with the revision and checksum in `docs/reproducing.md`'s corpus table. |
| `a4988_fidelity_rerun.claudesonnet4.6.json` | `scripts/fourth_component.py` | `uv run python scripts/fourth_component.py --model claudesonnet4.6 --pdf <path-to-A4988-motor-driver.pdf> --out results/a4988_fidelity_rerun.claudesonnet4.6.json` | agentic, chamber tools un-registered | Same claim set, re-run 2026-08-21T13:58Z on a later `datasheetindex` tool surface than the 2026-07-26 pair above -- the two are not tool-call-count comparable to each other, same reason as fact 1 above. **The archived commands as originally recorded did not run:** the claim file's `pdf_source` is the bare part label `"A4988"`, which is neither a URL nor a path, so every claim raised `FileNotFoundError` and was swallowed into `engine_error`. `--pdf` is now required and an unresolvable `pdf_source` fails before any billable call, the same disclosure the `wrong_content.*` rows carry for `--decoy`. Point `--pdf` at a local copy of the A4988 datasheet -- mirrored in the Causal Chambers repository as `hardware/datasheets/motor_driver.pdf`, with the revision and checksum in `docs/reproducing.md`'s corpus table. |
| `a4988_fidelity_rerun.gpt-5.1.json` | `scripts/fourth_component.py` | `uv run python scripts/fourth_component.py --model gpt-5.1 --pdf <path-to-A4988-motor-driver.pdf> --out results/a4988_fidelity_rerun.gpt-5.1.json` | agentic, chamber tools un-registered | Re-run 2026-08-21T14:11Z; same tool-surface caveat. **The archived commands as originally recorded did not run:** the claim file's `pdf_source` is the bare part label `"A4988"`, which is neither a URL nor a path, so every claim raised `FileNotFoundError` and was swallowed into `engine_error`. `--pdf` is now required and an unresolvable `pdf_source` fails before any billable call, the same disclosure the `wrong_content.*` rows carry for `--decoy`. Point `--pdf` at a local copy of the A4988 datasheet -- mirrored in the Causal Chambers repository as `hardware/datasheets/motor_driver.pdf`, with the revision and checksum in `docs/reproducing.md`'s corpus table. |

### Primary matrix

| Artifact | Producer | Command | Arm | Notes |
|---|---|---|---|---|
| `baseline_chamber.json` | `scripts/build_chamber_baseline.py` | `uv run python scripts/build_chamber_baseline.py --results-dir archive --out archive/baseline_chamber.json --force` | agentic + baseline, 3 models (claudesonnet4.6, gpt-5.1, qwen3.6-27b) x 25 claims | **Not** `chamber-run --engine baseline` directly -- that name only produces one engine's cells for one model (see `latest_chamber.*` below); this script is a separate consolidation step that reads each model's `latest_chamber.{model}.json` plus a `snapshot_layer2_agentic.{model}.json` per model that is **not shipped** in this archive (later source wins per model on overlap). A from-scratch run today only has the `latest_chamber.{model}.json` half of each model's sources. |
| `latest_chamber.claudesonnet4.6.json` | `chamberbench.harness.run` (`chamber-run`) | `uv run chamber-run --model claudesonnet4.6 --engine agentic --out results/` and `uv run chamber-run --model claudesonnet4.6 --engine baseline --out results2/` | agentic + baseline, 25 claims each = 50 cells | The shipped file holds **both** engines' cells in one file, but `ChamberResultsCollector.write_summary()` always overwrites from an empty `records` dict per process -- two `chamber-run` invocations at the same `--out` do not merge, the second clobbers the first engine's cells. Today's CLI cannot produce this merged shape in a single invocation; reproducing it needs a manual union of the two per-engine `results` dicts. `datasheetindex_version: null` (predates the version stamp -- fact 1 above). Timestamp 2026-06-05T10:43Z. Also written as the default model, this run produces the canonical mirror below. |
| `latest_chamber.gpt-5.1.json` | `chamberbench.harness.run` (`chamber-run`) | `uv run chamber-run --model gpt-5.1 --engine agentic --out results/` + `--engine baseline --out results2/` | agentic + baseline, 25 claims each | Same both-engines-merged caveat as the Sonnet row. Timestamp 2026-06-05T12:06Z. `datasheetindex_version: null`. |
| `latest_chamber.qwen3.6-27b.json` | `chamberbench.harness.run` (`chamber-run`) | `uv run chamber-run --model qwen3.6-27b --engine agentic --out results/` + `--engine baseline --out results2/` | agentic + baseline, 25 claims each | Same both-engines-merged caveat. Timestamp 2026-05-21T18:00Z -- earlier and on a different date than Claude/GPT-5.1 (matches `docs/reproducing.md`'s note that the Qwen arm was run separately, against a gateway that served qwen3.6-27b). |
| `latest_chamber.qwen3.5-27b.json` | `chamberbench.harness.run` (`chamber-run`) | `uv run chamber-run --model qwen3.5-27b --engine agentic --out results/` + `--engine baseline --out results2/` | agentic + baseline, 25 claims each | `qwen3.5-27b` is the **retired** alias (`qwen3.6-27b` replaced it on the gateway these runs used); not one of `build_chamber_baseline.py`'s 3 tracked models, so this is "further extractions outside the published matrix" (`archive/README.md`). Earliest timestamp in the archive, 2026-05-18T12:54Z. Same both-engines-merged caveat as above. |
| `latest_chamber.json` | `chamberbench.harness.run` (`chamber-run`) | *No separate command.* Automatic side effect of the `claudesonnet4.6` run above. | -- | `ChamberResultsCollector.write_summary()` writes this byte-identical mirror whenever `model == CHAMBER_DEFAULT_MODEL` ("claudesonnet4.6"), "so `chamberbench.quality_gates` / `chamberbench.classifier` keep working without a `--model` argument." Confirmed byte-identical timestamp (2026-06-05T10:43Z) to `latest_chamber.claudesonnet4.6.json`. |
| `latest_traces.claudesonnet4.6.jsonl` | `chamberbench.harness.run` (`chamber-run`) | *No separate command.* Automatic side effect of the `claudesonnet4.6` run above (`ChamberResultsCollector.trace_sink()`). | -- | One `TraceStep` JSON line per agent step, plus a leading `session_start` sentinel. `close()` also mirrors this to an unsuffixed `latest_traces.jsonl` when `model == CHAMBER_DEFAULT_MODEL`, but no such unsuffixed file is shipped in this archive -- either it was not committed, or that mirror was added after this run. |
| `latest_traces.gpt-5.1.jsonl` | `chamberbench.harness.run` (`chamber-run`) | *No separate command.* Side effect of the gpt-5.1 run above. | -- | Same shape as the Sonnet trace file. |
| `latest_traces.qwen3.6-27b.jsonl` | `chamberbench.harness.run` (`chamber-run`) | *No separate command.* Side effect of the qwen3.6-27b run above. | -- | Same shape. |

### Corrupt-success arms (the silent-failure detector evidence)

| Artifact | Producer | Command | Arm | Notes |
|---|---|---|---|---|
| `closed_book.claudesonnet4.6.json` | `scripts/null_tool.py` | `uv run python scripts/null_tool.py --mode closed-book --model claudesonnet4.6 --out results/closed_book.claudesonnet4.6.json` | closed-book: all datasheet tools un-registered | Measures P(memory correct) -- the corrupt-success class's base rate. Timestamp 2026-07-26T08:14Z. |
| `closed_book.gpt-5.1.json` | `scripts/null_tool.py` | `uv run python scripts/null_tool.py --mode closed-book --model gpt-5.1 --out results/closed_book.gpt-5.1.json` | closed-book | Timestamp 2026-07-26T08:15Z. |
| `closed_book.qwen3.6-27b.json` | `scripts/null_tool.py` | `uv run python scripts/null_tool.py --mode closed-book --model qwen3.6-27b --out results/closed_book.qwen3.6-27b.json` | closed-book | `qwen_enable_thinking: false` recorded in the file -- the script auto-forces `CHAMBER_QWEN_ENABLE_THINKING=false` for any qwen model unless the env var is already set, per the module docstring's "QWEN NOTE". Timestamp 2026-07-26T14:30Z. |
| `null_tool_injection.claudesonnet4.6.json` | `scripts/null_tool.py` | `uv run python scripts/null_tool.py --mode null --model claudesonnet4.6 --out results/null_tool_injection.claudesonnet4.6.json` (`--mode` defaults to `null`) | Arm C null-navigation: tools registered, calls return the tool's own natural empty-result message | Timestamp 2026-07-26T07:19Z. The shipped file's `"mode"` key is JSON `null`, not the string `"null"` the current script always writes -- an earlier revision of this script apparently didn't populate the field the same way. A fresh run's JSON will differ in this one field even though the run itself reproduces faithfully. |
| `null_tool_injection.gpt-5.1.json` | `scripts/null_tool.py` | `uv run python scripts/null_tool.py --mode null --model gpt-5.1 --out results/null_tool_injection.gpt-5.1.json` | Arm C null-navigation | Timestamp 2026-07-26T07:26Z. Same `"mode": null` schema note as the Sonnet row. |
| `null_tool_injection.qwen3.6-27b.json` | `scripts/null_tool.py` | `uv run python scripts/null_tool.py --mode null --model qwen3.6-27b --out results/null_tool_injection.qwen3.6-27b.json` | Arm C null-navigation | `qwen_enable_thinking: false` (auto-forced, default). Timestamp 2026-07-26T14:49Z -- run later the same day as the pair above, after (apparently) the `"mode"` field was added to the output schema: this file correctly shows `"mode": "null"` as a string. |
| `null_tool_injection.qwen3.6-27b.thinking_on.json` | `scripts/null_tool.py` | `CHAMBER_QWEN_ENABLE_THINKING=true uv run python scripts/null_tool.py --mode null --model qwen3.6-27b --out results/null_tool_injection.qwen3.6-27b.thinking_on.json` | Arm C null-navigation, reasoning explicitly re-enabled | Explicit override of the auto-force-false default, to demonstrate the QwenLM/Qwen3#1817 reasoning-mode tool-call bug the "off" arm above avoids. `qwen_enable_thinking: true` in the file. Timestamp 2026-07-26T07:20Z (same early batch as the Claude/GPT null_tool_injection files -- `"mode"` reads JSON `null`, not the string, same schema-era note). |
| `wrong_content.claudesonnet4.6.json` | `scripts/null_tool.py` | `uv run python scripts/null_tool.py --mode wrong-content --decoy <path-to-A4988-motor-driver.pdf> --model claudesonnet4.6 --out results/wrong_content.claudesonnet4.6.json` | wrong-content: tools serve a *different* datasheet | Decoy recorded in the file as `eval/chamber/datasheets/motor_driver.pdf`, the pre-port private-repo layout; that path does not exist in this release's `corpus/`. Point `--decoy` at wherever the A4988 PDF has been fetched (see `docs/reproducing.md`'s corpus table for the part/checksum). Timestamp 2026-07-26T08:32Z. |
| `wrong_content.gpt-5.1.json` | `scripts/null_tool.py` | `uv run python scripts/null_tool.py --mode wrong-content --decoy <path-to-A4988-motor-driver.pdf> --model gpt-5.1 --out results/wrong_content.gpt-5.1.json` | wrong-content | Timestamp 2026-07-26T08:41Z. Same decoy-path caveat. |
| `wrong_content.qwen3.6-27b.json` | `scripts/null_tool.py` | `uv run python scripts/null_tool.py --mode wrong-content --decoy <path-to-A4988-motor-driver.pdf> --model qwen3.6-27b --out results/wrong_content.qwen3.6-27b.json` | wrong-content | `qwen_enable_thinking: false` (auto-forced, default). Timestamp 2026-07-26T15:12Z. Same decoy-path caveat. |
| `fault_injection.json` | `scripts/fault_injection.py` | `uv run python scripts/fault_injection.py --out results/fault_injection.json` | Claude Sonnet 4.6 only (hardcoded, not a `--model` flag). Arm A: planted F1 (tool-bypass) + F5 (verification-skipped), fidelity verdict reused from the clean run. Arm B: clean control, cells reused from `archive/baseline_chamber.json` | Backs the detector's recall (Arm A, should be near-100%) and false-positive rate (Arm B, should be near-0%) claims for Claude only. |
| `fault_injection_gpt-5_1.json` | `scripts/fault_injection_multimodel.py` | `uv run python scripts/fault_injection_multimodel.py --model gpt-5.1 --out results/` | Same F1/F5 planted faults as `fault_injection.py`, GPT-5.1 | `--out` is a **directory**; the filename is built at runtime as `f"fault_injection_{args.model.replace('.', '_')}.json"`, which is why the archived name uses an underscore (`gpt-5_1`) where the model alias itself uses a dot (`gpt-5.1`) -- documented in `archive/README.md`'s naming-inconsistency note. A literal grep for `fault_injection_gpt-5.1.json` (with a dot) would find nothing. |
| `fault_injection_qwen3_6-27b.json` | `scripts/fault_injection_multimodel.py` | `CHAMBER_QWEN_ENABLE_THINKING=false uv run python scripts/fault_injection_multimodel.py --model qwen3.6-27b --out results/` | Same F1/F5 planted faults, qwen3.6-27b | Same runtime-constructed filename note (`qwen3_6-27b` vs the alias `qwen3.6-27b`). `CHAMBER_QWEN_ENABLE_THINKING=false` is not auto-forced by this script (unlike `null_tool.py`); the module docstring's own example command sets it explicitly to avoid the reasoning-mode tool-call-bug error storm. |

### Reproducibility-perturbation

| Artifact | Producer | Command | Arm | Notes |
|---|---|---|---|---|
| `perturbation_sweep.json` | `scripts/perturbation.py` | `uv run python scripts/perturbation.py --out results/perturbation_sweep.json --end-to-end` | Tier A: pure claimed-bound sweep (no model call) + Tier B: live agentic run on the perturbed DPS310 datasheet | The `tier_b` key is populated in the shipped file, which is why `--end-to-end` is part of the command (omitting it produces Tier A only). The perturbed PDF is built fresh from the original DPS310 datasheet with pymupdf (`--build-only`) rather than shipped, so no altered manufacturer datasheet circulates as genuine. Timestamp 2026-06-03T20:32Z. |

### Variance (Table 1's per-run spread)

| Artifact | Producer | Command | Arm | Notes |
|---|---|---|---|---|
| `variance_chamber.mainrun.bak.json` | `scripts/variance.py` | `uv run python scripts/variance.py --out results/variance_chamber.mainrun.bak.json` (default: all 3 models, `--fresh-repeats 2`) | agentic, 3 models x 3 repeats (1 imported from `archive/baseline_chamber.json` + 2 live) | Main 3-model variance run, timestamp 2026-05-22T13:53Z. Superseded input to the May consolidation below -- kept for provenance, **not** the published run (`archive/README.md`). |
| `variance_gpt_rerun.json` | `scripts/variance.py` | `uv run python scripts/variance.py --models gpt-5.1 --fresh-repeats 2 --out results/variance_gpt_rerun.json` | agentic, gpt-5.1 x 3 repeats | Re-run because the main run above used a stale 15-turn budget for GPT-5.1 (`scripts/consolidate_variance.py`'s docstring). `CHAMBER_MODEL_CONFIG["gpt-5.1"]["max_turns"]` in this release is already `30` -- the fix is now the permanent default, so no special override is needed to reproduce this leg going forward. Timestamp 2026-05-22T14:09Z. |
| `variance_qwen_r3.json` | `scripts/variance.py` | `uv run python scripts/variance.py --models qwen3.6-27b --fresh-repeats 1 --out results/variance_qwen_r3.json` | agentic, qwen3.6-27b x 1 fresh repeat (+ 1 imported) | Retry of the repeat-3 cells that failed with HTTP 503 ("all pods are down") when the self-hosted qwen vLLM pod dropped mid-run. `n_repeats: 2` in the file (1 imported + 1 live) confirms `--fresh-repeats 1`. Timestamp 2026-05-22T14:33Z. |
| `variance_chamber.json` | `scripts/consolidate_variance.py` | `uv run python scripts/consolidate_variance.py` | agentic, 3 models x 3 repeats | **Running this command against the three archived inputs above does NOT reproduce this file.** Per the script's own refusal-guard comment: the shipped file is dated 2026-06-05 with no `_consolidation` key, while the three inputs above are dated 2026-05-22 -- the Claude and GPT-5.1 legs of the *published* file come from a later re-run that is itself not archived. The script refuses to overwrite the shipped file without `--force` specifically because running it would silently move the paper's numbers (documented example: GPT-5.1 mean latency 236s -> 133s) back to the superseded inputs. Verified directly against the file: claudesonnet4.6 fidelity 25.0+/-0.0 (25 stable/0 flipped, 0 engine errors), gpt-5.1 25.0+/-0.0 (25/0, 0), qwen3.6-27b 19.0+/-4.0 (13/12, 17 engine errors) -- see fact 2 above. |
| `variance_qwen_no_think.json` | **Not recovered.** | **Provenance NOT recovered.** No script in this release reads `CHAMBER_QWEN_ENABLE_THINKING` from within `scripts/variance.py` itself -- only `src/chamberbench/harness/anthropic_path.py` reads the env var at call time, so this file was almost certainly produced by `uv run python scripts/variance.py --models qwen3.6-27b ...` with `CHAMBER_QWEN_ENABLE_THINKING=false` set in the shell environment. The exact `--fresh-repeats` value, `--out` path, and env invocation are not recorded anywhere in this release: no commit trail, no note inside the file itself. | agentic, qwen3.6-27b, reasoning disabled | Verified directly against the file rather than trusted from prose elsewhere: **4** repeats (1 imported from `baseline_chamber.json` + 3 live, not 3 total), fidelity per_run `[23, 25, 24, 24]`, mean 24, 23 stable / 2 flipped claims, 2 engine errors total. `archive/README.md` and `docs/reproducing.md` used to gloss this as "the 25/24/24 result" -- an approximation that dropped the 4th repeat and, with it, the lowest score; both now quote the `aggregate` block. Cited as the evidence that Qwen's instability (fact 2 above) is a serving-stack artifact, not a property of the weights. |

### Supporting artifacts

| Artifact | Producer | Command | Arm | Notes |
|---|---|---|---|---|
| `classifier_auto.claudesonnet4.6.json` | `scripts/prepare_gold_labelling.py` | `uv run python scripts/prepare_gold_labelling.py --seed 0 --n 30 --rebuild-auto-labels` | n/a (auto-applied manual-rule classifier labels) | **Not faithfully reproducible from what is shipped.** Per the script's own comment: these files are built by merging in `snapshot_layer2_traces.{model}.jsonl`, which is **not part of this release**; regenerating without it silently yields a truncated attribution list rather than erroring. Without `--rebuild-auto-labels` the script leaves the shipped files untouched and prints why. |
| `classifier_auto.gpt-5.1.json` | `scripts/prepare_gold_labelling.py` | same command | n/a | Same missing-snapshot caveat. |
| `classifier_auto.qwen3.6-27b.json` | `scripts/prepare_gold_labelling.py` | same command | n/a | Same missing-snapshot caveat. |
| `sampled_cells.json` | `scripts/prepare_gold_labelling.py` | same command | n/a | Companion artifact from the same run: the 30 sampled `(claim_id, engine, model)` tuples (10 per model, stratified by component) plus the random seed, written alongside the `classifier_auto.*` files. |
| `classifier_agreement.md` | `scripts/compute_classifier_agreement.py` | `uv run python scripts/compute_classifier_agreement.py --force` | n/a | Default `--gold data/classifier_gold.yaml` (single annotator). The shipped file has exactly one "Cohen's kappa" line and no inter-annotator section, so it was generated against one `--gold`, not the two-annotator `--gold data/classifier_gold.yaml --gold data/classifier_gold.annotator2.yaml` invocation that would add that section. `--force` is required because the file already exists (otherwise the script writes `classifier_agreement.regenerated.md` instead). Generated timestamp inside the file: 2026-08-26T09:11:46Z. |

### Figures (`archive/figures/`)

**These six figures are not shipped in the archive.** They are derived
artifacts, fully regenerable from the tracked `archive/*.json` by the commands
below, and matplotlib does not render them bit-for-bit reproducibly across
versions -- a committed PNG would drift against a reader's regenerated one and
invite a false "this does not match" conclusion. The archive holds the model
output; these are visualisations of it. The section is kept because knowing
which producer owns which figure, and in what order to run them, is exactly
what a reproduction needs. Expect `archive/figures/` to be absent in a fresh
clone; the producers create it.

| Artifact | Producer | Command | Arm | Notes |
|---|---|---|---|---|
| `cost_latency_scatter.png` | `scripts/render_paper_figures.py` | `CHAMBERBENCH_FIGURE_DIR=archive/figures uv run python scripts/render_paper_figures.py` | n/a | Reads `archive/baseline_chamber.json` and `archive/variance_chamber.json`. This script's own docstring says it rebuilds this figure (and the two below) because `chamberbench.analysis`'s version was found stale (a pre-audit 60-cell matrix); run this **after** `python -m chamberbench.analysis` if reproducing the full figure set, so this script's corrected version is the one left on disk. Also writes `reproducibility_perturbation.png` into the same directory; that one was dropped from the paper, so unlike the six rows here it is not documented below either. (None of the seven is shipped -- see this section's opening.) |
| `fidelity_heatmap.png` | `scripts/render_paper_figures.py` | `CHAMBERBENCH_FIGURE_DIR=archive/figures uv run python scripts/render_paper_figures.py` | n/a | Same run as the row above; one invocation produces both files (plus `tool_dispatch_heatmap.png` and the unshipped `reproducibility_perturbation.png`). |
| `tool_dispatch_heatmap.png` | `scripts/render_paper_figures.py` | `CHAMBERBENCH_FIGURE_DIR=archive/figures uv run python scripts/render_paper_figures.py` | n/a | Same run as the two rows above. |
| `engagement_over_revisions.png` | `chamberbench.analysis` | `CHAMBERBENCH_FIGURE_DIR=archive/figures uv run python -m chamberbench.analysis` | n/a | This module also (re)writes `fidelity_heatmap.png` / `tool_dispatch_heatmap.png` / `cost_latency_scatter.png`, but those three are the "stale" versions `render_paper_figures.py` supersedes -- this is the **only** one of the four this module produces that is not later superseded, so its version is the one a reproduction should end up with. (The archive keeps none of them; see this section's opening.) Run this module first, then `render_paper_figures.py`, if reproducing the whole set from a clean `archive/figures/`. |
| `confidence_distribution.png` | `chamberbench.calibration` | `uv run python -m chamberbench.calibration` | n/a | Writes directly into `archive/figures/` -- `FIGURES_DIR = archive_dir() / "figures"` is hardcoded in this module (unlike `analysis.py` / `render_paper_figures.py`, no `CHAMBERBENCH_FIGURE_DIR` override needed). Computed over the 95 `status == "ok"` cells in `archive/baseline_chamber.json`. |
| `confidence_vs_effort.png` | `chamberbench.calibration` | `uv run python -m chamberbench.calibration` | n/a | Same invocation as the row above; one run writes both confidence figures. |

## What a reproduction costs

The archive carries per-cell token usage, so the cost of a full
reproduction is computable offline, from the archive alone -- no
credentials, no network. `scripts/reproduction_cost.py` does this;
`uv run python scripts/reproduction_cost.py` produced the numbers below
against this release's shipped archive.

**These totals cover both engines** -- agentic and baseline, 25 claims
each, 50 cells per arm -- because a reader reproducing "the full
experiment" runs both; a reader reproducing only the agentic arm should
budget roughly half the input-token figure below (baseline cells carry
little tool-call overhead, so the split is not exactly 50/50, but agentic
is the larger half). `cache_read_tokens` is 0 for all three arms in this
archive; do not read that as a script defect, it is what the archived runs
actually recorded.

```
Reproduction cost per arm -- both engines (agentic + baseline), 25 claims each = 50 cells per arm

claudesonnet4.6      in=  5101728 out=  126716 cache_read=        0
gpt-5.1              in=  3255552 out=  209303 cache_read=        0
qwen3.6-27b          in=  3339776 out=  249299 cache_read=        0
```

These are the **archived run's actuals**, not an estimate -- a fresh run
will differ, and by how much differs sharply by arm. As an illustration
only, marked with a date rather than taken as current pricing: at
Anthropic's public list price on 2026-08-26 (\$3/M input, \$15/M output),
the `claudesonnet4.6` totals above would be on the order of \$17 for input
tokens and \$2 for output tokens, call it roughly \$19 for one full run of
that arm alone -- illustrative arithmetic a reader should redo against
their own provider's current rates, not a number to cite.

Token cost is only half the budgeting question; the other half is how many
runs it takes to get a result worth trusting. Section 6.3 of the design
spec puts this as three tiers of reproduction, in increasing strength, and
is explicit that only the first is guaranteed:

1. **Structural** -- the run completes, same schema, same 25 claims
   covered. Deterministic and cheap for a reader to check.
2. **Verdict-level** -- regrade the new run through Tier 1's
   `regrade_archive.py` and compare verdicts against the archive. This is
   where the two tiers join.
3. **Statistical** -- fidelity within the archive's own measured spread.
   From `archive/variance_chamber.json`, 3 repeats, fidelity as a count out
   of 25:

   | Arm | Fidelity | Claims stable / flipped | Engine errors |
   |---|---|---|---|
   | `claudesonnet4.6` | 25.0 +/- 0.0 | 25 / 0 | 0 |
   | `gpt-5.1` | 25.0 +/- 0.0 | 25 / 0 | 0 |
   | `qwen3.6-27b` | 19.0 +/- 4.0 | 13 / 12 | 17 |

   For Claude and GPT the outcome is *exact* across repeats, so tier 3
   collapses into an equality test (25/25, zero flipped) rather than a
   band. Only the Qwen arm needs a tolerance, and it needs a wide one.

Only tier 1 is guaranteed. Tiers 2 and 3 are how strong a claim a reader
can make about matching the paper, not something the harness enforces. For
Claude and GPT-5.1, tier 3 is not a tolerance band to fall inside -- it is
an equality test, because the archive shows zero spread across all three
repeats. For Qwen it is a genuine band, and a wide one: 12 of the 25
claims flip their pass/fail outcome between runs **on the authors' own
gateway**, with 17 engine errors across 3 repeats. A reader budgeting for
the Qwen arm should plan for retries and should not expect their run to
match either the archive or a second run of their own -- the instability
itself is the reported result for that arm, not a target to hit. See fact
2 above for the same table in the context of what changed since the
archive was produced.
