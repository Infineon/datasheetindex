# Reproducing the published numbers

Two questions get confused with each other, and they have different answers:

1. **Can you check our arithmetic?** Almost completely, offline, with no API
   key and no PDFs. Everything in this repository serves that question. The
   qualification is real and worth reading: the variance repeats store
   verdicts without the extractions that produced them, so Table 1's spread
   can be recomputed but not re-graded. See
   [Attacking the grading surface](#attacking-the-grading-surface).
2. **Can you re-run our agent and get our numbers?** Partly, and less than you
   might expect — for reasons that are mostly not about code. See
   [Why a re-run will differ](#why-a-re-run-will-differ).

Keeping these apart matters, because the first is the stronger claim and it is
the one this release actually supports.

## What regenerates what

Every command below reads only `archive/` and `data/`, except the one marked
**net** — it fetches the public Causal Chambers dataset (~1 MB, cached under
`CHAMBER_CACHE_ROOT`, default `/tmp/cc_data`). Nothing calls a model.

| Paper artifact | Command |
|---|---|
| Table 1, cross-model results | `python scripts/render_paper_tables.py` |
| Figures (dispatch, fidelity, cost/latency, perturbation) | `python scripts/render_paper_figures.py` |
| Classifier agreement, Cohen's κ | `python scripts/compute_classifier_agreement.py` |
| Blind re-derivation scoring | `python scripts/score_rederivation.py --derivation data/rederivation.anna.yaml` |
| Strict-fidelity re-score | `python scripts/strict_fidelity_rescore.py` |
| Reproducibility decomposition | `python scripts/repro_inconclusive_taxonomy.py` |
| Detector false-positive scan | `python scripts/silent_failure_fp_scan.py` |
| Re-grade the archive under the current claim set | `python scripts/regrade_archive.py` |
| Natural-divergence scan **(net)** | `python scripts/scan_natural_divergence.py` |
| Cost summary | `python scripts/chamber_cost_summary.py` |
| Baseline vs agentic | `python scripts/baseline_vs_agentic.py` |

**Create the cache directory first** — `causalchamber` does not create it, so a
first run otherwise reports that the dataset could not be fetched when the
network was fine:

```bash
mkdir -p /tmp/cc_data     # or $CHAMBER_CACHE_ROOT
```

The **(net)** command exits **1** if the Causal Chambers dataset cannot be
fetched, and prints which claims were skipped. That is deliberate: the claims
needing chamber data are exactly the ones that can produce a non-inconclusive
verdict, so an unreachable dataset must not be reported as "no failures found".
An exit of 1 there means *incomplete*, not *failed*.

`regrade_archive.py` uses the same convention: **exit 2** means your claim set
did not cover every claim in the archive, so the counts describe a subset. It
names the claims it could not grade. Exit 1 means nothing was gradable at all;
disagreements alone are a finding, not an error, and exit 0.

`tests/test_reproduces_paper.py` pins the headline numbers directly, so a
drift in the archive or the grading surface fails a test rather than quietly
changing a table.

## Attacking the grading surface

The interesting objection to this work is not "did they compute the mean
correctly" but "were the pass criteria written to fit the answers". The
criteria are two hand-written fields per claim: `value_contains` (substrings
the answer must include) and `confidence_min` (a floor on the agent's
self-reported confidence).

Re-score the same archive under your own version of those fields:

```bash
CHAMBERBENCH_DATA_DIR=/path/to/your/claims python scripts/regrade_archive.py
```

`regrade_archive.py` walks the archived extractions back through
`evaluate_case` — the same function that produced the published verdicts — so
the matcher is held fixed and only your claim set varies.

**The renderers do not work this way, and it matters.**
`render_paper_tables.py` and the other reporting scripts print verdicts that
were computed at run time and stored in the archive; pointing
`CHAMBERBENCH_DATA_DIR` at them changes nothing. The scripts that genuinely
honour it are `regrade_archive.py`, `score_rederivation.py` and
`strict_fidelity_rescore.py`.

**Only part of the archive is re-gradable.** `baseline_chamber.json` retains
`claim_result` (the raw extraction) for 149 cells. The variance repeats store
verdicts only — so repeats 2 and 3, which produce Table 1's per-run spread and
the Qwen instability headline, **cannot be re-graded from what is shipped**.
Those verdicts have to be taken on trust, and the earlier answer to "can you
check our arithmetic" is qualified by exactly that.

Run with no override and you should see **148 agree, 1 flip**, on
`acs70331-saturation-low`. It is expected, and the reason is worth
understanding because it is about the *matcher*, not the claim set.

That claim was gated on the symbol `VSAT_LOW`. At the time the archive was
produced, the string a needle was matched against still included the model's
own quoted span, so the symbol was reachable — five of the claim's six cells
quoted a span containing it and passed; the sixth did not and failed. The span
was then removed from the matched text, precisely so that quoting the right
table row could not by itself satisfy a value check. That made `VSAT_LOW`
unsatisfiable, and the needle was replaced with the value and its unit.

So the archive is internally consistent: every one of those six verdicts is
exactly predicted by whether that cell's quoted span contained the symbol. What
the flip measures is the grading-surface change, not a stale cell. The paper
documents the repair in its appendix on the claim list.

[`annotator_guide.md`](annotator_guide.md) is the instruction set the
independent annotator worked from. A disagreement is a finding; please report
it rather than assuming it is a bug.

## Version pinning — read this before re-running

**The published runs used `datasheetindex` 0.13.0 (Qwen, 2026-05-21/22) and
0.14.0 (Claude and GPT-5.1, 2026-06-05).** This repository's `main` is well
past both.

That gap is not cosmetic. The tool surface changed between 0.31 and 0.34:
`build_datasheet` began nudging `search_text` on LLM-reconstructed tables of
contents, and running headers and footers are now stripped from page-matched
text. **Tool-call counts and navigation traces are not comparable across that
boundary**, and the dispatch-level results in the paper are counts of exactly
that kind.

So: re-running at `main` is a legitimate experiment, and an interesting one.
It is not a reproduction of the paper, and should not be reported as one.

> **Known gap.** The archive does not record the `datasheetindex` version that
> produced it; the versions above were recovered from lockfile history. Runs
> produced from here on should stamp the resolved version into the result file.

## The corpus

Four manufacturer datasheets, not redistributed here. Three are third-party
and one is Infineon's own; all four are publicly downloadable from the
manufacturer. Identified by part, revision and checksum so a fetched copy can
be verified against the one we used:

| Role | Part | Vendor | Rev | Pages | SHA-256 (first 16) |
|---|---|---|---|---|---|
| barometer | DPS310 | Infineon | V1.1 | 41 | `440d2b01d1e9851a` |
| light sensor | Si115x | Silicon Laboratories | 1.4 | 65 | `d3e0f16e6fc95572` |
| current sensor | ACS70331 | Allegro MicroSystems | 2 | 26 | `c5e78d82d561d3e2` |
| motor driver (off corpus) | A4988 | Allegro MicroSystems | 5 | 20 | `c7341f95ab7d571d` |

All four are mirrored in the Causal Chambers repository under
`hardware/datasheets/` (`barometer.pdf`, `light_sensor.pdf`,
`current_sensor.pdf`, `motor_driver.pdf`), which is the easiest place to fetch
them; the checksums above are what you should get. They are of course also
available from each manufacturer by part number.

**If a checksum does not match, the vendor has reissued the document.** That is
expected over time and is not a failure of the benchmark; it does mean the
values in a re-run may legitimately differ from ours, and the claim set may
need re-deriving against the new revision.

The perturbed barometer datasheet used for the controlled-perturbation
experiment is a *derivative* of the DPS310 document with one bound altered. It
is regenerated from the original rather than shipped, so that no altered
manufacturer datasheet circulates as if it were genuine.

## Why a re-run will differ

Three reasons, in decreasing order of how much they should worry you.

**The tool surface moved.** See the version section above. This is the big one,
and it is entirely our doing.

**Model identity.** The runs were made through an internal LiteLLM gateway, so
the archive records gateway aliases (`claudesonnet4.6`, `gpt-5.1`,
`qwen3.6-27b`) rather than provider snapshot IDs. Re-running against the public
APIs requires choosing snapshots, and snapshot retirement will eventually make
the originals unavailable. A full re-run of the two frontier legs — 25 claims,
both engines, one repeat — cost about **$23** at list prices, so this is cheap
to attempt.

**Qwen is a serving-stack result, and that is the point.** The Qwen3.6-27B
instability the paper reports is a documented vLLM/reasoning-mode interaction
([QwenLM/Qwen3#1817](https://github.com/QwenLM/Qwen3/issues/1817)), not a
property of the weights: with reasoning disabled the same model passes 25/24/24.
Qwen3.6-27B is open-weights and widely hosted, so the leg can be re-run — but
**on a different serving stack the instability may well not appear, and that
outcome supports the paper's claim rather than contradicting it.** The paper
argues the failure belongs to the deployment, and a deployment that does not
reproduce it is evidence for exactly that.

## What is not here

The agent harness that produced the archive. It is not needed to check any
number reported in the paper, and releasing it is tracked separately. Both
dispatch-level detector rules are predicates over the `datasheetindex` tool
surface rather than over anything private, so the signals the paper recommends
can be implemented from the library alone — they are a few dozen lines on top
of it, and `chamberbench/silent_failure.py` is that implementation.
