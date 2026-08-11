# Running header/footer detection in the page-matched text file

Status: design approved, not yet implemented
Date: 2026-08-11
Target version: 0.33.0

## Problem

Every page of a datasheet carries running furniture -- a header naming the part,
a footer with the document title, a revision string and a page number. The
page-matched text file reproduces all of it, once per page, and every consumer
reads it.

The cost is not primarily tokens. It is search precision. Measured on the bundled
PSoC 6 datasheet (134 pages), before and after the algorithm this spec specifies
(counted with the result cap lifted, so the "before" figures are not truncated;
an agent calling `search_text` sees the default cap of 200):

| query | before | after |
|---|---|---|
| `PSOC` | 209 -- over the agent's 200 cap | 76 |
| `Datasheet` | 138 | 6 |
| `Rev. *S` | 133 | 1 |
| `002-23185` | 133 | 1 |

An agent searching for the part family gets a wall of identical hits and the cap
evicts the real ones.

The text file itself goes from 200,584 to 193,020 characters -- 7,564 removed,
**3.8%, ~1,891 tokens** -- across 265 dropped blocks.

Every figure in this spec was produced by a standalone prototype of the
algorithm, not by the integrated code. The block set and the decisions are the
same, but the prototype orders blocks by a plain `(y, x)` sort rather than
`_extract_page_text`'s column-aware order. That cannot change *which* blocks are
dropped, so the counts hold; re-confirm them during implementation before pinning
them in tests, and treat any disagreement as a finding rather than an excuse to
adjust the assertion.

An earlier draft of this spec claimed 5.6% / 11,044 characters. That number came
from a *line-level* scan whose recurring set included `Table #` (89 occurrences)
-- real captions this design deliberately keeps. The honest figure is 3.8%. The
benefit was always search precision rather than token volume, and that survives:
`Datasheet` falls 138 -> 6 and `PSOC` no longer reaches the cap.

A second motivation is downstream. Running furniture is the dominant obstacle to
a future non-LLM outline detector: on `motor_driver.pdf`, 91 of ~105
larger-than-body lines are the repeated title block, so a heading detector run
against today's text would emit twenty identical `A4988` nodes. Header/footer
detection is a prerequisite for that work, not a sibling of it.

## Why not the layout engine

`pymupdf.layout` already classifies `page-header` / `page-footer` blocks, and
0.32.0 uses exactly that to clean `extract_table_markdown`. It cannot serve the
text file:

- it costs ~0.95s/page (~128s for the 134-page PSoC against a ~8s build);
- it lives behind the optional `[layout]` extra (~49MB of ONNX models), which a
  plain `uv sync` deliberately excludes.

The text file must be built on the default lane. So this is a native PyMuPDF
detector -- and the layout model becomes the *oracle* we validate it against
rather than the mechanism we ship.

## Prior art: is there something to use instead?

The canonical algorithm is Lin, *Header and footer extraction by page-association*
(SPIE 2003): consider a page in the context of its neighbours and score candidate
lines by position, font and text similarity. The design below is a simplified
Lin, so this is an implementation of the standard method rather than a new one.

Surveyed implementations, against our constraints (default lane, no new runtime
dependency, must not delete table captions):

| candidate | verdict |
|---|---|
| **refinedoc** 1.0.1, Apache-2.0, pure stdlib, page-association by name | **Tested and rejected.** Works on text lines with no coordinates, so it cannot apply a position band. On the PSoC it classifies `Table 8 (continued) Multiple alternate functions` and three further `Table N` captions as *headers* -- it would delete exactly the captions `TocNode.continued_tables` is built from. Also inconsistent (header found on 75/134 pages, footer on 130/134), slow (7.15s on the PSoC, comparable to our whole build), and it `print()`s warnings to stdout, which is the MCP stdio JSON-RPC channel |
| `pdf_header_and_footer_detector` | Adds pdfminer; a script, not a maintained package |
| `pdf-parser-header-footer` 0.2.0 | 8 dependencies for one guarded function |
| `docling` (RT-DETR, DocLayNet classes), `unstructured` (116 deps) | Same ML class as `pymupdf.layout`, which we already have optionally and already rejected for the default lane on cost |
| `pymupdf4llm` `header=False, footer=False` | Already adopted in 0.32.0 for `extract_table_markdown`; layout-only, so unavailable here |

Nothing is reusable. The block-level position band is the specific thing the
pure-text implementations lack, and it is what protects table captions.

## LLM-driven detection: evaluated and rejected

Worth testing, because the task needs only the *distinct* candidate strings and
their page counts -- not the document -- so a prompt is 345-5,678 tokens
regardless of length, and latency was 1-2s. Cost is genuinely negligible; that
part of the hypothesis held.

Measured against the deterministic rule on five documents, using the self-hosted
`qwen3.6-27b` on the prod gateway, plus `gpt-4.1` and `mixtral` for contrast:

- **It agrees exactly with the deterministic rule on simple documents** (PSoC 6,
  TI PCN).
- **It finds real furniture we miss.** On tcan1044a-q1: `TCAN1044A-Q1 SLLSFJ3D -
  AUGUST 2023 - REVISED OCTOBER 2024` (14/42) and
  `Copyright (c) 2024 Texas Instruments Incorporated Submit Document Feedback 3`
  (13/42). TI alternates header layout by odd/even page, so each variant sits
  near a third of pages and falls under our 0.5 threshold.
- **Its precision collapses on complex documents.** On the same document
  `qwen3.6-27b` flags **73 of 198** candidates, including
  `PARAMETER TEST CONDITIONS MIN TYP MAX UNIT` (a table header row, 4 pages),
  `MIN NOM MAX UNIT`, section headings (`PACKAGE OUTLINE`, `EXAMPLE BOARD
  LAYOUT`) and diagram labels (`CANH`, `RXD`, `System Controller`, `3.1 2.9`).
  `mixtral` is worse, flagging `TAPE AND REEL INFORMATION`, package dimensions
  and, on the TI PCN, the document's own title and PCN metadata. `gpt-4.1` sits
  between them, still flagging `Revision History` -- a real section heading.

Rejected for three reasons, in order of weight:

1. **The error directions are not equivalent.** Because we delete, precision is
   what matters. The deterministic rule's failures are *misses* (text kept); the
   LLM's failures are *false deletions* of table header rows and section
   headings. It is wrong exactly where documents are hardest.
2. **It would break the library's contract.** The page-matched text file is one
   of two core deliverables and must build with no credentials. Making it
   LLM-dependent contradicts "the library is a pre-processor and toolbox" and
   would put an LLM in the path of every build.
3. **Availability is not free either.** During this evaluation the staging
   `qwen3.5-27b` returned 503 (all vLLM pods down) through five retries, and
   `llama3.3-70b` / `deepseekr1-70b` were listed by `/models` but unroutable.
   A core deliverable cannot depend on that.

**The recall gap it revealed is real and stays open**, recorded here rather than
silently dropped: TI-style alternating odd/even headers are not detected. The
cheap deterministic fix (lower the threshold) was tested and is worse -- see the
threshold note below. Revisit with odd/even page-parity buckets if it matters.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Disposition | Drop furniture from the text file | Fixes `search_text`, `get_section_text` and the LLM ToC fallback at once, with no new format for consumers to learn |
| Preamble | Keeps raw, unstripped text | The architecture doc advertises "zero heuristics"; page 1 is also where recurrence has least evidence and where real prose sits at the page foot |
| Matching | Exact text plus digit masking | Catches every furniture key measured across the corpus, including revision lines and page numbers that exact matching alone would miss; deterministic and easy to reason about |
| Fuzzy matching | Excluded | A similarity threshold can delete a genuine one-off line resembling its neighbours. Fails safe by keeping text |
| Traversal | Buffer blocks in the existing single pass | A second traversal costs +22% of the scan (0.33s of 1.50s) to save ~200KB we are not short of |
| Existing library | None reusable | See "Prior art" -- the only zero-dependency candidate deletes `Table N (continued)` captions |
| LLM detection | Rejected as the mechanism | See "LLM-driven detection" -- cheap and higher-recall, but deletes table header rows and section headings, and would put an LLM in a credential-free deliverable |

## Architecture

New leaf module `core/furniture.py`, pure functions over strings and counts. It
never touches a `pymupdf.Page`, so it is testable without a PDF and cannot reach
the layout engine.

```python
normalize_key(text: str) -> str
is_candidate(text: str) -> bool
detect_furniture(page_keys: Sequence[Sequence[str]], total_pages: int) -> frozenset[str]
```

`core/textfile.py` keeps all geometry:

- `_extract_page_blocks(page) -> list[tuple[str, bool]]` returns the
  already-column-ordered blocks as `(text, banded)` pairs.
- `_extract_page_text` becomes a thin join over it, so column detection and
  reading order stay in one place and are shared.
- `scan_pages` buffers per-page blocks during its existing traversal, calls
  `detect_furniture`, then joins the survivors.

Two consequences follow from putting the strip in `scan_pages` rather than in
`_extract_page_text`:

- **`preamble.py` is unaffected without a flag or a second code path.** It calls
  `_extract_page_text` directly (`preamble.py:245`) and keeps today's behaviour.
- **`figures.caption_entries` reads the stripped text.** That is why the caption
  keyword exclusion below is load-bearing rather than defensive: without it a
  figure caption near a page edge could disappear from the figure index.

## Algorithm

**Candidate gate.** A block is a candidate only if both hold.

- *Banded*: the bbox lies wholly within the top 20% (`y1 <= 0.2h`) or bottom 20%
  (`y0 >= 0.8h`) of that page, computed against that page's own height.
- *Eligible* (`is_candidate`): at most 200 characters of raw block text (measured
  before normalization, so masking cannot change eligibility), and does not begin
  with a caption keyword (`figure`, `fig.`, `table`, `chart`, case-insensitive,
  after leading whitespace is stripped).

> **There is deliberately no line-count rule.** An earlier draft of this spec
> excluded blocks of 3 or more lines, copying PageIndex. Measured, that is wrong
> for PyMuPDF: `get_text("blocks")` groups a whole footer into **one block of
> several short lines**. The PSoC 6 footer is a single 4-line, 41-character block
> (`Datasheet` / `46` / `002-23185 Rev. *S` / `2025-11-06`), so the rule would
> have discarded the footer on 132 of 134 pages -- the majority of the very
> furniture this feature exists to remove. Across seven documents the rule missed
> genuine footers on five of them, and on `current_sensor.pdf` it detected
> nothing at all.
>
> PageIndex's equivalent guard is characters-*per*-line (`info_weight/lines < 15`),
> not a line count, because its blocks are paragraph clusters. A chars-per-line
> variant was also measured and also over-excludes: it drops the Allegro address
> footer, the `current_sensor` running title and the TJA1051 legal footer. The
> 200-character cap alone separates furniture from body prose, and the recurrence
> threshold does the real work.
>
> | eligibility rule | furniture keys found across 7 documents |
> |---|---|
> | with `>= 3 lines` excluded | 9 -- misses real footers on 5/7 documents |
> | **character cap + caption keyword only (adopted)** | **16 -- all inspected, all genuine** |
> | chars-per-line variant | 13 -- still drops real multi-line footers |

**Key.** `normalize_key` collapses whitespace and masks digit runs to `#`, so
`002-23185 Rev. *S | 2025-11-06` becomes `#-# Rev. *S | #-#-#`. Page numbers and
revision dates match across pages while the letters must still agree.

**Threshold.** A key is furniture when it appears on at least
`max(3, ceil(0.5 * total_pages))` distinct pages, counted once per page.

The 0.5 is measured, not chosen by taste: across the seven-document corpus real
furniture landed at 52-100% of pages. The only keys below 92% are tcan1044a-q1's
two -- `www.ti.com` at 22/42 (52%) and `Product Folder Links: TCAN#A-Q#` at 26/42
(62%) -- and nothing non-furniture came within reach of the threshold. The margin
above 0.5 is therefore thin on exactly one document, which is why lowering it was
tested rather than assumed (below). The `max(3, ...)` floor means a 1- or 2-page
document can never have furniture, which is the honest answer when there is no
recurrence evidence to have.

**Lowering it was tested and rejected.** At 0.33 the PSoC gains
`6 Electrical specifications` (47/134) and at 0.25 the barometer gains
`Register description` (12/41) -- both **running section headings**, i.e. real
content repeated at the top of every page of a chapter. A lower threshold buys
one genuine TI header (`TCAN1044A-Q1 SLLSFJ3D ...`, 14/42) at the price of
deleting section titles. 0.5 is the last value at which the corpus is clean.

**Emit.** A block is dropped iff banded AND eligible AND its key is furniture.
Everything else is joined exactly as today, so a document with no running
furniture produces a byte-identical text file.

### Why block granularity, not lines

An earlier line-level scan found `Table #` recurring 89 times on the PSoC -- a
genuine caption that digit-masked matching would have deleted. At block
granularity `Table 43` is a body block and never enters the band, while the
PSoC's four footer lines collapse into one block. Measured with the band and the
adopted eligibility rule, across the seven-document survey corpus:

| document | dropped |
|---|---|
| PSoC 6 (134pp) | `PSOC(tm) # MCU` x133, `Datasheet # #-# Rev. *S #-#-#` x132 |
| motor_driver (20pp) | running title x20, `And Overcurrent Protection A#` x19, Allegro address x19 |
| tcan1044a-q1 (42pp) | `Product Folder Links: TCAN#A-Q#` x26, `www.ti.com` x22 |
| TI PCN (7pp) | `Texas Instruments Incorporated TI Information - Selective Disclosure ...` x7 |
| current_sensor (26pp) | running title x25, Allegro address footer x25 |
| barometer (41pp) | `DPS310 Digital XENSIV(tm) Barometric Pressure Sensor ...` x39, `# V#.# #-#-#` x39 |
| TJA1051 (25pp) | `High-speed CAN transceiver` x25, `NXP Semiconductors TJA#` x24, legal footer x23, `Product data sheet Rev. # -- ...` x23 |

Every key above was inspected and is genuine furniture. The out-of-band sanity
check -- that no *non*-furniture block recurs above threshold outside the band --
was run on the first three documents and found none; it is worth repeating over
the whole corpus during implementation.

### Deliberately excluded

- **Fuzzy / edit-distance matching.** Accepted cost: furniture whose *letters*
  vary per page (a per-chapter running title) is not detected. It fails safe.
- **Font and style analysis.** Not needed for the band + recurrence decision.
- **Cross-page geometric stability.** PageIndex requires a matched block to
  occupy nearly the same rectangle (sum of squared edge deltas < 100). The 20%
  band already constrains position and exact-key matching constrains content, and
  it tested clean across the seven-document corpus. This is the first thing to reach for if
  false positives appear -- it is not built speculatively.

## Integration and data flow

```
scan_pages(doc)
  # pass 1 -- one traversal, buffering
  for each page:
      pages[i] = _extract_page_blocks(page)      # [(text, banded)], column-ordered
      keys[i]  = { normalize_key(t)
                   for (t, banded) in pages[i]
                   if banded and is_candidate(t) }      # a set: dedupes within the page

  furniture = detect_furniture(keys, total_pages=len(doc))

  # pass 2 -- over the buffer, no PDF access
  for each page i:
      kept = [ t for (t, banded) in pages[i]
               if not (banded and is_candidate(t)
                       and normalize_key(t) in furniture) ]
      emit "--- PAGE {i+1} ---"
      emit "\n".join(kept)
```

Page markers are unchanged and emitted for every page, including one left empty
by stripping, so page alignment and `get_section_text`'s page ranges are
unaffected.

**Escape hatch.** `DATASHEETINDEX_FURNITURE` disables detection entirely when set
to any of `0`, `false`, `no`, `off` (case-insensitive, whitespace-stripped);
unset or anything else enables it. This mirrors `_parallel_enabled_by_env`
exactly, including its reason: matching only the literal `"0"` would silently
ignore `=false` and leave the escape hatch looking broken to the person who most
needs it. Cheap insurance for a change that deletes text from a downstream
consumer's input.

**Observability.** One INFO log line per build naming the detected keys and the
number of blocks dropped. No new public API surface: the count is a debugging
aid, not a signal an agent acts on.

**Artifact cache.** `artifact_cache` fingerprints on `datasheetindex_version`, so
the version bump invalidates stale text files automatically. No migration.

## Failure modes

| Failure | Behaviour | Rationale |
|---|---|---|
| Document under 3 pages | No furniture detected | Insufficient recurrence evidence; keeping text is the safe default |
| Page with no text blocks | Unchanged from today (empty string) | Existing behaviour preserved |
| Every block on a page is furniture | Page marker emitted, body empty | Page alignment must not shift |
| Furniture varies in letters per page | Not detected, text kept | Fails safe by construction |
| Detection raises | Must not be possible: `furniture.py` is pure and total over its inputs. No try/except is added, so a genuine bug surfaces rather than silently disabling stripping | Matches the repo's preference for loud failure over silent degradation |

## Testing

**Unit, default lane** (`tests/test_furniture.py`) -- `normalize_key` masking and
whitespace collapse; `is_candidate` rejecting over-long and caption-prefixed
blocks **and accepting a multi-line short block**, which pins the removed
line-count rule so it cannot be reintroduced; `detect_furniture` threshold
arithmetic including the `max(3, ...)` floor and the per-page dedupe.

**Integration, default lane** (`tests/test_textfile.py`):

1. Synthetic multi-page PDF with a running header and footer: both dropped, body
   text and table rows kept.
2. **Caption guard**: a `Table N` caption placed high on the page, recurring on
   every page. Must NOT be dropped. This is the regression test for the failure
   the line-level approach would have caused.
3. **Short-document guard**: a 2-page PDF with an identical header on both pages.
   Nothing dropped.
4. **No-furniture guard**: a document without running furniture produces
   byte-identical text to 0.32.0.
5. Escape hatch: `DATASHEETINDEX_FURNITURE=0` restores 0.32.0 output exactly.

**Real-document, `real_pdf` marker** -- on the bundled PSoC: the two known
furniture shapes are absent; a known body string is still present; the **4-line
footer block** is gone (the specific case the removed line-count rule would have
kept); and search improves to the measured figures -- `Datasheet` 138 -> 6,
`Rev. *S` 133 -> 1, and `PSOC` no longer reaching the default 200 result cap.

Assert those measured numbers, not round targets. An earlier draft of this spec
asserted `search_text("PSOC")` would fall "to under 20"; measured, it falls to
**76**, because `PSOC` legitimately appears throughout the body of a PSoC
datasheet. That assertion would have failed, and the temptation on a red test is
to weaken the threshold rather than ask which number was wrong.

**Oracle validation, `layout` marker** -- compare our decisions against
`pymupdf.layout`'s `page-header` / `page-footer` labels on the bundled real PDFs.

Assert *precision*: of the blocks we drop, the fraction the model also labels
`page-header` / `page-footer`. The procedure is fixed even though the number is
not: measure precision during implementation, and if it is below 0.95, treat that
as a design defect and fix the detector rather than lower the bar. Pin the
observed value in the test with no downward slack, so a later regression fails
here. If a specific disagreement is judged acceptable, it is named in the test
with its reason rather than absorbed into a looser threshold.

Recall is computed and logged but never asserted. The model is a cross-check, not
ground truth, and we knowingly detect less than it does -- we skip fuzzy matching
and it does not.

## Compatibility

- **The page-matched text file changes** for any document with running furniture.
  This is the point. Page markers, page ranges and section boundaries are
  unaffected.
- **`preamble` / `preamble_pages` are unchanged**, byte for byte.
- **`extract_table_markdown` is unchanged** -- it has its own suppression from
  0.32.0 via the layout model, on a different code path.
- **No public signature changes.** `generate_text`, `scan_pages`,
  `get_section_text` and `search_text` keep their contracts.
- Downstream (`datasheet-agent`) sees cleaner text with no API change; the
  escape hatch covers an unforeseen regression.

## Out of scope

- The non-LLM outline detector. This is its prerequisite; it gets its own spec.
- The ToC enumeration gate (a `"Page N"` outline scoring 0.68 against a 0.3
  fallback threshold). Unrelated, still open.
