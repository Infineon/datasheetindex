# Running header/footer detection in the page-matched text file

Status: design approved, not yet implemented
Date: 2026-08-11
Target version: 0.33.0

## Problem

Every page of a datasheet carries running furniture -- a header naming the part,
a footer with the document title, a revision string and a page number. The
page-matched text file reproduces all of it, once per page, and every consumer
reads it.

The cost is not primarily tokens. It is search precision. Measured on the
bundled PSoC 6 datasheet (134 pages):

| query | matches | of which running header |
|---|---|---|
| `PSOC` | 200 (hits the result cap) | 133 |
| `Datasheet` | 138 | 133 |
| `Rev. *S` | 133 | 133 |

An agent searching for the part family gets a wall of identical hits and the cap
evicts the real ones. Total furniture is 5.6% of the text file on the PSoC
(11,044 characters, ~2,700 tokens) and 5.9% on the TI PCN.

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

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Disposition | Drop furniture from the text file | Fixes `search_text`, `get_section_text` and the LLM ToC fallback at once, with no new format for consumers to learn |
| Preamble | Keeps raw, unstripped text | The architecture doc advertises "zero heuristics"; page 1 is also where recurrence has least evidence and where real prose sits at the page foot |
| Matching | Exact text plus digit masking | Catches all six shapes measured on the PSoC; deterministic and easy to reason about |
| Fuzzy matching | Excluded | A similarity threshold can delete a genuine one-off line resembling its neighbours. Fails safe by keeping text |
| Traversal | Buffer blocks in the existing single pass | A second traversal costs +22% of the scan (0.33s of 1.50s) to save ~200KB we are not short of |

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
- *Eligible* (`is_candidate`): fewer than 3 lines, at most 200 characters of raw
  block text (measured before normalization, so masking cannot change
  eligibility), and does not begin with a caption keyword (`figure`, `fig.`,
  `table`, `chart`, case-insensitive, after leading whitespace is stripped).

**Key.** `normalize_key` collapses whitespace and masks digit runs to `#`, so
`002-23185 Rev. *S | 2025-11-06` becomes `#-# Rev. *S | #-#-#`. Page numbers and
revision dates match across pages while the letters must still agree.

**Threshold.** A key is furniture when it appears on at least
`max(3, ceil(0.5 * total_pages))` distinct pages, counted once per page.

The 0.5 is measured, not chosen by taste: real furniture landed at 52-100% of
pages across three documents, with `www.ti.com` on tcan1044a-q1 the floor at
22/42, and nothing non-furniture came within reach. The `max(3, ...)` floor means
a 1- or 2-page document can never have furniture, which is the honest answer when
there is no recurrence evidence to have.

**Emit.** A block is dropped iff banded AND eligible AND its key is furniture.
Everything else is joined exactly as today, so a document with no running
furniture produces a byte-identical text file.

### Why block granularity, not lines

An earlier line-level scan found `Table #` recurring 89 times on the PSoC -- a
genuine caption that digit-masked matching would have deleted. At block
granularity `Table 43` is a body block and never enters the band, while the
PSoC's three footer lines collapse into one block. Measured, with the band
applied, across three documents:

| document | dropped | out-of-band recurrences above threshold |
|---|---|---|
| PSoC 6 (134pp) | `PSOC(tm) # MCU` x133, `Datasheet # #-# Rev. *S #-#-#` x132 | none |
| motor_driver (20pp) | running title x20, `And Overcurrent Protection A#` x19, Allegro address x19 | none |
| tcan1044a-q1 (42pp) | `Product Folder Links: TCAN#A-Q#` x26, `www.ti.com` x22 | none |

### Deliberately excluded

- **Fuzzy / edit-distance matching.** Accepted cost: furniture whose *letters*
  vary per page (a per-chapter running title) is not detected. It fails safe.
- **Font and style analysis.** Not needed for the band + recurrence decision.
- **Cross-page geometric stability.** PageIndex requires a matched block to
  occupy nearly the same rectangle (sum of squared edge deltas < 100). The 20%
  band already constrains position and exact-key matching constrains content, and
  it tested clean on three documents. This is the first thing to reach for if
  false positives appear -- it is not built speculatively.

## Integration and data flow

```
scan_pages(doc)
  for each page:
      blocks = _extract_page_blocks(page)        # column-ordered, banded flags
      buffer blocks; keys[page] = [normalize_key(t) for banded+eligible blocks]
  furniture = detect_furniture(keys, len(doc))
  for each page:
      text = "\n".join(text for (text, banded) in blocks if not dropped)
      emit "--- PAGE N ---" then text
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
whitespace collapse; `is_candidate` rejecting 3-line, over-long and
caption-prefixed blocks; `detect_furniture` threshold arithmetic including the
`max(3, ...)` floor and the per-page dedupe.

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
furniture shapes are absent; a known body string is still present; and
`search_text("PSOC")` drops from the 200-result cap to under 20. That last
assertion is the user-facing goal stated as a test.

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
