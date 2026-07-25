# Figure indexing: raster regions and captions

Design, 2026-07-25. Status: approved, not implemented.

Independent of the artifact-reuse and preamble specs of the same date.

## Problem

Content rendered as a raster image is invisible to the index, and nothing tells
the agent it is missing.

The motivating document is the TI PCN in the repository root. Page 5 yields 294
characters of prose and hides an image covering 25.5% of the page. That image is
a **Product Attributes table**: 8 device columns against roughly 22 rows --
automotive grade level, operating temperature range, wafer fab supplier and
process, die size, assembly site, package group, designator and size, body
thickness, pin count, lead finish and pitch, mount and mold compound suppliers
with their part numbers, bond wire composition and diameter, plus MSL
qualification footnotes. It is the substance of the document, and it is a
multi-device comparison table -- exactly what a comparison workflow needs.

Page 6 is worse: 59% of the page as raster across four images, 754 characters of
text.

**The gap is discovery, not reading.** `inspect_page(page, region)` already
ships, so an agent can read that table today -- it has no way to learn it is
there. Nothing in the ToC, the text file, or the preamble suggests page 5 is
withholding anything. `extract_table_markdown` cannot help either: pymupdf4llm
works on the text layer, so a raster table is invisible to it too.

## Why raster, and why vector detection is excluded

| | raster images | vector drawings | text |
|---|---|---|---|
| PCN p3 | 1 | 162 | 2071 |
| PCN p5 (Product Attributes) | 2 | 15 | 294 |
| PCN p6 | 4 | 0 | 754 |
| PSoC 6 p32 (pinout diagram) | 0 | 232 | 1191 |

Two different mechanisms with very different reliability:

- **Clustering vector drawings to find figures is unreliable**, which is the
  known limitation this design does not try to overcome. 162 drawing operations
  on PCN page 3 are table rules, not 162 figures; 232 on the PSoC pinout page
  could be one diagram or fifty.
- **Enumerating raster images is exact.** `get_image_info` reads the PDF's image
  XObjects and returns real bboxes and pixel dimensions. Nothing is inferred, so
  there is no false-positive rate to calibrate.

And the blind spot is genuinely raster-only: the PSoC pinout page has no raster
image at all, 232 vector drawings, and still extracts 1191 characters, because
**vector figures leak their text** -- note text and pin labels come through. The
layout is lost; the content is not invisible. Restricting scope to raster
regions therefore targets the actual failure rather than settling for part of it.

## Two signals, because neither covers both documents

- **Captions from the text layer.** 30 `Figure N` matches in the PSoC across 134
  pages. Exact, free, and yields a human-meaningful name.
- **Raster regions.** The PCN's table has its title (`Product Attributes`)
  rendered *inside* the image, so no caption exists in the text layer to find.

A datasheet's figures are typically vector with captions; this PCN's are raster
without them. Both signals are needed.

## Design

Report evidence; do not act on it. Same division of labour as the preamble
design: the library says what is there, the agent decides what to do.

### 1. A `figures` array

New top-level key in the ToC JSON. Page-keyed rather than node-attached, because
the geometry is a page property and because the documents that need this most
have the weakest ToCs -- the PCN's is LLM-generated. A node's figures are the
ones whose `page` falls in its `start_page..end_page`, which the agent can
already compute, so there is one source of truth rather than a duplicated list.

```json
"figures": [
  {
    "page": 5,
    "kind": "raster",
    "region": {"left": 0.097, "top": 0.356, "right": 0.904, "bottom": 0.671},
    "bbox": [58, 300, 538, 565],
    "pixels": [1656, 916],
    "page_area_pct": 25.5,
    "page_text_chars": 294,
    "caption": null,
    "caption_source": null
  },
  {
    "page": 8,
    "kind": "caption",
    "region": null,
    "bbox": null,
    "caption": "Figure 1. PSOC 6 MCU block diagram",
    "caption_source": "text"
  }
]
```

**`region` is normalized 0.0-1.0 and is the primary field**, because that is
exactly what `inspect_page`'s `region` parameter consumes (`top`/`bottom`/
`left`/`right`). `get_image_info` returns absolute PDF points, so emitting only
those would force the agent to divide by the page dimensions mid-task, with a
wrong division silently yielding a confident reading of the wrong part of the
page. `bbox` in points is retained alongside it for consumers that need absolute
coordinates, such as highlighting in a viewer.

`page_text_chars` is denormalized onto each entry. It is the "is the agent blind
here" signal -- 294 characters beside a 25.5% image is the whole finding -- and
putting it on the entry saves the consumer a join.

### 2. Captions are reported separately, never merged with regions

A page carrying both a raster image and a `Figure N` caption is *probably* one
figure, and associating them by vertical proximity is a heuristic. It is left
out deliberately: a wrong association actively misleads, telling the agent the
Product Attributes table is "Figure 3. Package outline drawing". Reporting them
as separate entries cannot mislead, and an agent can correlate them from the
page number if it wants to.

The cost is mild redundancy on captioned raster figures. That is the right trade
against a plausible-looking wrong name in a published artifact.

### 3. Threshold, and it is not silent

`min_area_pct`, default 1.0, excludes decorative images -- a vendor logo
repeated on 134 pages would otherwise produce 134 junk entries.

The cap is never silent: a sibling top-level key records what it dropped.

```json
"figures_excluded": {"below_min_area_pct": 3, "min_area_pct": 1.0}
```

Recording the threshold beside the count matters -- a consumer seeing 3 exclusions
cannot judge them without knowing the bar, and the default may change.

`min_area_pct` is a parameter on the enrichment function with a module-level
default. It is deliberately **not** plumbed through `DatasheetIndex.build()` or
the tool schemas until a caller asks: adding a knob to the public surface for a
value nobody has needed to change is the kind of speculative API this design
otherwise avoids.

**The threshold is precautionary and uncalibrated.** Measured distribution:

| | placements | <0.5% | 0.5-2% | 2-5% | >5% |
|---|---|---|---|---|---|
| PSoC 6 (134p) | 22 | 0 | 0 | 1 | 21 |
| PCN (7p) | 9 | 0 | 1 | 2 | 6 |

Neither document contains a single image below 0.5%, so this corpus cannot
calibrate the value. It is set low on purpose: excluding real content is the
expensive error, and a few logo entries are the cheap one.

### 4. Opt-in VLM captions

An uncaptioned raster region is located but unnamed, so an agent asking "where
is the die size table" cannot find it without inspecting pages one by one. An
optional pass renders each uncaptioned region and asks a VLM for one line,
stored as `caption` with `caption_source: "llm"`.

The shape mirrors the existing `include_summaries`: a flag plus a model, off by
default, and the deterministic index is fully functional without it.

**The prompt must ask for a description and forbid transcription.** Something
of the form: name the kind of content (table, schematic, plot, photo) and its
subject in one sentence; do not transcribe values. This is the guard that keeps
the artifact from carrying hallucinated numbers a downstream consumer would
trust. `research.md` is explicit that VLMs hallucinate table rows, and a caption
is a navigation aid, not data.

Six regions on the PCN, so cost is trivial.

### 5. Where it runs

Enumerating every raster placement standalone measures 815 ms for the 134-page
PSoC and 502 ms for the 7-page PCN -- the per-page cost tracks content
complexity, not page count. Against an 8 s build that is a 10% addition, which is
avoidable: `generate_text` already iterates every page with the page object
loaded, so the enumeration belongs in that existing pass at near-zero marginal
cost rather than in a second sweep.

## Rejected

- **Verbalizing raster tables to markdown.** Directly solves the PCN and is
  rejected anyway: `research.md` records that VLMs hallucinate table rows and
  that OCR pipelines beat VLM-only by 7.2% on structured data, and it makes the
  library the extraction engine the architecture rejects. An agent that wants
  the values can call `inspect_page` on the region and read it with its own
  vision, where the result is visibly its own inference rather than a stored
  fact.
- **An OCR text layer.** `page.get_textpage_ocr()` exists and `research.md`
  prefers OCR to VLMs for structured data, but it needs a Tesseract binary,
  which breaks the property that PyMuPDF is the only runtime dependency, and OCR
  still flattens table structure into prose.
- **Vector figure detection.** Unreliable per the numbers above, and largely
  unnecessary since vector figures already leak their text.

## Compatibility

`figures` and `figures_excluded` are new top-level keys in the ToC JSON. Purely
additive; no existing key changes. A document with no qualifying raster images
and no figure captions emits an empty `figures` array rather than omitting the
key, so a consumer can distinguish "none found" from "this artifact predates the
feature" by the key's presence.

## Testing

Synthetic PDFs, so assertions are exact and no fixture is required:

- A page with an embedded raster image yields one entry with the correct `page`,
  `pixels`, and `page_area_pct`.
- `region` round-trips: the normalized values, multiplied back by the page
  dimensions, reproduce `bbox` within rounding. This is the test that protects
  the coordinate contract with `inspect_page`.
- `region` is accepted by `inspect_page` unchanged -- assert against the real
  tool rather than by inspection, so the two cannot drift.
- A `Figure N` caption in the text layer yields a `captioned` entry with
  `caption_source: "text"`.
- A page with both an image and a caption yields **two** entries, pinning the
  no-merge decision so a future change cannot quietly start associating them.
- Images below `min_area_pct` are excluded and counted, and the count is
  non-zero in the emitted JSON.
- A document with neither yields an empty array, not a missing key.
- Everything above runs with no LLM under a plain `uv sync`; the VLM caption
  test skips without credentials, following `tests/test_summarizer.py`.

## Out of scope

- Transcribing or extracting any figure content (above).
- Vector figure detection (above).
- Merging captions with regions (above).
- Any change to `inspect_page`, which already does what is needed.
- Figure indexing for the page-matched text file. The text file mirrors what is
  extractable; a raster region has nothing to contribute to it, and the ToC JSON
  is where structural metadata belongs.
