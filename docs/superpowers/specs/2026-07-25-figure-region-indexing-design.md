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
  there is no false-positive rate to calibrate. The bbox is the *placement*
  rectangle, which can extend past the page; section 3 clips it.

And the blind spot is genuinely raster-only: the PSoC pinout page has no raster
image at all, 232 vector drawings, and still extracts 1191 characters, because
**vector figures leak their text** -- note text and pin labels come through. The
layout is lost; the content is not invisible. Restricting scope to raster
regions therefore targets the actual failure rather than settling for part of it.

## Two signals, because neither covers both documents

- **Captions from the text layer.** 24 caption lines in the PSoC across 134 pages.
  Free, and yields a human-meaningful name -- but *not* exact, and the naive
  pattern is actively wrong; see "Captions are a pattern, not a search" below.
- **Raster regions.** The PCN's table has its title (`Product Attributes`)
  rendered *inside* the image, so no caption exists in the text layer to find.
  The PCN has **zero** `Figure` lines of any form, so on that document captions
  contribute nothing at all.

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
    "page": 10,
    "kind": "caption",
    "region": null,
    "bbox": null,
    "figure_number": 2,
    "caption": "Figure 2 Block diagram",
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

### 2. Captions are a pattern, not a search

Matching `Figure N` anywhere in the text layer publishes prose as captions.
Measured on the PSoC, and the numbers are decisive:

| pattern | matches | what they are |
|---|---|---|
| `Figure N` anywhere | 36 | mostly prose |
| line-anchored `^Figure N <text>` | 6 | **all six are prose**: "Figure 2 shows the major subsystems...", "Figure 3 shows that the clock system..." |
| same-line `^Figure N[.:] <title>` | **0** | this form does not occur in either fixture |
| bare `^Figure N$` | **24** | the actual captions |

So a naive pattern is not merely imprecise, it is inverted: line-anchoring alone
yields six matches of which zero are captions, while the 24 real ones are missed
because this vendor puts the number and the title on **separate lines**:

```
Figure 2
Block diagram
```

Two accepted forms, and prose is excluded structurally rather than by scoring:

- **Split form.** A line that is exactly `Figure N` or `Fig. N`, optional trailing
  `.`/`:`, nothing else. The title is the next non-empty line. `caption` is the
  two joined; `figure_number` carries `N`.
- **Same-line form.** `Figure N` followed by a **mandatory** `.` or `:` separator,
  then the title. The mandatory punctuation is exactly what excludes
  "Figure 2 shows the major subsystems" -- that separator is a space. This form
  occurs zero times in both fixtures and is included because it is common in other
  vendors' documents; it is therefore **unverified here** and its test is
  synthetic.

A `Figure N` mention that matches neither form emits **nothing**. Rejected the
alternative of emitting it as a "text mention": an entry saying page 12 mentions
Figure 3 is noise the agent must filter, and the whole point of the array is that
its entries are worth acting on.

**Two known limits, stated rather than hidden.** The split form's title comes from
line adjacency, so a bare `Figure N` whose next line is body text yields a wrong
title -- textual adjacency is far stronger evidence than the geometric proximity
section 4 rejects, but it is still an inference. And adjacency must be evaluated on
the **same column-aware extraction the text file carries** (`_extract_page_text`),
not `page.get_text()`, or a two-column page can interleave the number and title
with unrelated lines. Both fixtures were measured with the raw extractor, so the
24 must be re-confirmed against the column-aware one during implementation.

### 3. Regions are clipped to the page

`get_image_info` returns the placement bbox, which can extend past the page --
a bleed image, or a placement the producer never intended to be fully visible.
Normalizing an unclipped bbox yields a coordinate outside `0..1`, and
`inspect_page` **raises** on that (`vision.py:96-100`: "Region 'top' and 'bottom'
must be between 0.0 and 1.0"). The spec would then be publishing a `region` that
its own documented consumer rejects.

So: intersect the bbox with `page.rect` first, and derive `region`,
`page_area_pct`, and the emitted `bbox` from the **visible** rectangle. An image
whose intersection is empty is dropped, not emitted with a degenerate region --
`inspect_page` also requires `top < bottom` and `left < right`.

Normalize **relative to the page rect's origin**, not to zero:
`left = (x0 - rect.x0) / rect.width`. `inspect_page` builds its clip as
`rect.x0 + left * rect.width` (`vision.py:105-110`), so origin-relative is the
contract; a page whose CropBox does not start at `(0, 0)` would otherwise be
offset by exactly that origin.

Neither fixture exercises either case -- 0 of 22 images off-page on the PSoC, 0 of
9 on the PCN, and every page rect origin is `(0, 0)` in both. Like `min_area_pct`,
this is precautionary and the tests for it are synthetic.

### 4. Captions are reported separately, never merged with regions

A page carrying both a raster image and a `Figure N` caption is *probably* one
figure, and associating them by vertical proximity is a heuristic. It is left
out deliberately: a wrong association actively misleads, telling the agent the
Product Attributes table is "Figure 3. Package outline drawing". Reporting them
as separate entries cannot mislead, and an agent can correlate them from the
page number if it wants to.

The cost is mild redundancy on captioned raster figures. That is the right trade
against a plausible-looking wrong name in a published artifact.

### 5. Threshold, and it is not silent

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

### 6. Opt-in VLM captions

An uncaptioned raster region is located but unnamed, so an agent asking "where
is the die size table" cannot find it without inspecting pages one by one. An
optional pass renders each uncaptioned region and asks a VLM for one line,
stored as `caption` with `caption_source: "llm"`.

#### The flag is `caption_figures`, and captioning requires it

Captioning must **not** be triggered by a callable merely happening to support
vision. `DatasheetIndex.build()` creates a default client on its own whenever ToC
quality is below threshold (`index.py:565-567`), and that client is
vision-capable, so vision-capability alone as the gate would make every weak-ToC
build silently start issuing VLM calls. The gate is therefore
`caption_figures=True` **and** a vision-capable callable; either alone yields no
captions.

`caption_figures: bool = False` is plumbed exactly as `include_summaries` is,
which is the existing precedent for an opt-in LLM cost:

| site | change |
|---|---|
| `index.py` `build()` | new keyword parameter, defaults `False` |
| `tools/bound.py` `build_datasheet()` | new parameter; raises `ValueError` when set with `model=None`, mirroring the `include_summaries` guard at `bound.py:166` |
| `tools/bound.py` `_BuildOptions` | new field -- so it keys the in-memory cache and the sidecar's `build_options` |
| `tools/defs.py` | `"caption_figures": {"type": "boolean"}` in the input schema, `args.get("caption_figures", False)` at the call site, and an `IMPORTANT - caption_figures` paragraph in the tool description warning that it costs one VLM call per uncaptioned region |
| `batch.py` `build_batch()` | new keyword parameter, passed through |
| `cli.py` | `--caption-figures`, with the same "requires `--model`" check as `--include-summaries` |

**It belongs in the disk-cache key.** `2026-07-25-on-disk-artifact-reuse-design.md`
records the resolved `build_options` in the sidecar, and adding the field to
`_BuildOptions` puts it there by construction. Without that, a captioned artifact
would be served to a caller who did not ask for captions and, worse, an
uncaptioned one to a caller who did.

**The deterministic index is not flag-gated.** Regions and text-layer captions
cost near zero inside the existing page pass and are always emitted; only the VLM
pass is opt-in. So an agent always learns a figure is *there*, and pays only for
learning what it is.

**`min_area_pct` stays out of the cache key**, consistent with section 5: it is
not plumbed through `build()`, so it cannot vary between two builds of the same
document by the same version. A change to its default ships with a release, and
the sidecar already keys on the exact version.

#### The same model as the ToC fallback and summaries, and no new model knob

`"gpt-4.1"` is already the default in `create_llm_client` (`llm/client.py:250`)
and what `index.py:671` uses for the automatic fallback, and it is
vision-capable. `caption_figures` gates *whether* captioning runs; the model
comes from the callable the caller already supplies.

Two consequences of reusing it:

- **`"gpt-4.1"` is currently a duplicated literal**, in `client.py:250` and
  `index.py:671`. A third copy for figure captions would make drift a matter of
  time, so this work should collapse the default to one shared constant rather
  than add to it.
- **Vision capability is not guaranteed.** A caller may configure any model
  name, and a text-only one will fail at the gateway. A caption failure must
  therefore leave `caption` null, log at warning, and leave the deterministic
  index untouched -- the same posture as a failed summary or a failed sidecar
  write, never a failed build.
- **A swallowed caption failure must also mark the artifact incomplete.** Logging
  a warning and returning is exactly how a degraded artifact gets written, and
  `2026-07-25-on-disk-artifact-reuse-design.md` would then cache it permanently:
  every fingerprint field matches on the next request, so the transient gateway
  error becomes a document with no captions, forever. So the build sets
  `llm_enrichment_incomplete` (that spec's field) when any caption call raises,
  and reuse is refused. A *successful* caption pass that legitimately produces no
  captions -- no uncaptioned regions to name -- is complete, not incomplete.

**`LlmCallable` cannot carry an image, so the protocol needs extending.** It is
`__call__(system, user) -> str` (`client.py:19`) with no image parameter. Follow
the precedent set for structured output in 0.19.0: an *optional* protocol
extension detected by duck-typing, not a change to the base protocol.

```python
class VisionLlmCallable(Protocol):
    """Optional image-input interface for figure captioning."""

    def describe_image(
        self, system: str, image_base64: str, *, media_type: str = "image/png"
    ) -> str: ...

def get_vision_client(llm_callable: object | None) -> VisionLlmCallable | None:
    """Return the vision interface when the callable exposes one."""
    describe_image = getattr(llm_callable, "describe_image", None)
    if callable(describe_image):
        return cast(VisionLlmCallable, llm_callable)
    return None
```

A callable without `describe_image` -- including any third-party
`(system, user) -> str` a consumer injects today -- simply yields no captions,
exactly as a callable without `structured_json` falls back to the free-text ToC
prompt. This is additive and breaks no existing consumer.

**`inspect_page` is the renderer; do not write a second one.**
`tools/vision.py:35` already takes `region` as the normalized 0.0-1.0 dict this
spec emits and returns `[{"type": "image", "data": <base64>, "mime_type":
"image/png"}]`. So the `region` field serves two consumers through one
coordinate contract: the agent inspecting a figure, and this captioning pass
rendering it. That is also why the round-trip test against the real
`inspect_page` matters -- it now guards an internal caller as well as an
external one.

**The prompt must ask for a description and forbid transcription.** Something
of the form: name the kind of content (table, schematic, plot, photo) and its
subject in one sentence; do not transcribe values. This is the guard that keeps
the artifact from carrying hallucinated numbers a downstream consumer would
trust. `research.md` is explicit that VLMs hallucinate table rows, and a caption
is a navigation aid, not data.

Six regions on the PCN, so cost is trivial.

### 7. Where it runs

Enumerating every raster placement standalone measures 815 ms for the 134-page
PSoC and 502 ms for the 7-page PCN -- the per-page cost tracks content
complexity, not page count. Against an 8 s build that is a 10% addition.

**Folding it into the existing pass does not make it free, and the earlier draft
overclaimed.** `get_image_info()` still runs once per page and still costs whatever
it costs; what the fold avoids is a *second* traversal -- reopening and re-loading
every page object, which `generate_text` has already paid for. So the 502-815 ms
standalone figure is an **upper bound** on the addition, and the fold removes only
the page-loading share of it, which was not measured separately.

It is still the right placement: strictly cheaper than a second sweep, with no
offsetting cost, and the pass already holds everything the entries need. But the
build gets measurably slower, and the honest number to plan against is up to
~800 ms on a 134-page document until the split is measured. Measure it during
implementation and record the real marginal cost here.

That pass also already holds everything the entries need: the page rect for
normalizing `region`, the extracted text for `page_text_chars` and for
`Figure N` captions.

**`generate_text` keeps its signature; a new function returns both.** It is
`generate_text(doc) -> str` (`core/textfile.py:213`) with one production caller
but fourteen call sites in `tests/`, so widening its return type would be a
gratuitous break. Same shape as the preamble spec's `build_front_matter` /
`generate_preamble` pair:

```python
@dataclass(frozen=True)
class PageScan:
    text: str                              # byte-identical to generate_text today
    figures: list[dict[str, object]]
    excluded_below_min_area: int

def scan_pages(doc, *, min_area_pct: float = 1.0) -> PageScan: ...

def generate_text(doc) -> str:
    """Retained wrapper; the page-matched text file alone."""
    return scan_pages(doc).text
```

`index.py` switches to `scan_pages` and emits `figures` and `figures_excluded`
from the result. `min_area_pct` is keyword-only and reaches its module default
here, per section 5.

**Implementation note, the same hazard the preamble spec carries.**
`tests/test_index.py` monkeypatches `datasheetindex.index.generate_text` with
`lambda _doc: "--- PAGE 1 ---\n..."` in seven places (lines 269, 365, 478, 552,
602, 687, 733). Once `index.py` calls `scan_pages`, those stubs patch a function
that is no longer called. This one fails *loudly* rather than silently -- the real
`scan_pages` runs against the synthetic document and the assertions see unstubbed
text -- but they must still be repointed to return a `PageScan`, and the assertion
that the stub value reaches the emitted JSON keeps them honest afterwards.

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
additive; no existing key changes. `caption_figures` is a new keyword parameter
defaulting to `False` on `build()`, `build_datasheet()`, and `build_batch()`, and
an optional property in the tool schema, so every existing caller and every
existing tool invocation behaves exactly as today. A document with no qualifying raster images
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
- **An image extending past the page is clipped**, and the resulting `region` is
  within `0..1`. Assert by passing the emitted region to the real `inspect_page`:
  an unclipped one raises `ValueError`, so this test fails loudly on a regression
  rather than merely reporting an odd number. Also assert `page_area_pct` and the
  emitted `bbox` describe the visible rectangle, not the placement.
- **An image entirely off-page is dropped**, not emitted with a degenerate region
  that `inspect_page`'s `top < bottom` check would reject.
- **A page whose rect origin is not `(0, 0)`** (a CropBox offset) normalizes
  relative to that origin: the emitted region, fed back through `inspect_page`,
  crops the same rectangle. Synthetic -- every page in both fixtures starts at
  `(0, 0)`, so nothing else would catch an absolute-coordinate mistake.
- **The split caption form**: a page whose text has `Figure 2` alone on a line and
  `Block diagram` on the next yields one entry with `figure_number: 2` and
  `caption: "Figure 2 Block diagram"`, `caption_source: "text"`. This is the form
  24 of the PSoC's captions actually take.
- **The same-line form**, with mandatory punctuation: `Figure 3. Package outline`
  yields that caption. Synthetic, since the form occurs in neither fixture.
- **Prose is not a caption, three ways.** `as Figure 5 shows` mid-line, a line
  *opening* `Figure 2 shows the major subsystems` (which the mandatory separator
  excludes), and `See Figure 3` each yield **zero** entries. These are the 30
  false positives a naive pattern produced, so they are the tests that matter most
  in this section -- publishing prose as a caption puts a wrong figure name in the
  artifact.
- **Caption detection runs on the column-aware extraction**, not `page.get_text()`:
  a two-column page whose raw text order would interleave the number and title
  still yields the correct pairing. Assert against `_extract_page_text` output.
- A bare `Figure N` as the last line of a page yields no title rather than
  reaching into the next page or raising.
- A page with both an image and a caption yields **two** entries, pinning the
  no-merge decision so a future change cannot quietly start associating them.
- Images below `min_area_pct` are excluded and counted, and the count is
  non-zero in the emitted JSON.
- A document with neither yields an empty array, not a missing key.
- `generate_text(doc)` returns exactly `scan_pages(doc).text`, so the retained
  wrapper cannot drift from the function it delegates to.
- The seven repointed `generate_text` stubs in `tests/test_index.py` return a
  `PageScan` and their text still reaches the emitted artifacts.
- **`caption_figures` defaults False, and a vision-capable callable alone does
  not caption.** Pass a stub exposing `describe_image` without setting the flag
  and assert every `caption` is null and `describe_image` was never called. This
  is the test for the weak-ToC build that creates its own vision-capable client.
- `caption_figures=True` with `model=None` raises on `DatasheetTools`, mirroring
  the `include_summaries` guard.
- **Two builds differing only in `caption_figures` do not share a cache entry**,
  on the in-memory cache and on the sidecar once artifact reuse lands.
- **A callable without `describe_image` yields no captions and does not error**
  with `caption_figures=True`. Pass a plain `(system, user) -> str` stub and
  assert the deterministic index is complete with every `caption` null. This is
  the compatibility guarantee for consumers injecting their own callable.
- **A `describe_image` that raises leaves the build successful**, with `caption`
  null and a warning logged -- the text-only-model case -- **and sets
  `llm_enrichment_incomplete`**, so the caption-less artifact is not cached
  permanently.
- The captioning pass calls `inspect_page` with the emitted `region` unmodified,
  asserted with a recording stub, so an internal caller cannot start
  transforming coordinates the agent is told to use verbatim.
- Everything above runs with no LLM under a plain `uv sync`; the live VLM caption
  test skips without credentials, following `tests/test_summarizer.py`.

## Out of scope

- Transcribing or extracting any figure content (above).
- Vector figure detection (above).
- Merging captions with regions (above).
- Any change to `inspect_page`, which already does what is needed and is reused
  as-is by the captioning pass. Its `0..1` validation is treated as the contract to
  satisfy, not a restriction to relax -- section 3 clips to fit it.
- Caption forms beyond the two in section 2. `Table N`, `Diagram N`, non-English
  labels, and vendor-specific numbering (`Figure 3-2`) are all real and all
  uncalibrated on a two-document corpus. Emitting nothing is the safe answer;
  widening the pattern needs a wider corpus first.
- Correlating a caption entry with a raster entry on the same page (section 4).
- Any new model configuration. Figure captions use the callable the caller
  already provides, on the same model as the ToC fallback and summaries.
- Figure indexing for the page-matched text file. The text file mirrors what is
  extractable; a raster region has nothing to contribute to it, and the ToC JSON
  is where structural metadata belongs.
