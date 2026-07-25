# Figure indexing: raster regions and captions

Design, 2026-07-25. Status: approved, not implemented. Section 6 revised the same
day: VLM captioning moved from opt-in to on-by-default with deterministic triage
and a caller-settable cap.

Independent of the preamble spec of the same date. **Depends on artifact reuse**,
shipped in 0.24.0: section 6 uses its `llm_enrichment_incomplete` field and its
`build_options` cache key.

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
otherwise avoids. Section 6's `max_figure_captions` *is* plumbed, and the
difference is not inconsistency -- see "Why this is plumbed when `min_area_pct`
is not" there.

**The threshold is precautionary and uncalibrated.** Measured distribution:

| | placements | <0.5% | 0.5-2% | 2-5% | >5% |
|---|---|---|---|---|---|
| PSoC 6 (134p) | 22 | 0 | 0 | 1 | 21 |
| PCN (7p) | 9 | 0 | 1 | 2 | 6 |

Neither document contains a single image below 0.5%, so this corpus cannot
calibrate the value. It is set low on purpose: excluding real content is the
expensive error, and a few logo entries are the cheap one.

### 6. VLM captions, on by default where the text layer failed

An uncaptioned raster region is located but unnamed, so an agent asking "where
is the die size table" cannot find it without inspecting pages one by one. A pass
renders each such region and asks a VLM for one line, stored as `caption` with
`caption_source: "llm"`.

#### Only where the deterministic signal failed

Captioning every raster region would be waste, and measurably so. On the PSoC all
22 raster placements sit on pages that already carry a `Figure N` caption in the
text layer; on the PCN none of the 8 do.

| | raster placements >=1% | on already-captioned pages | VLM candidates |
|---|---|---|---|
| PSoC 6 (134p) | 22 | 22 | **0** |
| PCN (7p) | 8 | 0 | **8** |

The rule: **a page's raster regions are candidates unless the page carries at
least as many text-layer captions as raster regions.** A well-formed datasheet
therefore costs nothing, and the calls land on exactly the documents where the
index is otherwise blind -- on the PCN they include the page-5 Product Attributes
table that motivates this design.

Both counts come from the `scan_pages` result. Caption entries and raster entries
are already in one page-keyed array, so the comparison needs no new plumbing.

"At least as many" rather than "no caption at all" is what stops a page with one
caption and three raster regions from silently skipping two. **Neither fixture
contains such a page** -- every raster-bearing page in the PSoC has exactly one
caption and exactly one region, and no page in either document has more regions
than captions -- so like the clipping in section 3 this branch is precautionary
and its test is synthetic. When the counts do disagree, *all* of that page's
regions become candidates rather than some arbitrary subset: section 4 already
accepts caption/region redundancy rather than guess an association, and picking
"which two of three are uncaptioned" would be exactly that guess.

#### The gate is a key, not a flag

`caption_figures: bool = True`. Captioning runs when the flag is set **and** a
vision-capable callable is available; either alone yields no captions. A caller
with no credentials configured gets the deterministic index and nothing else,
which is the intended fallback rather than an error.

This follows the ToC fallback, not summaries, and the distinction is repair versus
enhancement. `index.py:578` already creates a client on its own when ToC quality
is below threshold -- cost incurred on key presence, with no explicit flag,
because the deterministic path failed and the artifact is degraded without it.
Summaries sit behind `include_summaries` because they apply to every document and
scale with its size. Triaged captioning has the fallback's shape: it fires only
where the text layer yielded no caption, and on a normal datasheet it does not
fire at all.

`caption_figures` remains a real off switch, for a caller who wants a key for the
ToC fallback but no figure spend.

**One client, not two.** `build()` self-creates a client only for the weak-ToC
branch today. It must now also create one when caption candidates exist, and both
branches must share that single client and the existing `close_llm_client` in the
`finally` -- a second client would double the connection cost and leak on the path
where only captions need it.

**There is no `model=None` guard, deliberately.** `include_summaries` raises when
set without a model (`bound.py:199`) because an explicit opt-in that cannot be
honoured is a caller error. That reasoning inverts here: with the default `True`,
the same guard would make **every keyless build raise**, destroying the fallback
this design is built around. A missing client yields no captions and a recorded
note instead.

#### The cap is a parameter

`max_figure_captions: int = 20`. Candidates are ordered by visible area
descending, so the cap retains the most substantive regions. What it dropped is
never silent:

```json
"figure_captions_excluded": {"above_max": 14, "max_figure_captions": 20}
```

The key is emitted on every build, with `above_max: 0` when nothing was dropped,
matching `figures_excluded` -- a consumer reads the effective cap from the
artifact rather than assuming this version's default.

Neither fixture approaches the cap. It exists for the shape neither fixture has:
a scanned document whose every page is one full-page image over an empty text
layer, where the triage rule authorizes one VLM call per page inside a single
`build_datasheet` the caller is blocked on.

**Hitting the cap leaves the artifact complete and cacheable.** The cap is
deterministic, recorded in the artifact, and reproducible from the same inputs, so
reuse serves exactly what a rebuild would produce. A caption call that *raises* is
the opposite -- transient and not reproducible -- and still sets
`llm_enrichment_incomplete`. Keeping the two apart is the point: a disclosed limit
is a fact about the artifact, a gateway error is not.

**Why this is plumbed when `min_area_pct` is not.** Section 5 keeps `min_area_pct`
off the public surface because no caller has needed to change it. That reasoning
does not transfer. `min_area_pct` changes what the free deterministic index
contains; `max_figure_captions` bounds spend and latency, which a caller can
reasonably want to set per document. The tool description must say so plainly --
each caption is one VLM call, and raising the cap raises cost proportionally.

Both fields are plumbed as `include_summaries` is:

| site | change |
|---|---|
| `index.py` `build()` | `caption_figures: bool = True`, `max_figure_captions: int = 20` |
| `tools/bound.py` `build_datasheet()` | both parameters, no `model=None` guard (above) |
| `tools/bound.py` `_BuildOptions` | both as new fields -- so they key the in-memory cache and the sidecar's `build_options` |
| `tools/defs.py` | `"caption_figures": {"type": "boolean"}` and `"max_figure_captions": {"type": "integer"}` in the input schema, read with the same defaults at the call site, and an `IMPORTANT - figure captioning cost` paragraph stating that captioning runs by default when credentials are configured, that each caption is one VLM call, and that raising `max_figure_captions` raises cost proportionally |
| `batch.py` `build_batch()` | both keyword parameters, passed through |
| `cli.py` | `--no-caption-figures` and `--max-figure-captions`; no "requires `--model`" check, per the guard note above |

#### They belong in the cache key

`2026-07-25-on-disk-artifact-reuse-design.md` records the resolved `build_options`
in the sidecar, so adding both fields to `_BuildOptions` puts them there by
construction. Without that, a captioned artifact would be served to a caller who
turned captioning off and, worse, an uncaptioned one to a caller who did not.

Keying on `max_figure_captions` over-keys slightly: caps of 20 and 50 produce
identical artifacts on a document with 8 candidates, yet miss each other's cache
entries. That is the safe direction -- a spurious rebuild costs one build, while
an under-keyed hit serves an artifact the caller did not ask for.

**`min_area_pct` stays out of the cache key**, consistent with section 5: it is
not plumbed through `build()`, so it cannot vary between two builds of the same
document by the same version. A change to its default ships with a release, and
the sidecar already keys on the exact version.

#### A missing key marks the artifact incomplete

When candidates exist and no vision-capable client is available, the build appends
`figure_captions_no_client` to the enrichment notes, so `llm_enrichment_incomplete`
is set and the artifact is not reused. This mirrors `toc_fallback_no_client`
(`index.py:582-586`) exactly.

The consequence is deliberate and worth stating plainly: **a keyless build of a
document with blind regions loses on-disk reuse.** Adding credentials later then
yields captions without hand-clearing the output directory, which is the behaviour
the reuse design intends. The measured blast radius is small -- the PSoC produces
zero candidates and stays cacheable, and the PCN was already uncacheable for its
ToC -- so this costs reuse only on documents whose artifacts are genuinely
degraded.

A document with **no** candidates appends nothing and stays complete. A missing key
is only a defect where there was something to caption.

**The deterministic index is never gated.** Regions and text-layer captions cost
near zero inside the existing page pass and are always emitted, whatever the flag,
the cap, or the credentials. An agent always learns a figure is *there*, and pays
only for learning what it is.

#### The same model as the ToC fallback and summaries, and no new model knob

`"gpt-4.1"` is already the default in `create_llm_client` (`llm/client.py:250`)
and what `index.py:671` uses for the automatic fallback, and it is
vision-capable. `caption_figures` gates *whether* captioning runs; the model comes
from the callable the caller supplies, or from the one `build()` self-creates.

Three consequences of reusing it:

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

#### Rendering is serial, dispatch is concurrent

PyMuPDF is not thread-safe for concurrent page work -- the parallel table scan
already carries that scar, and measured *wrong counts* rather than merely slower
ones. So every selected region is rendered first, serially, through
`inspect_page`. The resulting base64 PNGs are then dispatched to the VLM through
a small bounded thread pool, which is network I/O and safe to overlap.

Eight candidates on the PCN, so its cost is trivial either way. At the default cap
of 20, serial dispatch at typical multi-second VLM latency would put tens of
seconds inside one `build_datasheet`, which is why the split is worth its
complexity. **Per-call latency is unmeasured** -- neither fixture has been run
against a live gateway for this -- so implementation must measure it and record
the real serial and concurrent numbers here, exactly as section 7 requires for the
scan cost.

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

`figures`, `figures_excluded` and `figure_captions_excluded` are new top-level
keys in the ToC JSON. Purely additive; no existing key changes. A document with no
qualifying raster images and no figure captions emits an empty `figures` array
rather than omitting the key, so a consumer can distinguish "none found" from
"this artifact predates the feature" by the key's presence.

`caption_figures` and `max_figure_captions` are new keyword parameters on
`build()`, `build_datasheet()` and `build_batch()`, and optional properties in the
tool schema, so no existing call site needs to change.

**`caption_figures` defaulting to `True` is a behaviour change, not an addition**,
and it is the one thing here an existing caller can notice. A build that supplies
-- or self-creates -- a vision-capable client will begin issuing VLM calls
wherever the triage finds candidates. Three things bound it: the deterministic
index is identical either way, a well-formed datasheet yields zero candidates
(measured, section 6), and `caption_figures=False` restores today's behaviour
exactly.

Artifacts written by an earlier version are already refused by the sidecar's
version check, which `reuse_blocker` evaluates before `build_options`, so the two
added fields cannot be misread against an older sidecar.

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
- **A page whose caption count matches its raster count is not captioned.** One
  image and one `Figure N` caption, a `describe_image` stub, defaults everywhere:
  assert the stub was never called. This is the PSoC's shape on all 22 placements
  and the test that keeps a normal datasheet free.
- **A page with more raster regions than captions makes them all candidates.**
  Two images and one caption yields two `describe_image` calls, not one. Synthetic
  -- no page in either fixture has this shape.
- **`caption_figures` defaults True**: an uncaptioned region plus a
  `describe_image` stub yields a caption with `caption_source: "llm"` without the
  caller setting anything. `caption_figures=False` with the same inputs yields
  none and never calls the stub.
- **`caption_figures=True` with `model=None` does not raise**, unlike
  `include_summaries` -- it degrades to no captions. Assert the build succeeds;
  this pins the deliberate absence of that guard.
- **The cap retains the largest regions and discloses the rest.** With 25
  candidates of differing area and `max_figure_captions=20`, exactly the 20
  largest carry captions, `figure_captions_excluded` reads
  `{"above_max": 5, "max_figure_captions": 20}`, and the artifact is **not**
  marked incomplete -- a disclosed deterministic limit stays cacheable.
- **Two builds differing only in `caption_figures`, and two differing only in
  `max_figure_captions`, do not share a cache entry** -- on the in-memory cache
  and on the sidecar.
- **A callable without `describe_image` yields no captions and does not error.**
  Pass a plain `(system, user) -> str` stub with candidates present and assert the
  deterministic index is intact with every `caption` null. This is the
  compatibility guarantee for consumers injecting their own callable.
- **No vision client plus candidates marks the artifact incomplete**, with a
  `figure_captions_no_client` note, so it is refused for reuse. **No vision client
  plus no candidates stays complete and cacheable** -- the pair matters more than
  either alone, since the second is what keeps keyless reuse working on a normal
  datasheet.
- **A `describe_image` that raises leaves the build successful**, with `caption`
  null and a warning logged -- the text-only-model case -- **and sets
  `llm_enrichment_incomplete`**, so the caption-less artifact is not cached
  permanently.
- **Every render precedes every dispatch.** With a recording stub on both
  `inspect_page` and `describe_image`, assert no render is interleaved with a
  dispatch, pinning the thread-safety split rather than leaving it to the reader.
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
