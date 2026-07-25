# Figure indexing: raster regions and captions

Design, 2026-07-25. Status: approved, not implemented. Section 6 revised the same
day: VLM captioning moved from opt-in to on-by-default, bounded by a
caller-settable cap rather than by triage -- two triage rules were designed,
measured, and rejected as unsound in the process. Sections 2, 3 and 5 were then
recalibrated against a **14-document, 998-page, 5-vendor corpus** (TI, Infineon,
Microchip, Nexperia, Diodes) rather than the two documents they were designed on;
that measurement corrected the caption number pattern, promoted page clipping from
precautionary to necessary, and confirmed both thresholds. The corpus is not in the
repository -- it produced numbers, not fixtures, and the suite stays synthetic.

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

- **Captions from the text layer.** 492 caption lines across a 14-document,
  998-page corpus; 24 of them in the PSoC. Free, and yields a human-meaningful
  name -- but *not* exact, and the naive pattern is actively wrong; see "Captions
  are a pattern, not a search" below.
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
    "figure_number": "2",
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

Matching `Figure N` anywhere in the text layer publishes prose as captions. The
pattern below was designed on the PSoC alone and then measured against a
**14-document, 998-page, 5-vendor corpus** (TI, Infineon, Microchip, Nexperia,
Diodes), read through `_extract_page_text` -- the column-aware extractor the text
file uses, not `page.get_text()`. The corpus both vindicated the design and found
one real defect.

| form | example | occurrences | documents |
|---|---|---|---|
| same-line, section-relative | `Figure 10-1. Reset Logic` | **404** | 9 / 14 |
| bare, section-relative | `Figure 3-2` + next line | 49 | 1 / 14 |
| bare, plain | `Figure 12` + next line | 33 | 2 / 14 |
| same-line, plain | `Fig. 10. Enable and disable times` | 6 | 1 / 14 |
| no separator (**rejected**) | `Figure 6-2 shows the structure of...` | 70 | prose, correctly excluded |

Two findings, in order of importance.

**The mandatory-separator rule is correct, and it is the load-bearing part.** It
divides 404 real captions from 70 prose lines across 998 pages with no scoring and
no tuning: every rejected line reads "Figure X shows/presents/illustrates...".
Designed against a document where the same-line form occurred *zero* times, it
turned out to describe the most common caption form in the corpus.

**The number pattern was the defect.** `(\d+)` does not match `10-1`, so the
original pattern found captions in **2 of 14** documents. The number is therefore
`(\d+(?:[-–]\d+)?)` -- section-relative numbering, en-dash included, since
publishers use both. That single change takes coverage to **11 of 14** documents
and 492 captions. The three documents still yielding none genuinely contain no
figure captions.

Two accepted forms, then, with prose excluded structurally rather than by scoring:

- **Split form.** A line that is exactly `Figure N` or `Fig. N`, optional trailing
  `.`/`:`, nothing else. The title is the next non-empty line. `caption` is the two
  joined.
- **Same-line form.** `Figure N` followed by a **mandatory** `.` or `:` separator,
  then the title. That separator is what excludes "Figure 2 shows the major
  subsystems", where it is a space.

**`figure_number` is a string, always** -- `"10-1"` as readily as `"12"`. Not an
integer that becomes a string when hyphenated: a union type costs every consumer a
branch, and the field is an identifier to display and match on, never an arithmetic
value.

A `Figure N` mention matching neither form emits **nothing**. Rejected the
alternative of emitting it as a "text mention": an entry saying page 12 mentions
Figure 3 is noise the agent must filter, and the whole point of the array is that
its entries are worth acting on.

**Two known limits, stated rather than hidden.** The split form's title comes from
line adjacency, so a bare `Figure N` whose next line is body text yields a wrong
title -- textual adjacency is far stronger evidence than the geometric proximity
section 4 rejects, but it is still an inference. And adjacency must be evaluated on
the column-aware extraction, or a two-column page can interleave the number and
title with unrelated lines; the corpus numbers above were measured that way, so
they are the numbers the implementation should reproduce.

**A missed caption is cheap, and that bounds this whole section's risk.** Captions
gate nothing: since section 6 abandoned triage, a raster figure whose caption the
pattern misses is still enumerated and still VLM-captioned. The pattern is an
optimization that supplies a free name, not the mechanism by which figures are
discovered. The one case where a miss is unrecoverable is a **vector** figure,
which has no raster region and so no fallback -- see the layout engine under
Rejected.

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

**Clipping is not precautionary; a real vendor document needs it.** The original
two fixtures showed 0 off-page placements of 22 and of 9, which made this look
defensive. Across the 14-document corpus there are **9 placements extending past
the page**, all in `ti-tlv9061` -- a 99-page TI datasheet. Without clipping those
nine normalize outside `0..1` and `inspect_page` rejects every one of them, so the
feature would ship publishing regions its own documented consumer refuses.

A non-zero page rect origin remains unobserved -- 0 pages of 998 -- so *that* half
stays precautionary with a synthetic test. The two cases are no longer in the same
category and the spec should not keep implying they are.

### 4. Captions are reported separately, never merged with regions

A page carrying both a raster image and a `Figure N` caption is *probably* one
figure, and associating them by vertical proximity is a heuristic. It is left
out deliberately: a wrong association actively misleads, telling the agent the
Product Attributes table is "Figure 3. Package outline drawing". Reporting them
as separate entries cannot mislead, and an agent can correlate them from the
page number if it wants to.

The cost is mild redundancy on captioned raster figures. That is the right trade
against a plausible-looking wrong name in a published artifact.

This ruling does more work than it appears to. Section 6 reaches the same answer
for cost -- every raster region is a VLM candidate, including ones a caption may
already name -- because any rule for skipping "already captioned" regions is this
same association wearing a different hat.

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

**The threshold is load-bearing, and the original two documents said the
opposite.** On the PSoC and PCN alone, not one image fell below 0.5%, which made
`min_area_pct` look like dead code kept for safety. Across the 14-document corpus:

| | placements | below 1.0% (excluded) | below 0.5% | documents with a repeated xref |
|---|---|---|---|---|
| 14 documents, 998 pages | 168 | **73 (43%)** | 61 (36%) | 4 / 14 |

So the threshold discards nearly half of all raster placements, and the repeated
image XObject -- the vendor logo stamped on every page, which the two-document
corpus disproved -- occurs in 4 of 14 documents. Both the threshold and its default
are retained on this evidence rather than on caution.

It stays set low on purpose: excluding real content is the expensive error, and a
few logo entries are the cheap one. Since section 6 dropped triage, the cheap error
now also costs a VLM call, but area-descending ordering sorts logos last, so the
cap absorbs them before they displace anything substantive.

### 6. VLM captions, on by default and bounded by a cap

An uncaptioned raster region is located but unnamed, so an agent asking "where
is the die size table" cannot find it without inspecting pages one by one. A pass
renders each such region and asks a VLM for one line, stored as `caption` with
`caption_source: "llm"`.

#### Every raster region is a candidate, and the triage that looked obvious is unsound

The tempting economy is to skip regions the text layer already names. It cannot
be made sound on any signal available here, and the measurements say why.

| | raster placements >=1% | on pages carrying a caption | on pages under 300 chars |
|---|---|---|---|
| PSoC 6 (134p) | 22 | 22 | 10 |
| PCN (7p) | 8 | 0 | 2 |

Two rules were considered and both fail:

- **Comparing per-page caption and raster counts is association by arithmetic.**
  "One caption and one raster region, therefore that caption names that region"
  is the claim section 4 refuses to make geometrically, reached by counting
  instead. It fails the same way: a page carrying a captioned *vector* figure
  beside an uncaptioned raster table has equal counts, so the blind table is
  skipped. The counts do match on all 22 of the PSoC's raster pages -- but
  nothing verifies those captions refer to those images, and a matching count is
  correlation, not association.
- **Thresholding page text avoids the association and does not discriminate.**
  The PSoC's raster pages are *more* text-starved than the PCN's: 10 of 22
  placements sit on pages under 300 characters, one of them at 114 characters on
  a page 59.3% covered by an image, while the PCN's motivating page 5 has 289. No
  threshold separates the two corpora, because on this evidence they are not
  actually different.

So **every raster region above `min_area_pct` is a candidate**, ordered by visible
area, bounded by `max_figure_captions`. This is section 4's ruling applied to cost
rather than to naming: mild redundancy on already-captioned figures is the right
trade against skipping a genuinely blind one. A redundant caption costs one VLM
call; a skipped Product Attributes table is the failure this design exists to
prevent.

The cost is bounded twice. `max_figure_captions` caps it per document, and
`2026-07-25-on-disk-artifact-reuse-design.md` makes it a **once-per-document**
cost rather than once-per-build -- a second build of the same datasheet reuses the
captioned artifact and issues no calls at all.

#### The gate is a key, not a flag

`caption_figures: bool = True`. Captioning runs when the flag is set **and** a
vision-capable callable is available; either alone yields no captions. A caller
with no credentials configured gets the deterministic index and nothing else,
which is the intended fallback rather than an error.

**The precedent this follows is weaker than an earlier draft claimed, and the
honest case for default-on is different.** `index.py:578` self-creates a client
when ToC quality is below threshold -- cost on key presence, no flag -- because
the deterministic path *failed*. Summaries sit behind `include_summaries` because
they apply to every document and scale with its size. Without triage, captioning
has the second shape, not the first: it fires on any document with raster
regions, whether or not anything failed.

Default-on rests on three things instead:

- **The gap is discovery.** An agent that must know to ask for captions has the
  problem this design opened by describing -- nothing tells it the figure is
  there to ask about. An opt-in flag closes the gap only for callers who already
  suspect it.
- **The cost is bounded and amortized.** `max_figure_captions` is a hard per-
  document ceiling the caller sets, and disk reuse charges it once per document
  rather than once per build.
- **The off switch is real.** `caption_figures=False`, or `max_figure_captions=0`,
  restores today's behaviour exactly -- for a caller who wants a key for the ToC
  fallback but no figure spend.

That is a genuine trade rather than a free win, and it is recorded as one.

**One client, not two.** `build()` self-creates a client only for the weak-ToC
branch today. It must now also create one when caption candidates exist, and both
branches must share that single client and the existing `close_llm_client` in the
`finally` -- a second client would double the connection cost and leak on the path
where only captions need it. When `bound.py` hands in a probe client, `build()`
constructs none at all; see "The capability probe owns a client" below for who
closes what.

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

**It is validated as an integer `>= 0`, at every entry point.** A negative value
reaching `candidates[:max_figure_captions]` does not cap anything -- it slices
from the *end*, so `-1` captions all but the last candidate and any value more
negative than the candidate count captions none, both of them silently and
neither of them what the caller asked for. A cost ceiling that can be inverted by
a sign is not a ceiling. So `build()`, `build_datasheet()` and `build_batch()`
raise `ValueError` on a negative value, `tools/defs.py` carries
`"minimum": 0` beside the `integer` type, and the CLI rejects it at parse time
rather than passing it through. `0` is valid and means the deterministic index
with no captioning -- the same result as `caption_figures=False`, reached by the
cost knob instead of the switch.

**The default is calibrated, not guessed.** Across the 14-document corpus the
candidate count per document has a median of **4** and a maximum of **29**, and
only **2 of 14** documents exceed 20. So the cap is generous for the typical
document and binds only on the outliers, which is what a cost ceiling should do.

**The PSoC is one of those two**, at 22 candidates against a default of 20, so the
two largest-area survivors displace two smaller regions and
`figure_captions_excluded` reads `{"above_max": 2, "max_figure_captions": 20}`.
That is the intended behaviour rather than a mis-set default: the ordering keeps
the substantive figures, the exclusion is disclosed, and a caller who wants all 22
raises the cap. It also means the default is exercised by a real fixture rather
than only by synthetic tests.

The cap matters more for the shape neither fixture has:
a scanned document whose every page is one full-page image over an empty text
layer, where every page is a candidate and one VLM call per page lands inside a single
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

**`figure_captions_pending` is a build outcome, not a build option**, and the
distinction is load-bearing. `build_options` records what the caller *asked for*
and is compared for equality; `figure_captions_pending` records what the build
*achieved* and is compared against the current environment by the rule below. Put
in `_BuildOptions` it would key both caches on an outcome, which is neither
meaningful nor stable.

It therefore lands in **two** places, and both are required: on `ArtifactRecord`
beside the existing fingerprint fields, for the sidecar, and on
`DatasheetArtifacts`, for the in-memory gate. Adding it to only the record leaves
the hole described under "A missing client is a capability, not a defect" -- the
same instance keeps serving its caption-less artifact from memory, never reaching
the sidecar logic that would have rebuilt it.

#### A missing client is a capability, not a defect

The obvious move -- append a `figure_captions_no_client` note, set
`llm_enrichment_incomplete`, refuse reuse -- mirrors `toc_fallback_no_client`
(`index.py:582-586`) and is **wrong here**. A plain `uv sync` excludes the `[llm]`
extra, so `_try_create_default_llm_client` raises `ImportError` and returns `None`
on the *default* installation. Every document with a raster region would then be
marked incomplete and rebuilt on every request, forever, for the majority of
users. `llm_enrichment_incomplete` exists to stop a **transient** failure being
cached permanently; a machine with no `openai` installed is not transient, and
treating a stable environment as a defect destroys the reuse 0.24.0 just shipped.

So capability is recorded rather than flagged. `figure_captions_pending` is the
number of candidates left uncaptioned because no vision-capable client was
available. Reuse is refused only when that count is non-zero **and** vision
capability is available now:

| `figure_captions_pending` | vision available now | outcome |
|---|---|---|
| 0 | either | reuse |
| > 0 | no | **reuse** -- nothing has changed, and a rebuild would produce the same artifact |
| > 0 | yes | rebuild, and caption |

A keyless machine therefore reuses its artifacts indefinitely, and the moment
credentials appear the artifact is rebuilt with captions -- without anyone
hand-clearing the output directory.

**It counts eligible candidates, not all of them.** The count is taken *after*
`caption_figures` and `max_figure_captions` are applied: it is the number of
regions that would have been captioned had a client existed. With
`caption_figures=False` or `max_figure_captions=0` it is therefore **0**, never
the candidate count. Counting all candidates instead would mark work pending that
the caller explicitly declined, and the rule above would then refuse reuse on
every build that has a key -- an artifact rebuilt forever precisely because the
caller asked for no captions. Regions dropped by the cap are likewise not pending;
they are excluded and disclosed by `figure_captions_excluded`, and a rebuild would
drop them again.

**The same rule governs the in-memory cache, not only the sidecar.** A keyless
build leaves `_artifacts` populated and *complete* -- the whole point of not
flagging it -- so the gate at `bound.py:221-233`, which tests
`llm_enrichment_incomplete`, would hand back the caption-less artifact on the next
call even after credentials appeared. That is the same class of bug as the
incomplete-artifact hole closed in 0.24.0, reintroduced through a field that gate
does not know about. So `figure_captions_pending` is a field on
**`DatasheetArtifacts`** as well as on the sidecar record, and the in-memory gate
gains the identical `pending > 0 and vision available` condition. One rule, two
caches, stated once and applied in both places.

This is strictly better than the note-and-flag approach for the ToC fallback too,
which has the same defect today: a keyless machine never caches a weak-ToC
document. **Changing that is out of scope here** -- it is shipped behaviour with
its own tests -- but it is worth a follow-up, and this section is the argument
for it.

A caption call that *raises*, or returns empty, still sets
`llm_enrichment_incomplete` as above. The distinction throughout is
transient-versus-stable, not present-versus-absent.

#### The capability probe owns a client, and its lifecycle is explicit

An earlier draft of this section called the probe free. It is not.
`create_llm_client` builds a real HTTP client and its own docstring says the
caller owns it -- it exposes `close()` for exactly that reason. Probing by
construction therefore creates a resource, and probing on every reuse check would
leak one per check. Three rules, in order:

- **Probe last, and only when it can matter.** `reuse_blocker` stays a pure
  function of the record and gains no I/O: it reports a non-zero
  `figure_captions_pending` as a *signal*, not a blocker. `_reuse_from_disk`
  probes capability only after every cheap deterministic check -- version,
  incomplete flag, build options, size, hashes -- has passed and the pending count
  is non-zero. A version bump, a changed source, or a fully captioned artifact
  therefore constructs nothing at all, which is the common case on every path.
- **A supplied callable is never owned.** When the caller passed an
  `llm_callable`, capability is `getattr(callable, "describe_image", None)` --
  free, no construction, nothing to close.
- **A self-created probe is owned by `bound.py` until it is either closed or
  handed to `build()`.** Absent capability means `_try_create_default_llm_client`
  returned `None` and there is nothing to close. Present capability forces a
  rebuild under the rule above, and the probe is then **passed into `build()`**
  rather than discarded and reconstructed. `build()` leaves `owns_llm_callable`
  `False` for a passed-in callable, so ownership stays with `bound.py`, which
  closes it in a `finally` covering the build. Every other exit -- a
  deserialization failure discovered after the probe, an exception mid-check --
  closes it too.

The invariant worth testing directly is that **no path returns while holding an
unclosed probe**, and the reason for handing it to `build()` rather than closing
it is that this is what keeps "one client, not two" true across the reuse
boundary: constructing a client to answer "should I rebuild?", discarding it, and
having `build()` immediately construct another would double connection setup on
exactly the path already doing the most work.

**One probe per `build_datasheet` call, threaded through every stage.** The
in-memory gate, the disk check and `build()` are three independent construction
sites on one path: a populated `_artifacts` with pending captions, credentials
now present, and a sidecar that also has pending captions walks memory -> disk ->
rebuild and would construct a client at each step. The rules above make each
*stage* correct in isolation and still allow three clients in a row.

So capability is resolved by a single **per-call** lazy resolver, created at the
top of `build_datasheet` and closed in a `finally` around the whole call. It holds
one of three states -- not yet asked, resolved to a callable, resolved to `None`
-- constructs at most once, and records whether it owns what it returns: a
caller-supplied `llm_callable` is returned as-is and never closed, a self-created
client is closed exactly once at the end unless it was handed to `build()`, which
does not close what it does not own. Every stage asks the resolver instead of
calling `_try_create_default_llm_client` itself.

It is **per call, never an instance attribute.** Caching capability on the
instance would hold a connection pool open for the object's lifetime and, worse,
freeze the answer: credentials appearing between two calls on the same instance
is precisely the case the in-memory rule exists to catch, and a memoized `None`
would defeat it. Laziness is preserved either way -- a resolver nobody asks
constructs nothing, so the common paths still probe zero times.

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
  captions -- no candidate regions to name -- is complete, not incomplete.

**An empty response is a failure, not a caption.** A call can return `""`, or
whitespace, or a lone newline without raising -- a truncated stream, a refusal, a
model returning nothing for an unreadable render. Treated naively that either
writes an empty string into `caption` or leaves it null while the artifact is
marked complete, and the reuse design then caches that captionless artifact
permanently. So the response is `strip()`ped, and an empty result takes exactly
the path a raised call takes: `caption` stays null, a warning names the page and
region, and `llm_enrichment_incomplete` is set. The distinction that matters is
transient-versus-deterministic, and an empty response is transient.

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

**The implementation is Responses-API shaped, and the internal protocol must
widen.** The client calls `client.responses.create(model=, instructions=, input=)`
(`client.py:90`, `client.py:153`) -- the Responses API, not Chat Completions. Image
input there is a structured `input` list, not a string:

```python
input=[{"role": "user", "content": [
    {"type": "input_text", "text": prompt},
    {"type": "input_image", "image_url": f"data:{media_type};base64,{image_base64}"},
]}]
```

Two consequences. `_ResponsesApi.create` declares `input: str` (`client.py:55`),
so that annotation must widen to accept the list form -- `ty` runs over the whole
repo and will reject the call otherwise. And the content-part shapes differ
between the two APIs in a way that type-checks either way: on the Responses API
`image_url` is a **plain string**, while the far more commonly documented Chat
Completions form nests it (`{"type": "image_url", "image_url": {"url": ...}}`).
Copying the Chat Completions snippet produces valid Python that fails at the
gateway. The `media_type` parameter on `describe_image` exists to build the data
URI, and `inspect_page` already returns both the base64 payload and its
`mime_type`, so nothing needs to guess.

**Send `detail: "low"`, and it is a design choice rather than a tuning knob.**
`detail` is a sibling key on the `input_image` part -- `low`, `high`, `auto`
(the default), or `original`. Low detail resizes the image to 512x512 and costs a
flat, documented 85 tokens under the tile-based tokenization, against an
`auto`/`high` cost that scales with the region's pixel dimensions. Two reasons
beyond price:

- **It makes the cap's cost calculable.** A flat per-image token count means
  `max_figure_captions` translates directly into a bounded spend the tool
  description can state honestly, rather than a number that varies by however
  large the vendor's images happen to be.
- **It structurally enforces the no-transcription guard.** The prompt below
  forbids transcribing values, but a prompt is a request. At 512x512 a dense
  table's cell values are physically unreadable, so the model cannot fabricate
  rows from detail it never received. `research.md`'s warning about VLMs
  hallucinating table rows is answered by the input, not only by the wording.

**The risk this carries, and how to settle it.** The PCN's Product Attributes
table renders its own title *inside* the image -- that is why it has no text-layer
caption -- and a title legible at 1656px wide may not survive downscaling to 512.
If low detail cannot name the subject, the caption degrades to "a table" and the
motivating case is only half solved. So implementation must validate `low`
against **PCN page 5 specifically** before settling, and escalate to `high` with
the measured cost recorded here if it cannot. `client.responses.input_tokens.count`
prices an image input without paying for generation, so both options can be
measured before choosing. Confirm the 85-token figure for the configured model
the same way rather than assuming it.

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
a thread pool of **4 workers**, which is network I/O and safe to overlap. Four
rather than one-per-candidate: an unbounded pool at the default cap opens twenty
simultaneous gateway connections, which is how a shared LiteLLM deployment starts
rate-limiting, and the wall-clock gain past a handful of workers is small against
that risk.

**Results are applied in candidate order, never completion order.** Each caption
is written back to the entry of the candidate that produced it, and the `figures`
array keeps its deterministic page-then-position order regardless of which call
returned first. This is not cosmetic: `2026-07-25-on-disk-artifact-reuse-design.md`
fingerprints the artifact by hashing its bytes, so an array whose order followed
completion would hash differently on every build and defeat reuse outright, while
also making two builds of the same document gratuitously diff against each other.

Eight candidates on the PCN and 20 on the PSoC, so the concurrency is not
theoretical: at the cap, serial dispatch at typical multi-second VLM latency would
put tens of seconds inside one `build_datasheet` that an agent is blocked on, on
the *common* document rather than a pathological one. **Per-call latency is
unmeasured** -- neither fixture has been run against a live gateway for this -- so
implementation must measure it and record the real serial and concurrent numbers
here, exactly as section 7 requires for the scan cost. If the concurrent figure is
still large, lowering the default cap is the lever to reconsider, not removing the
bound.

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
- **Vector figure detection by clustering drawing operations.** Unreliable per the
  numbers above, and largely unnecessary since vector figures already leak their
  text.
- **ML layout analysis (`pymupdf.layout`), for this release.** This is the
  technically superior mechanism and it is rejected on cost, not on capability, so
  the evidence is recorded rather than the verdict alone. The engine classifies
  regions with DocLayNet labels -- `caption`, `picture`, `table`,
  `section-header` -- and on these documents it works: it labels TI's
  section-relative captions with no pattern at all, finds the PCN's raster table,
  and finds a `picture` on PSoC page 32, **a page with zero raster images**. That
  last one is the vector-figure blind spot nothing else in this design addresses.

  Measured warm, after model load: 1.28 s/page over 5 pages, 0.89 s/page over 20,
  extrapolating to **~119 s for the 134-page PSoC** against a current ~8 s build.
  Roughly 15x, in the default path, for every document. It also requires the 49 MB
  `[layout]` extra a plain `uv sync` deliberately excludes, brings the
  process-global hook hazard documented in `core/engine.py`, and unlike
  `get_image_info` is a model with an error rate rather than an exact enumeration.

  **The project already ships this engine** -- `extract_table_markdown` runs
  `layout_engine()` (`tools/bound.py:185`) and `mcp_server.py:51` pre-warms it --
  which is exactly the shape that makes it affordable: one page, on demand, when
  the agent asks. That, not a build-time sweep, is how figure captioning by layout
  analysis should arrive if it does: an on-demand `describe_figure(page)` tool
  mirroring `extract_table_markdown`, with its own spec. Deferred, not dismissed.

- **Caption detection from PDF structure tags or a List of Figures.** Both would be
  exact rather than heuristic, and neither exists in practice: of 14 documents,
  **1** carries a structure tree (with zero `/Figure` elements) and **none** has a
  List of Figures. Outline entries naming figures appear in exactly one document.
  Datasheet publishers do not ship structure metadata, so there is nothing exact to
  read and the pattern in section 2 is the best available source.
- **Triaging which regions get a VLM caption**, by either of the two rules that
  looked workable. Comparing per-page caption and raster counts is section 4's
  forbidden association reached by arithmetic; thresholding page text does not
  discriminate, because the PSoC's raster pages are measurably *more* text-starved
  than the PCN's. Section 6 carries the numbers. The cap replaces triage as the
  cost control, and unlike triage it cannot silently skip a blind region.

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
-- or self-creates -- a vision-capable client will begin issuing VLM calls for
every raster region above `min_area_pct`, up to the cap. On the PSoC that is 20
calls on the first build of the document.

State it as the cost it is rather than minimizing it. What bounds it: the
deterministic index is byte-identical either way, `max_figure_captions` is a hard
ceiling the caller controls, disk reuse charges it once per document rather than
once per build, and `caption_figures=False` or `max_figure_captions=0` restores
today's behaviour exactly. A caller who wants the old default back has two ways to
say so and neither costs them anything else.

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
  emitted `bbox` describe the visible rectangle, not the placement. This case is
  **real, not hypothetical** -- `ti-tlv9061` has 9 such placements -- so a
  regression here breaks a shipping vendor's datasheet, not an imagined one.
- **An image entirely off-page is dropped**, not emitted with a degenerate region
  that `inspect_page`'s `top < bottom` check would reject.
- **A page whose rect origin is not `(0, 0)`** (a CropBox offset) normalizes
  relative to that origin: the emitted region, fed back through `inspect_page`,
  crops the same rectangle. Synthetic -- every page in both fixtures starts at
  `(0, 0)`, so nothing else would catch an absolute-coordinate mistake.
- **The split caption form**: a page whose text has `Figure 2` alone on a line and
  `Block diagram` on the next yields one entry with `figure_number: "2"` and
  `caption: "Figure 2 Block diagram"`, `caption_source: "text"`. This is the form
  24 of the PSoC's captions take.
- **The same-line form**, with mandatory punctuation: `Figure 3. Package outline`
  yields that caption. This is the corpus's most common form, 404 of 492.
- **Section-relative numbering, in both forms.** `Figure 10-1. Reset Logic` yields
  `figure_number: "10-1"`, and a bare `Figure 3-2` with its title on the next line
  yields `"3-2"`. Include an en-dash variant (`Figure 10–1.`), since publishers use
  both characters. Without these the pattern finds captions in 2 of 14 corpus
  documents instead of 11, so this is the highest-value test in the section.
- **`figure_number` is a string in the emitted JSON**, for plain numbers too --
  assert `"2"`, not `2`. A consumer that branches on the type is the thing this
  prevents, and a plain number is where an int would silently creep back in.
- **`Fig.` is accepted wherever `Figure` is**, in both forms; the corpus's
  same-line plain captions are all `Fig. 10. Enable and disable times`.
- **Prose is not a caption, four ways.** `as Figure 5 shows` mid-line, a line
  *opening* `Figure 2 shows the major subsystems`, `Figure 6-2 shows the structure
  of the 32 general purpose registers` (the section-relative form of the same
  trap), and `See Figure 3` each yield **zero** entries. The mandatory separator is
  what excludes all four. Across the corpus this rule rejected 70 prose lines while
  admitting 404 real same-line captions, so these are the tests that matter most in
  this section -- publishing prose as a caption puts a wrong figure name in the
  artifact, and widening the number pattern widened the prose surface with it.
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
- **A region on a page that already carries a caption is still captioned.** One
  image, one `Figure N` caption, defaults everywhere: assert `describe_image` was
  called and the region carries an `llm` caption beside the separate `text`
  caption entry. This pins the no-triage decision, and it is the direct
  counterpart of the no-merge test above -- a future change that starts skipping
  "already captioned" regions fails here rather than silently going blind.
- **`caption_figures` defaults True**: a raster region plus a
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
- **A keyless build is reused, not rebuilt.** Build with no vision client on a
  document with candidates, assert `figure_captions_pending` is non-zero and the
  artifact is **not** `llm_enrichment_incomplete`, then build again with no client
  and assert the sidecar was reused and no rebuild ran. This is the regression
  test for the default `uv sync` installation, where `[llm]` is absent -- without
  it, every user on the default install rebuilds every document forever.
- **Capability appearing invalidates the artifact.** Same document, same output
  directory, second build *with* a vision client: assert reuse is refused, the
  rebuild captions, and `figure_captions_pending` drops to zero. Then a third
  build reuses. The three together are the whole capability contract, and the
  middle one is what stops a keyless artifact outliving the credentials that
  would fix it.
- **`figure_captions_pending` of zero reuses under both capability states**, so a
  fully captioned artifact is never rebuilt merely because the client went away.
- **The in-memory cache obeys the same rule, on one instance, with exactly one
  probe.** Build keyless on a single `DatasheetTools`, make vision capability
  appear, and call `build_datasheet` again on that *same* instance: assert it
  rebuilds and captions rather than returning `_artifacts`. The test must not
  create a second instance -- doing so silently tests the disk path instead. Then,
  with the construction/close recorder attached, assert **exactly one construction
  and exactly one close** across that call. This is the full memory -> disk ->
  rebuild path, the only one that visits all three probe sites, so a resolver that
  is merely per-stage rather than per-call fails here with three constructions and
  nowhere else. It is also the direct analogue of the retry-on-the-same-instance
  test 0.24.0 added for `llm_enrichment_incomplete`.
- **`figure_captions_pending` counts only eligible candidates.** With no vision
  client: `caption_figures=False` yields 0, `max_figure_captions=0` yields 0, and
  22 candidates under a cap of 20 yields 20 rather than 22. Then assert the first
  two artifacts are reused *even with capability present* -- the regression this
  guards is an artifact rebuilt forever precisely because the caller asked for no
  captions.
- **No path returns holding an unclosed probe.** Wrap
  `_try_create_default_llm_client` with a recorder that tracks every construction
  and every `close()`, and assert the counts match after: a reuse hit, a
  capability-triggered rebuild, and a reuse attempt that fails deserialization
  after the probe. `create_llm_client` builds a real HTTP client, so a leak here
  is a leaked connection pool per call.
- **The probe is not constructed when it cannot matter.** With the same recorder,
  assert zero constructions when `figure_captions_pending` is 0, and zero when a
  cheaper check -- a version mismatch or a changed source hash -- already blocks
  reuse. This pins the probe-last ordering rather than leaving it to reviewer
  vigilance.
- **A capability-triggered rebuild reuses the probe client.** Assert `build()`
  received the same object the probe constructed and did not create a second, and
  that `owns_llm_callable` stayed `False` so ownership remained with `bound.py`.
  This is the "one client, not two" guarantee across the reuse boundary.
- **A `describe_image` that raises leaves the build successful**, with `caption`
  null and a warning logged -- the text-only-model case -- **and sets
  `llm_enrichment_incomplete`**, so the caption-less artifact is not cached
  permanently. Distinguish it from the keyless case above with a stub that raises
  on some regions and succeeds on others: the artifact is incomplete even though
  `figure_captions_pending` is zero.
- **An empty or whitespace-only `describe_image` response is a failure**, not a
  caption. Return `""`, then `"   \n"`: `caption` stays null both times, a warning
  is logged, and `llm_enrichment_incomplete` is set. Assert specifically that
  `caption` is `None` rather than an empty string, since an empty string in a
  published artifact reads as "the model said this figure has no description".
- **A negative `max_figure_captions` raises rather than slicing.** Assert
  `ValueError` from `build()`, `build_datasheet()` and `build_batch()`, a non-zero
  exit from the CLI, and `"minimum": 0` present in the tool schema. Then assert
  `max_figure_captions=0` is accepted and yields no captions with the deterministic
  index intact -- the boundary the guard must not swallow.
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
- Caption forms beyond the two in section 2. Section-relative numbering
  (`Figure 3-2`) has moved **into** scope on corpus evidence -- it was the single
  defect the wider measurement found. What stays out: `Table N`, `Diagram N`, and
  non-English labels. `Table N` is not a small omission -- **427 occurrences across
  the corpus** -- but table structure is already served by `table_count` and the
  continued-table enrichment, so a parallel label index would duplicate an existing
  signal rather than fill a gap. Non-English labels remain uncalibrated: all 14
  corpus documents are English, so widening there would repeat exactly the mistake
  this revision corrected.
- Correlating a caption entry with a raster entry on the same page (section 4).
- Any new model configuration. Figure captions use the callable the caller
  already provides, on the same model as the ToC fallback and summaries.
- Figure indexing for the page-matched text file. The text file mirrors what is
  extractable; a raster region has nothing to contribute to it, and the ToC JSON
  is where structural metadata belongs.
