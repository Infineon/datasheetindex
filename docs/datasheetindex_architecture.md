# DatasheetIndex: Agent-First Parameter Extraction from Technical Datasheets

## Philosophy

**The library doesn't extract parameters. The agent does.**

`datasheetindex` has two jobs:
1. **Pre-processing:** Generate an enriched, structured ToC (JSON) and a page-matched text file from a PDF datasheet
2. **Tooling:** Provide the agent with sharp, focused tools for when the text file isn't enough

All intelligence — deciding where to look, when to escalate, how to validate — lives in the agent. The library gives the agent the best possible starting context and the right tools to handle edge cases.

---

## Why This Matters

The Claude Agent SDK already provides built-in tools to read sections of files and search within files. If we give the agent:
- A well-structured JSON map of the document (enriched ToC)
- A clean text file of the full document (with correct page markers)

...then for most well-structured datasheets, the agent can navigate and extract parameters using just these built-in capabilities. The custom tools only get called when the agent encounters problems — a garbled table, a figure it needs to see, a footnote it needs to resolve.

This keeps the common path fast and cheap, and only spends on expensive operations (table re-extraction, vision) when the agent judges it necessary.

---

## Conventions

**All page numbers are 1-indexed.** Page 1 is the first page of the PDF, matching what a human sees in a PDF viewer.

This applies everywhere:
- JSON fields: `start_page`, `end_page`, `total_pages`
- Text file markers: `--- PAGE 1 ---`, `--- PAGE 2 ---`, ...
- Tool parameters: `inspect_page(page=22)` means page 22
- Agent output: `"source_page": 9` means page 9

PyMuPDF uses 0-indexed access internally (`doc[0]` = page 1), but this is an implementation detail that never leaks into the public API or output artifacts. The conversion happens once during extraction:

```python
for page_idx, page in enumerate(doc):
    page_num = page_idx + 1  # 1-indexed, used everywhere
```

This convention matches PyMuPDF's `get_toc()`, which returns 1-indexed page numbers.

**Text file size.** A 73-page datasheet produces ~50KB of text; a 300-page datasheet may produce 300KB+. The agent is expected to read slices of the text file (by page range), not load the entire file into context. The Claude Agent SDK's built-in file reading tools support this natively.

**The table engine is process-global; `core/engine.py` owns it.** PyMuPDF's `find_tables()` consults a module global, `pymupdf._get_layout`, on *every* call. Importing `pymupdf4llm` (the optional `[layout]` extra) installs an ONNX-backed callable there, for the whole process. So which table engine runs is a property of the process, not of the document.

`core/engine.py` is therefore the only module under `src/` allowed to import `pymupdf4llm` or to read or write that hook. It exposes two context managers, serialized on one re-entrant lock:

- `classic_tables()` — suppresses the hook, pinning `find_tables()` to PyMuPDF's classic geometric detector. Both table-counting paths (parallel workers and the sequential fallback) scan inside it, which is why `table_count` is a stable property of the document rather than of the indexing process.
- `layout_engine()` — imports `pymupdf4llm` **inside** the lock and yields it with its hook installed. Only `extract_table_markdown` uses it.
- `layout_active(module)` — whether `to_markdown` will take the *layout* branch. Reaching `layout_engine()` does not imply it: `to_markdown` dispatches on the module global `_use_layout` at call time, and a pymupdf4llm whose own `import pymupdf.layout` failed (an unimportable `onnxruntime` suffices) imports fine with it `False` and routes to the classic renderer. That renderer swallows unknown keywords into `**kwargs` and `print()`s a notice naming them to **stdout** — the MCP stdio transport's JSON-RPC channel — so a layout-only keyword must be withheld there. `extract_table_markdown`'s `header=`/`footer=` are gated on this.

The import must happen inside the lock, because the import *is* the hook installation. An import racing `classic_tables()` lets the guard save `None`, the import install the hook, and the guard restore its stale `None`. Since `pymupdf4llm._use_layout` stays `True`, `to_markdown()` then iterates a `None` `page.layout_information` and raises `TypeError` — permanently, because the module is cached in `sys.modules`. `build_datasheet` and `extract_table_markdown` both run under `asyncio.to_thread`, so this race is reachable.

The two engines are different heuristics, not better and worse. On a real 68-page datasheet the classic detector finds 75 tables to the ML engine's 39 and is ~4.4x faster; the ML engine's extra misses are the "Typical Characteristics" plot pages, where the classic detector false-positives on chart gridlines. Counts are defined as the classic detector's answer so they do not depend on which optional extras happen to be installed.

---

## The Two Deliverables

### Deliverable 1: Enriched ToC JSON

Not just a flat table of contents — a hierarchical tree with enough metadata for the agent to make informed navigation decisions. Includes a **preamble** — page-marked raw text from pages 1-2 — so the agent can orient itself before extraction.

The `preamble` is generated automatically with zero LLM calls. Rather than fragile heuristics to detect product names or classify ToC entries (which break across manufacturers), the library embeds the raw text of pages 1-2 as a `preamble` — giving the agent the context to orient itself.

`build_front_matter(doc, *, max_pages=2, max_chars=5000)` produces it. Each page is introduced by a `--- PAGE N ---` marker in the same format as the page-matched text file, so every line is attributable and citable, and each cap that bites appends its own `=== NOTE: ... ===` line — truncation is disclosed, never silent. `max_chars` bounds *document text* (`chars_shown`), not the returned string: markers and notes are tool framing, and counting them against the budget would make the amount of content a caller receives depend on how long the framing happens to be. The companion top-level key `preamble_pages` reports per-page evidence (`chars`, `bullets`, `has_features_heading`) computed on the whole page read, not the fragment shown, so truncation cannot skew it. They are heuristic counts to be weighed, not thresholded on — see "Decisions already settled by measurement" below for what each one has and has not been measured to do.

Why not parse it programmatically? Because:
- Part number regex produces false positives ("JEDEC51", "AEC100") and misses wildcards ("TPS6513x")
- ToC keyword matching is manufacturer-specific ("Operating ranges" vs "Functional range" vs "Recommended operating conditions")
- An LLM call would work but adds a dependency to pre-processing (the happy path currently requires zero LLM calls)

The agent IS the LLM — let it reason about the preamble text directly.

#### Decisions already settled by measurement

Every number in this subsection comes from one **21-document, 1047-page,
six-vendor corpus** (Diodes, Infineon, Microchip, Nexperia, onsemi, TI): 14
datasheets and 7 product-change notices, 2 to 294 pages, swept with
`build_front_matter` at the defaults and re-swept after every change to the
extractor. The sweep also checks the invariants -- marker order, the framing
formula, notes agreeing with `char_truncated` / `pages_omitted`, per-page
`chars` matching the full extracted page -- and reports zero failures.

**Skipping a cover or legal page is rejected.** Detecting front matter that
is not front matter and dropping it was considered. The error is asymmetric:
wrongly skipping page 1 of a real datasheet costs the general description and
half the features -- the most valuable page in the document -- while wrongly
keeping a cover page costs some tokens. The 21-document corpus does separate
the two classes -- all seven product-change notices score 0 bullets and no
features heading on page 1, while 13 of the 14 datasheets have a page-1
features heading -- but that is an argument for publishing the signals, not for
acting on them: a library that skips forecloses the alternative, and a caller
given the signals can implement skipping itself. So `preamble_pages` reports
and the agent decides. This is the same shape of decision as the table-engine
note in `CLAUDE.md`: stability is the point.

**The 5000-character budget is measured, and it does cut 8 of the 21
documents.** 13 fit whole; the widest of those, the PSoC 6 at 4746 characters
over pages 1-2, uses 95% of the budget. The cut 8 are six of the seven TI
datasheets, whose pages 1-2 run 5332-7404 characters, and two of the four
onsemi product-change notices; between 170 and 2424 characters fall past the
cut. The default stays at 5000 for what it *keeps*, not in spite of what it
drops: on the TI documents the general description and the features list are
inside the budget, and what the cut loses is page 2's table of contents, its
revision history and its copyright footer -- on the TPS54331, lines like
"Updated the inductor current equations for IL(RMS) and IL(PK)" and "Product
Folder Links: TPS54331". Two honest exceptions: the MSP430F5529 also loses the
tail of its general description and its Device Information package table, and
the onsemi IPCN26979Z loses the continuation of a qualification-test table.
Neither is silent -- the `=== NOTE: preamble truncated at 5000 characters ===`
line names the cut, and `max_chars` is the caller's to raise -- and raising the
default to cover them would spend the budget of every document on the tail of
a few.

**Unit density is deliberately not a signal.** A count of numeric-plus-unit
tokens looks like the obvious third signal. A naive ASCII pattern undercounts
badly -- 5 matches on PSoC page 1 -- because it misses `150-MHz` (hyphen
separator), `1.1-V`, and `40 uA` (micro sign, which needs both U+00B5 and
U+03BC). The corrected pattern then false-positives on part numbers, which
datasheets are full of: `8/A` from `CY8C62x8/A`, `4F` from `Cortex-M4F`. No
count for a corrected pattern is quoted here on purpose: the pattern was never
kept, so the figure cannot be re-derived, and an unreproducible number in a
tracked doc is worse than none. Noisy in both directions, and `bullets` plus
`has_features_heading` already do as much separating as has been asked of them:
all seven product-change notices score 0 bullets and no features heading on page
1, and 13 of the 14 datasheets report a page-1 features heading. That separation
runs in one direction only -- the Infineon IRF540N is a datasheet scoring 0
bullets and no heading on *both* pages, indistinguishable from a cover letter on
these two signals -- and it is `has_features_heading` that carries it: page-1
`bullets` is 0 on three of the fourteen datasheets (Diodes AH1751, Infineon
IRF540N, Microchip ATmega328P), so the PSoC 6's 34 and 43 illustrate what the
count looks like on a dense features page rather than measure what it
discriminates. Add unit density later if a consumer needs it, calibrated against
part-number forms.

**A legal-vocabulary count was designed, built, measured and then not
shipped.** A third signal, `legal_hits`, counted disclaimer vocabulary
(`warranty`, `liability`, `trademarks`, the `provided "as is"` idiom) in a
page's prose, to mark front matter that is a cover letter rather than
specifications. On the 21-document corpus it measured *anti-correlated* with
that purpose. It scores 2-3 on page 1 of all seven TI **datasheets** --
`warranty` and `disclaimers` from TI's standard page footer, plus `Copyright`
and `Trademarks` on page 2 -- and **0 on pages 1 and 2 of every one of the
seven product-change notices**, including the TI PCN the design was written
from, whose footer reads "TI Information - Selective Disclosure" and a
disclosure classification is not a liability disclaimer. A high count meant
"this is a TI datasheet", not "this page is boilerplate rather than
specifications", which is the opposite of what it was built to say. It was
deleted before release rather than kept: a field measured in the wrong
direction is worse than no field, and removing a published one afterwards is
breaking. The signal is easy to re-propose, so the measurement is recorded
here as the answer.

**Both signals are heuristic counts, so do not threshold on exact values.**
The patterns behind `bullets` and `has_features_heading` are calibrated against
what has been measured and will change as more documents are measured -- the
`bullets` marker pattern was widened once before release, when the corpus
showed it was missing most of Infineon's markers, and `has_features_heading`
was taught to strip a leading section number when the corpus showed all seven
TI datasheets write `1 Features`. `preamble_pages` is an additive key whose
field *names* and types are a compatibility surface; the numbers in it are
evidence for an agent to weigh, not a stable API.

#### Example artifact

```json
{
  "source": "infineon-tle9009dqu-datasheet-en.pdf",
  "total_pages": 73,
  "preamble": "--- PAGE 1 ---\nTLE9009DQU\nLi-ion battery monitoring and balancing IC\n\nFeatures\n• Voltage monitoring of up to 9 battery cells connected in series\n• Hot plugging support\n• Dedicated 16-bit high precision delta-sigma ADC for each cell...\n--- PAGE 2 ---\n• Integrated cell balancing switches\n• Operating temperature -40 to +105 C\nSpecifications are subject to change without notice...\n=== NOTE: preamble covers pages 1-2 of 73; later pages were not examined ===",
  "preamble_pages": [
    {"page": 1, "chars": 1954, "bullets": 22, "has_features_heading": true},
    {"page": 2, "chars": 2087, "bullets": 15, "has_features_heading": false}
  ],
  "toc": [
    {
      "node_id": "0001",
      "title": "1 Block diagram",
      "level": 1,
      "start_page": 5,
      "end_page": 5,
      "has_tables": true,
      "table_count": 2,
      "breadcrumb": "1 Block diagram",
      "nodes": []
    },
    {
      "node_id": "0002",
      "title": "2 Pin Configuration and Description",
      "level": 1,
      "start_page": 6,
      "end_page": 9,
      "has_tables": true,
      "table_count": 3,
      "breadcrumb": "2 Pin Configuration and Description",
      "nodes": [
        {
          "node_id": "0003",
          "title": "2.1 Pin Assignments",
          "level": 2,
          "start_page": 6,
          "end_page": 7,
          "has_tables": true,
          "table_count": 2,
          "breadcrumb": "2 Pin Configuration and Description > 2.1 Pin Assignments",
          "nodes": []
        }
      ]
    },
    {
      "node_id": "0010",
      "title": "5 Electrical Characteristics",
      "level": 1,
      "start_page": 22,
      "end_page": 45,
      "has_tables": true,
      "table_count": 14,
      "breadcrumb": "5 Electrical Characteristics",
      "nodes": [
        {
          "node_id": "0011",
          "title": "5.1 Absolute Maximum Ratings",
          "level": 2,
          "start_page": 22,
          "end_page": 23,
          "has_tables": true,
          "table_count": 5,
          "breadcrumb": "5 Electrical Characteristics > 5.1 Absolute Maximum Ratings",
          "nodes": []
        },
        {
          "node_id": "0012",
          "title": "5.2 Operating Conditions",
          "level": 2,
          "start_page": 24,
          "end_page": 30,
          "has_tables": true,
          "table_count": 8,
          "breadcrumb": "5 Electrical Characteristics > 5.2 Operating Conditions",
          "nodes": []
        }
      ]
    },
    {
      "node_id": "0020",
      "title": "21 Revision history",
      "level": 1,
      "start_page": 72,
      "end_page": 72,
      "has_tables": true,
      "table_count": 1,
      "breadcrumb": "21 Revision history",
      "boilerplate_category": "revision",
      "nodes": []
    }
  ]
}
```

Note: `has_tables` and `table_count` are heuristic hints from PyMuPDF's classic
geometric table detector (false positives on block diagrams and on plot
gridlines are expected). They are identical whether or not the optional
`[layout]` extra is installed, and whichever internal scan path runs, so the
count is a stable property of the document. The ML layout engine is used only by
`extract_table_markdown`. `source` is always the filename, not the full path. In
this example, node "0001" has `table_count: 2` which are false positives from
block diagram boxes.

**`breadcrumb` semantics:** Pre-computed full ancestry path joined by `" > "`, including the node's own title. Lets downstream agents and RAG indexers see structural context without re-traversing parents. Computed once in `assign_breadcrumbs()` during `build_tree()`, so the LLM ToC fallback path gets it too. Omitted from JSON only when empty -- which happens for a bare `TocNode` constructed in isolation (e.g. legacy code) or a node with an empty title and no ancestry.

**`boilerplate_category` semantics:** Title-pattern classification into one of six categories -- `legal`, `ordering`, `revision`, `contact`, `toc`, `glossary` -- so agents can deprioritize disclaimers, ordering tables, revision histories, contact lists, ToC/index pages, and glossaries during navigation. Empty when the title doesn't match any known boilerplate pattern (the common case for substantive sections). Subsections of a boilerplate-flagged parent inherit the parent's category. No LLM call, no text scan -- title-only regex matching after normalization strips leading section numbering (`12.3.4`, `Appendix A:`, `Chapter 3`) and trailing punctuation. The agent can choose how strict to be: skip flagged sections entirely, scan them last, or ignore the field.

**What's included (deterministic, no LLM needed):**
- Hierarchical structure with node IDs and page ranges
- Nesting level

**Computing `end_page`:** PyMuPDF's `get_toc()` returns `[level, title, start_page]` — no end page. The library computes `end_page` for each node:

- A node's `end_page` is the page before the next sibling's `start_page`
- If no next sibling, it inherits the parent's `end_page`
- The last top-level section extends to `total_pages`

```python
# Flat ToC from get_toc():  [level, title, start_page]
# [1, "Block diagram",              5]
# [1, "Pin Configuration",          6]
# [2, "Pin Assignments",            6]
# [2, "Pin Description",            8]
# [1, "Electrical Characteristics", 10]
#
# Result:
# "Block diagram"              start=5,  end=5   (next L1 sibling starts at 6)
# "Pin Configuration"          start=6,  end=9   (next L1 sibling starts at 10)
#   "Pin Assignments"          start=6,  end=7   (next L2 sibling starts at 8)
#   "Pin Description"          start=8,  end=9   (inherits parent's end_page)
# "Electrical Characteristics" start=10, end=73  (last section, uses total_pages)
```

**`table_count` semantics:** The count covers all tables detected by PyMuPDF on pages `start_page` through `end_page` for that node. For parent nodes, this is the sum across all pages in range (including pages covered by children). The agent uses this as a rough navigation hint, not an exact count.

**Table/figure detection (benchmarked across 3 libraries):**

We benchmarked table detection across three libraries (PyMuPDF, pdfplumber, pymupdf4llm) on the TLE9009DQU datasheet. All three have false positives on diagram pages (block diagram boxes detected as table cells) and none can detect vector figures. The differences between them are marginal — not worth adding extra dependencies or processing time.

**Decision: Use PyMuPDF only.** The `has_tables` flag is just a navigation hint — the agent reads the actual section text and can judge for itself whether a table is present. When text looks garbled, the agent calls `inspect_page`. It doesn't need a perfectly accurate metadata flag to make that decision.

**Raster figures are enumerated exactly; vector figures still are not.** `get_image_info()` reads the PDF's image XObjects and returns real bboxes and pixel dimensions -- nothing is inferred, so there is no false-positive rate to calibrate for that half. The top-level `figures` array (see "Figure indexing" below) is how the agent learns a raster region exists. Clustering vector *drawing operations* to find figures remains unreliable and out of scope -- 232 vector drawings on a pinout page could be one diagram or fifty -- but that blind spot matters less than it sounds: vector figures leak their text (note text and pin labels extract normally), so the agent is not blind there, only unaware of the layout.

**What's optionally included (LLM-powered, debatable):**
- `summary` per node — useful for very large datasheets (300+ pages) where the agent needs help deciding which of 50 sections to look at. For smaller datasheets (< 100 pages), the section titles alone are usually descriptive enough. This should be configurable.

**The decision on summaries:** If the ToC is high quality (descriptive titles, proper hierarchy, correct page numbers), summaries add cost without much value. If the ToC is sparse or uses cryptic section numbers, summaries become essential. The library should score ToC quality and recommend whether summaries are worth generating.

### ToC quality: what the score decides, and how a caller overrules it

`assess_toc_quality` produces one number with exactly one consumer:

```python
needs_toc_fallback = regenerate_toc or toc_quality.score < TOC_FALLBACK_THRESHOLD  # 0.3
```

It decides whether to spend an LLM call rebuilding the table of contents.
Everywhere else the score appears -- on `DatasheetArtifacts`, in the ToC JSON,
in `get_artifact_manifest` -- it is information rather than control.

Four weighted factors, gated by a fifth that multiplies:

```
score = (0.3*entry + 0.3*coverage + 0.2*depth + 0.2*title) * informativeness
informativeness = distinct(normalize_key(breadcrumb or title)) / entry_count
```

`normalize_key` is `core/furniture.py`'s -- whitespace collapsed, digit runs
masked to `#` -- so `Page 1` and `Page 2` share one key. Both `furniture.py` and
`quality.py` are pure (no PyMuPDF, no environment, no I/O), so the reuse adds no
coupling; a comment above `normalize_key` names this second consumer, because a
change to the masking rules moves ToC quality scores as well as furniture
detection.

**The gate is appealable.** `build_datasheet(regenerate_toc=true)`, and
`DatasheetIndex.build(regenerate_toc=True)` beneath it, forces the fallback
whatever the score says. It exists because the manifest already hands the agent
both `toc_quality` and the full `toc` tree -- so the agent can see an outline is
useless while the library insists it is fine -- and `force_rebuild` is not the
lever: it busts the cache and then re-runs the same deterministic scoring to the
same conclusion. Four details carry the weight:

- **It is in the artifact-reuse key** (`_BuildOptions`), because it changes
  artifact *content*. Same lesson as `DATASHEETINDEX_FURNITURE` in 0.33.0: an
  option absent from the key means the rebuild serves the stale artifact, and
  `json_sha256` still agrees because it hashes that same stale file.
- **It overrides the score comparison in `_accept_llm_toc_candidate`, and only
  that one.** An enumerated outline scores 0.680, so requiring the replacement to
  beat the baseline would let the broken number veto its own repair. The
  non-empty, entry-floor and coverage-regression guards still apply: they protect
  against a degenerate *candidate*, which an explicit request says nothing about.
- **Without a client it raises** rather than returning the un-regenerated ToC. A
  tool that appears to ignore the parameter invites the agent to retry forever.
  The guard is in **two** places, and the second is not redundant.
  `DatasheetIndex.build` raises for the Python API path, but by the time it runs
  `DatasheetTools._build_or_reuse` has already called `remove_sidecar` -- so a
  keyless `regenerate_toc=True` used to destroy the sidecar of a perfectly good
  cached artifact on its way to failing, costing a full rebuild in the next
  process. `build_datasheet` therefore resolves the requirement up front, with
  the same `_VisionResolver` the build would have used, before any invalidation.
  Message text lives once, in `index.REGENERATE_TOC_REQUIRES_CLIENT`.
- **The description names the next action** and the parameter is in the tool's
  `input_schema`, not only in the Python signature -- an MCP host validates
  against the schema, so a Python-only parameter is unreachable.

**Transient versus stable, in the artifact cache.** A build that wanted the
fallback and found no LLM client records `toc_fallback_pending` on the artifact
and the sidecar. It deliberately does **not** feed `llm_enrichment_incomplete`,
which `reuse_blocker` treats as a reason to refuse the artifact: rebuilding
cannot create credentials, so that made a bookmark-less PDF on a credential-free
install -- the default install, since `[llm]` is an extra -- rebuild on every
request forever. `toc_fallback_pending` follows `figure_captions_pending`
instead: `reuse_blocker` stays pure and ignores it, and the *caller* resolves it
with a real capability probe, reusing while no client exists and invalidating the
moment one appears. The probe is `_VisionResolver.has_client()` rather than
`get()`, because `get()` returns the vision-filtered client while the ToC
fallback needs any text client. `toc_fallback_raised` and `figure_caption_failed`
stay transient -- those are genuine failures worth retrying.

One failure is carved out of that, in one direction only. A TLS certificate the
local trust store rejects raises `LlmTlsVerificationError` out of the ToC
fallback and out of `build()`, rather than becoming `toc_fallback_raised`: it is
permanent, it is fixed in one variable (`LITELLM_TLS_VERIFY`, or the trust
store), and its note would otherwise sit unread inside an artifact whose ToC is
empty for a reason indistinguishable from a document that genuinely has no
outline. **Figure captioning is deliberately NOT carved out.** Captioning runs
at step 6b and the artifacts are written at step 8, so raising there would
abort the build and write nothing at all -- for a document whose index, and
possibly whose ToC, is otherwise complete. `figure_caption_failed` therefore
still absorbs a TLS failure; only the logged message gets better. Summaries are a
third case and a pre-existing one: `add_summaries` has no exception handler at
all, so any error there propagates out of `build()` -- unchanged here, and only
reachable with `include_summaries=True`.

#### Decisions already settled by measurement

Every number here comes from one **7-vendor corpus** (TI, Espressif, Bosch,
Microchip, Raspberry Pi, Infineon, Vishay), 5 to 784 pages. **24 documents have a
ToC and are scored**, and every scored figure below comes from those 24. ST, NXP
and Analog Devices blocked scripted download and are absent.

The unscored tail is smaller than first recorded. The original run reported 26
documents of which 2 had no ToC; re-measured later, the corpus on disk holds 25
of which **1** has none (`vishay_1n4001`). The scored set of 24 is identical
either way, so every score below stands; only the count of bookmark-less
documents changed, and it moved *down*.

- **Repairing `title_score` is insufficient, and was measured before being
  rejected.** Its check rejects only *purely* numeric titles, so `"Page 1"` scores
  perfectly -- but forcing the entire factor to zero still leaves the enumerated
  outline at **0.48** (134 pages) to **0.66** (20 pages). Three of the four
  factors, 80% of the weight, measure whether the outline *spans* the document,
  and enumeration maximises all three by construction: one entry per page is a
  plausible entry count and perfect page coverage, and only depth suffers. A 20%
  component cannot pull a score across 0.3 -- which is why informativeness
  multiplies rather than joining as a fifth weighted factor. It is a
  precondition, not a contribution.
- **Raising the threshold cannot work, because the ordering was inverted.**
  `Page 1..20` scored **0.860**, above the bundled PSoC 6's real 89-entry outline
  at **0.820**; `Page 1..134` scored 0.680 and a single real section 0.620. No
  cutoff separates two populations when the bad one ranks above the good one.
- **Keyed on the breadcrumb, not the bare title.** Two chapters legitimately share
  a subsection name and the ancestry path separates them: bare title against
  breadcrumb, `esp32_trm` (784p) **0.727 / 0.995**, `raspi_rp2040` (642p)
  0.744 / 0.943, `micro_atmega328` (294p) 0.794 / 0.955. Up to 27 points on the
  large reference manuals, which are the repetition stress case. A flat
  `Page 1..N` outline has `breadcrumb == title`, so it still collapses to one key
  and is still caught, and `TocNode.breadcrumb` defaults to `""` on a directly
  constructed node, so the title stays the fallback. `assign_breadcrumbs` runs
  inside `build_tree`, through which the LLM candidate also returns, so baseline
  and candidate are normalised identically.
- **Fallback decisions that flip: 0.** The worst real document, `raspi_pico`,
  scores **0.697** against the 0.300 threshold -- 2.3x headroom -- while the best
  degenerate outline manages 0.043, a **16x** gap. That separation is what makes
  the rule safe, not the precision of the normaliser. The bundled PSoC 6 is
  unchanged at **0.820** with informativeness exactly **1.000**, pinned by a test
  so a future change to `normalize_key` that reintroduces collisions fails loudly.
- **Numbered siblings collide, and that is accepted rather than fixed.**
  `Port P# (P#.# to P#.#) Input/Output` x8 on `ti_msp430f5529` (Ports P1-P8),
  `GPIOR# - General Purpose I/O Register #` x3 on `micro_atmega328`, and
  `D.#. # June #` x5 on `raspi_pico` (release dates). `ti_msp430f5529` pays
  **0.820 -> 0.741**. Unfixable by inspecting titles alone -- `Page 1`/`Page 2`
  and `Port P1`/`Port P2` are structurally identical, differing only in whether
  the stem carries meaning -- and removing the masking is not an option, since it
  is the whole mechanism. The corpus bounds the damage: on real documents
  collisions are always a *minority* (worst measured **21%**, on `raspi_pico`),
  whereas a degenerate outline collapses essentially every entry.
- **Frequency, stated plainly: no degenerate outline appeared anywhere in the
  corpus.** The shape was observed on a real document earlier in development,
  but among mainstream vendor datasheets it is rare. This is cheap insurance
  against a demonstrably wrong gate, not a fix for a common case, and it should
  not be described as one.
- **A library-side LLM judge of ToC quality is out of scope.** The agent already
  holds the full ToC, so a second model call would pay to reach a conclusion the
  first model can already draw -- and it would put nondeterminism inside a cached
  artifact. The score stays a cheap deterministic floor, because `[llm]` is
  optional and a credential-free build must still emit a `toc_quality` block;
  judging belongs to the agent, which is what `regenerate_toc` is for.
- **A consequence taken knowingly:** `recommend_summaries` is
  `score < 0.5 or entry_count > 40`, so every newly low-scoring document now
  recommends summaries too. Defensible -- a useless ToC is where summaries help --
  but it is an LLM cost that follows automatically from the score change.

#### Why there is no non-LLM ToC fallback

A deterministic fallback -- reconstructing an outline from the body text with no
LLM call, for the credential-free default install -- was designed and rejected
on measurement over the same corpus. The trigger population is empty in a way
that no amount of implementation quality can fix.

- **"Weak ToC" is not a real category; "no bookmarks at all" is.** Documents with
  an outline scoring below the 0.3 threshold: **0 of 25**. Documents with no
  native outline: **1 of 25**. The fallback has exactly one trigger and, in this
  corpus, exactly one document.
- **That document does not want a ToC.** `vishay_1n4001` is 5 pages and 9,757
  characters, of which the automatic preamble already covers pages 1-2. Its whole
  text file is roughly 2,500 tokens -- cheaper to read end to end than any outline
  is to consult. A ToC is a routing device, and routing earns nothing when the
  destination is five pages away.
- **The most reliable technique can only fire where it is not needed.** Parsing
  the document's own *printed* ToC (dot-leader rows) is the sturdiest non-LLM
  route available. Measured: **19 of 25 documents carry printed-ToC rows, and all
  19 already have bookmarks**; the one document without bookmarks has zero. That
  is not luck -- bookmarks and the printed ToC are emitted by the same publishing
  pipeline (Word, FrameMaker, LaTeX), so their presence is correlated by
  construction. The leader-row detector is a lower bound and may have missed
  ToCs in unusual layouts, which only strengthens the correlation; the
  bookmark-less document's zero was confirmed by reading all five of its pages,
  not inferred from the regex.
- **Heading clustering by font size aims at the same 5-page datasheet.** At best
  it recovers `FEATURES / MECHANICAL DATA / MAXIMUM RATINGS / ELECTRICAL
  CHARACTERISTICS / ORDERING INFORMATION` -- five headings the agent already sees
  in the preamble it gets for free.
- **The cost argument that would have justified it does not fire either.** The
  LLM fallback genuinely scales badly: 15,000-character chunks with a 1s
  inter-chunk delay put `raspi_rp2040` (642p) at ~88 sequential calls and
  `esp32_trm` (784p) at ~85. A deterministic path would be free and instant --
  but only on a *large bookmark-less* document, and there is none: every document
  over 30 pages in the corpus has bookmarks. On the one document that does
  trigger, the LLM fallback is a single call.
- **The gap it was meant to fill is already closed.** `_NO_TOC_HINT`
  (`tools/bound.py`) fires whenever the returned ToC is empty and tells the agent
  how to navigate without a section map, and `toc_fallback_pending` (above) stops
  the credential-free build from rebuilding forever. What remained was a third
  structural source (`toc_source: "heuristic"`), its own accept/reject gate
  against the existing outline, and its own failure modes, for a population of
  one.

**What would reopen this, stated so it is falsifiable.** The corpus is public,
mainstream vendors; the deployment is internal documents -- internal specs,
application notes, customer documents and scanned legacy parts. If a meaningful
share of those are long *and* bookmark-less, the arithmetic changes. That is
cheap to find out: `extract_toc()` plus a page count over a directory of internal
PDFs, no build required. Until that number exists, the fallback is speculation.

One caveat if scanned documents turn out to be that population: a scan has no
text layer, so **neither** fallback works -- the LLM one reads body text too. The
answer there is OCR or an explicit no-text-layer signal in the manifest, which is
a different gap from this one and is not currently emitted.

### Figure indexing

A second top-level key, `figures`, sits alongside `toc` -- page-keyed rather
than node-attached, since the geometry is a page property computed in the same
pass that produces the text file (`core/figures.py`, folded into
`core/textfile.py:scan_pages`). It carries two entry kinds, and they are never
merged into one, even when a raster region and a caption share a page:
associating them by proximity is a heuristic that can name the wrong figure,
and two honest, separate entries cannot mislead the way a wrong association
would.

- **`"raster"`** -- one image XObject placement per raster region at or above
  `min_area_pct` (a module constant, default 1.0%; smaller placements are
  dropped as decorative and counted in the sibling `figures_excluded` key
  rather than silently discarded). `region` is the primary field: normalized
  `0.0-1.0` and clipped to the visible page, in exactly the
  `{"top", "bottom", "left", "right"}` shape `inspect_page(region=...)`
  consumes. That shared contract is deliberate -- a `figures` entry's `region`
  can be handed to `inspect_page` unmodified, by the agent or by the VLM
  captioning pass below, with no coordinate math and no risk of a silent wrong
  division. `bbox` (raw PDF points) and `pixels` are carried alongside for
  consumers that need absolute geometry. `page_text_chars` is denormalized
  onto the entry as the "is the agent blind here" signal: a large
  `page_area_pct` beside a small `page_text_chars` marks a page whose
  substance a picture is withholding from the text layer. `xref` names the
  image XObject the placement draws, which is what makes two entries
  recognizable as the same picture -- see the captioning dedup below.
- **`"caption"`** -- a `Figure N` / `Fig. N` mention recognized in the
  column-aware page text, in either of two forms (same-line with a mandatory
  `.`/`:` separator, or split across two lines), including section-relative
  numbering (`"10-1"`). `figure_number` is always a string, never coerced to
  an int -- it is an identifier to display and match on, not an arithmetic
  value, and a union type would cost every consumer a branch for no benefit.

Every raster region above the threshold is also a candidate for VLM
captioning (`caption_source: "llm"`), which fills in a short description for
regions the text layer never named -- see `llm/figure_captions.py` and the
README for the cost, the cap, and the default-on behaviour. The caption names
the kind of content and then, immediately, its most identifying labels: for a
table, its row labels first and then its column headings; for a plot, its
axes and plotted quantity. Row-labels-first is deliberate, not stylistic --
see the measurement below.

**The unit of captioning is a picture, not a placement.** Placements sharing
an `xref` are grouped (`llm/figure_captions.py:_image_groups`); the largest is
rendered and described, every placement in the group receives that caption,
and `max_figure_captions` bounds groups. This is exact, not a heuristic: a PDF
XObject's content cannot vary between placements, only its scale, so one
description cannot be wrong for another placement of it. Without the grouping
a vendor logo in a page header is a fresh figure on every page -- on onsemi's
four-page product-change notices it was the *only* region above the area
threshold, so the document spent its entire caption budget describing one logo
four times, in four slightly different wordings. An `xref` of 0 or missing is
**not** an identity and never groups; treating unknown as equal would give one
picture's caption to every unidentified figure in a document, which is the
failure mode that produces confident nonsense rather than a visible error.

**The agent is handed a digest, not the array.** `build_datasheet`'s manifest
(`tools/bound.py:get_artifact_manifest`) carries a bounded `figures` block --
`total` / `raster` / `captioned` counts, plus one `{page, figures, caption}`
row per page holding figures in ascending page order, that row's caption
being the page's **largest-area** captioned entry (by `page_area_pct`), not
merely the first one in document order -- see the digest-selection fix below.
The array itself stays in the ToC JSON: the manifest is returned on every
build, and a scanned document can hold one full-page raster per page, so the
digest is capped at 40 rows with one 350-character caption each
(`pages_with_figures` and `truncated` disclose what was dropped). Carrying
*something* is not optional -- the MCP agent receives only the manifest, and
per the WSL namespace gotcha `json_path` may not even be readable from where
the agent runs, so a digest is the difference between the agent knowing a
page holds a figure and never learning the figure index exists.

#### Decisions already settled by measurement

Every number below comes from a **14-document, 998-page, 5-vendor corpus** (TI,
Infineon, Microchip, Nexperia, Diodes) measured 2026-07-25, plus a live run
against an OpenAI-compatible gateway. They are recorded because each is a
decision a future reader would otherwise redo.

- **`pymupdf.layout` was evaluated for figure discovery and rejected on cost.**
  Warm, after model load: 1.28 s/page over 5 pages, 0.89 s/page over 20 --
  extrapolating to **~119 s for the 134-page PSoC 6 against a ~8 s build**,
  roughly 15x, in the default path, for every document. It also requires the
  ~49 MB `[layout]` extra a plain `uv sync` excludes, brings the process-global
  hook hazard documented in `core/engine.py`, and unlike `get_image_info()` it is
  a model with an error rate rather than an exact enumeration. `get_image_info()`
  is exact and free.
- **A per-page `describe_figure(page)` tool was proposed and rejected, and the
  analogy that motivates it is seductive.** `extract_table_markdown` earns its
  layout-engine cost by giving the agent something it *cannot* produce itself: an
  exact table from the text layer, no vision error, few tokens. Layout
  classification gives it something *weaker* than looking -- an agent already
  holding a page has `inspect_page`, and its own vision beats a DocLayNet label.
  It also serves the wrong axis: discovery is a breadth question ("which of 134
  pages hides something?") that a per-page call cannot answer without sweeping
  every page, which is the 119 s above.
- **The caption pattern's mandatory separator is load-bearing.** Requiring
  punctuation after the figure number is what divides **404 real captions from 70
  prose lines** across 998 pages, with no scoring and no heuristics. Without it,
  "Figure 6-2 shows the structure..." parses as a caption. Do not relax it.
- **Section-relative numbering is the common case, not an edge case.**
  `Figure 12` alone matched captions in only **2 of 14** documents; widening to
  `(\d+(?:[-–]\d+)?)` to admit `Figure 10-1` reached **11 of 14**. The same-line
  section-relative form is the corpus's most frequent, 404 of 492 caption lines,
  present in 9 of 14 documents. This is why `figure_number` is always a string.
- **`min_area_pct = 1.0` is not defensive dead code.** It excludes **73 of 168
  placements (43%)** across the corpus and changes the output in 4 of 14
  documents -- mostly vendor logos repeated on every page.
- **Region clipping is exercised by real data.** **9 placements** in the corpus
  extend past the page edge. `inspect_page` *raises* on a coordinate outside
  `0.0-1.0`, so clipping to `page.rect` before normalizing is what keeps the
  coordinate contract that makes `figures` directly usable as `inspect_page`
  input.
- **No exact caption source exists in practice.** Of the 14 documents, **1 is a
  tagged PDF** -- and it carries zero `/Figure` structure elements -- and **none**
  carries a List of Figures. Reading structure instead of text is not an
  available shortcut.
- **Serial render, concurrent dispatch, 4 workers.** PyMuPDF is not thread-safe
  for concurrent page work, so rendering stays serial; only the network calls are
  parallel. Live on the PSoC 6 at the default cap of 20: serial dispatch ~119 s
  (~6 s/call), 4-worker concurrent ~13 s. Results are applied in **candidate
  order, never completion order** -- artifact bytes are fingerprinted for reuse,
  so completion-order output would be non-deterministic and would silently defeat
  the cache.
- **Regions render at `detail="high"` and are now sent at `detail: "high"` too,
  reversing the earlier `detail: "low"` choice on measured evidence of
  fabrication rather than on a hunch.** `"low"`
  downscales to 512x512 before the model ever sees the image, which reads as the
  safer choice for the no-transcription rule -- the model cannot fabricate rows
  from detail it never received. Measured on the motivating PCN's page-5
  "Product Attributes" table (20 rows, 9 columns) with an explicit "list the row
  headings verbatim, or say you cannot read them" probe, it did exactly the
  opposite: `"low"` invented `Voltage`, `Wafer Base Supplier`, `Wafer Fab
  Location`, `Package Fab (OSAT)`, `Package Type`, `Mold Compound Lot Number`,
  and `Mold Compound Location` -- none of which are real rows -- and missed real
  ones, including both supplier rows. At `"high"` the same probe returned 19 of
  20 row headings verbatim correct (the one error: `Die Composition` for `Bond
  Wire Composition`), including both supplier rows and the grey section rows
  `Die Attributes` and `Package Attributes`. A prompt asking for row labels at a
  resolution where they are illegible does not get a safe "I cannot read this"
  -- it gets confident fabrication, so the resolution is part of the
  anti-fabrication design, not independent of it.
- **The per-image token cost is now measured, not documented from a spec
  sheet.** A separate token-*counting* endpoint on the validation gateway
  returns `405 Method Not Allowed`, which is why an earlier note here called the
  cost unconfirmed -- but `usage` on a real response works, and gave real
  numbers: **120 input tokens per image at `detail: "low"`, 1074 at
  `detail: "high"`** -- about 9x, or roughly 2.4k to 21.5k input tokens per
  document at the default cap of 20. Paid once per document, then cached on disk
  by the existing artifact reuse.
- **Row labels before column headings is load-bearing, not a style choice.**
  With column headings named first in the prompt, the table's identifying words
  (the supplier names) landed at character 442 of the reply on the PCN's page-5
  table -- past the digest's caption clip. Moving row labels first moved the
  same hook (`Mount Compound Supplier`) to character 310, inside the clip. The
  digest clip and the prompt's label ordering are therefore one design, not two
  independent choices.
- **A single illustration can arrive as many overlapping raster XObjects, and
  this is now measured, not merely suspected.** `ti-tlv9061.pdf` page 46's
  mechanical package drawing is exported as **17** overlapping raster
  placements, several of them empty fragments -- confirmed by a live
  corpus run where one such fragment was blank white space captioned as "a
  schematic diagram ... optocoupler component" (fixed by the blank-region
  guard in `llm/figure_captions.py`; see the CHANGELOG). Clustering
  fragments into one figure entry remains out of scope -- the same
  ambiguity that keeps vector-drawing clustering unreliable applies here --
  but the consequence is worth disclosing rather than leaving for the next
  reader to rediscover: a fragmented figure inflates `figures` counts on
  that page well beyond the visual figure count, and each fragment is a
  separate candidate that can consume a `max_figure_captions` slot the cap
  intended for a distinct figure elsewhere on the document.
- **The digest's per-page caption picked the first captioned entry in array
  order, which is the topmost figure on the page -- a bug, found on the same
  PCN.** Page 5 carries a 7.5%-of-page product-label photo above a
  25.5%-of-page "Product Attributes" table, in that document order. Under
  first-in-order selection the digest told the agent page 5 was "a photo of a
  product label" and the table -- the one holding the answer to "does this
  document mention SUMITOMO" -- was silently dropped, even though `SUMITOMO`
  appears 13 times in that table's `Mount Compound Supplier` and `Mold Compound
  Supplier` rows and a text search for it correctly returns zero hits (the word
  is pixels). The fix selects each row's caption from the page's **largest-area**
  captioned entry instead (`page_area_pct`), which is already the signal that
  ranks caption candidates for the `max_figure_captions` cap, so this is a
  consistent selection rule rather than a new heuristic. Ties keep whichever
  entry the scan reaches first in the array's own document order -- never a
  dict or set's -- so the digest stays byte-stable across runs. On the PCN this
  changes exactly one row (page 5).
- **Every LLM call goes over Chat Completions, not the Responses API, and the
  reason is a measured 50% silent loss.** Captioning moved first, in 0.28.0, and
  the text and structured calls followed in 0.30.0 once the same failure was
  measured on them directly (see "One transport for every call shape" below). On
  an OpenAI-compatible
  gateway the Responses API is a *bridge* for any model the gateway does not
  serve natively, and that bridge can file the model's answer as a `reasoning`
  item carrying `reasoning_text`. `output_text` concatenates only `output_text`
  chunks, so the caption arrives as the empty string with its text sitting in
  the same payload. Measured with the shipping client against a self-hosted
  `qwen3.6-27b` (vLLM) over 16 real figure regions: **8 to 12 of 16 captions
  empty** over five runs, and a **different subset each run** -- it is per-call
  sampling, not a property of any figure. Chat Completions does not work around
  the bridge, it bypasses it: same model, same images, same prompt, **0 empty in
  144 calls**, and the raw message carries no reasoning channel at all. The
  severity is higher than "some captions missing": `caption_figures_in_place`
  scores a blank reply as `failed`, which marks the artifact incomplete, so a
  coin-flip transport re-captions the document on every future build forever --
  the cache-poisoning failure the `blank`/`failed` split exists to prevent.
  **One path for every model, not a branch on model name.** gpt-4.1 was
  re-measured over the same 16 regions on Chat Completions and is
  indistinguishable from the Responses path (1084 median input tokens either
  way, which doubles as the check that the nested `image_url` object's
  `detail: "high"` is honoured -- `"low"` would show as roughly a tenth of
  that). Note the two forms are not interchangeable: `image_url` is a plain
  string on Responses and an object here, and the wrong one type-checks and
  fails at the gateway.
- **One transport for every call shape, and the text path was measured before it
  moved.** 0.28.0 moved captioning alone and deferred the text calls, warning
  against pointing a non-native model at the ToC fallback until the same work
  was done. That work is 0.30.0, and the deferral was justified: running the
  real ToC fallback over the Responses API against `qwen3.6-27b` on the prod
  gateway, across the repo's own PSoC 6 (15 chunks) and TI PCN (1 chunk)
  fixtures, **7-8 of 15 chunks came back with an empty `output_text` on the
  PSoC and 1 of 1 on the PCN** -- on the structured *and* the free-text path.
  gpt-4.1 over the same 90 calls: **0 empty**. So the bridge failure is a
  property of the transport, not of image input. The same matrix on Chat
  Completions: **0 empty in 192 calls**, and on the PCN `qwen3.6-27b` goes from
  zero ToC entries to a working ToC. Caught in the act on a raw request, same
  model and prompt on two runs: `output` item types `["message"]` with 3727
  characters of `output_text`, then `["reasoning"]` with **0** -- both reporting
  `status: "completed"`, which is why nothing downstream could tell.
- **On the PCN that meant no table of contents at all, silently, at double the
  cost.** The paths compound: the structured extractor parses `""`, raises, and
  aborts the run; `generate_toc_from_text` sees zero entries and retries the
  whole document with the free-text prompt **over the same transport**; that
  returns `""` too; the assembled candidate is empty and
  `_accept_llm_toc_candidate` drops it. Two full passes over the document are
  paid for, no ToC is produced, and the only log line says "retrying". That
  silence is why `_read_chat_reply` now logs the model and `finish_reason` on
  any blank reply -- the same guard captioning got, for the same reason.
- **Structured output survives the move, which was the one real risk.**
  `text.format=json_schema` becomes `response_format={"type": "json_schema",
  "json_schema": {...}}`, one level deeper, and strict-schema support across
  gateway backends is not uniform. Verified on both models before shipping.
  `finish_reason` replaces the Responses `status`: `"stop"` is completion,
  anything else is reported as incomplete so a truncated chunk still raises
  instead of parsing as half a document. A **missing** `finish_reason` stays
  `None` rather than becoming "incomplete" -- `_parse_structured_chunk_response`
  already reads `None` as "this gateway does not report one", and mapping the
  absence to a failure would discard every good chunk such a gateway returns.
- **Any benchmark of this must vary the prompt per call.** The gateway caches
  identical Chat Completions payloads -- verified by repeats returning the same
  completion `id` and `created` plus an `x-litellm-cache-key` response header.
  An unvaried repeat measures a replay, which cannot reveal a per-call coin
  flip and reports a latency that is not the model's. The live regression test
  in `tests/test_figure_captions_live.py` appends an attempt number for exactly
  this reason. This applies to the text path too, and the first pass at the
  0.30.0 measurement forgot it: its repeats came back in 0.2s serving identical
  counts, so only the first run of each cell -- plus the distinct per-chunk
  payloads within it -- was real evidence. The re-run varied the prompt.
- **The caption call caps output at 300 tokens, because prompt compliance is
  not uniform across models.** Over the same 16 regions the median reply is
  71-102 tokens and gpt-4.1's worst case is 197, comfortably inside the
  prompt's "under 60 words". qwen answered a 128-pin TQFP pinout by enumerating
  all 128 pins -- **667 tokens** against gpt-4.1's 134 on the same image.
  Truncation is not a new failure mode: the caption prompt already orders its
  output for it ("your text may be truncated, so identifying labels must come
  before any description of structure"), so a clipped caption keeps the part
  that earns its place in the index. The cap sits above every compliant answer
  measured, so it binds on runaways only.
- **The vision model is a setting, not a default.** `DATASHEETINDEX_VISION_MODEL`
  overrides the model for figure captioning alone; unset, captioning follows the
  same model as summaries and the ToC fallback. It exists because captioning is
  the only per-figure cost in a build and the cheapest capable vision model is a
  property of the *gateway*, not of this library -- the self-hosted model that
  motivated the knob meters at $0 input / $0.13 per MTok output against gpt-4.1's
  $2 / $8, roughly 250x cheaper per document at the default cap of 20, at
  caption quality a side-by-side over 16 real regions found competitive. Naming
  such a model in the source would be wrong twice over: it is absent from its own
  gateway's staging tier, and it means nothing to anyone pointing
  `LITELLM_BASE_URL` somewhere else.
- **So is the text model, and for one call more than symmetry.**
  `DATASHEETINDEX_MODEL` names the model for summaries and the ToC fallback;
  unset, it is `gpt-4.1`. Before it existed the only way to name a text model
  was `build_datasheet`'s `model` argument -- which the agent is told to omit
  unless the user asked for a specific one -- so the *automatic* ToC fallback,
  the path that runs without anyone deciding anything, was pinned to a name
  this library cannot know a given gateway serves. A deployment whose gateway
  did not serve it had no way to make the fallback work at all.

  The two knobs sit at different levels on purpose, and the rule is where the
  deciding information lives. Which models a gateway serves is deployment
  knowledge, so both are env vars. Whether *this document* is worth a better
  model is per-call knowledge, so the text model is also a tool argument, and
  it wins over the env var. Nothing about a single document tells an agent
  which vision model a gateway has, so there is no vision tool argument -- and
  a wrong name there is the expensive failure: every caption call fails, the
  artifact is marked incomplete, and the document is rebuilt on every request
  until it is corrected.

### Deliverable 2: Page-Matched Text File

The full PDF converted to text with clear page markers, using **PyMuPDF `get_text("blocks")`** with column-aware reading order:

**Why block-based extraction with column detection:**
- **Fast** — 1.2s for 73 pages, no overhead on single-column pages
- **Column-aware** — two-column datasheet layouts (common in TI, NXP, STMicro) are read left column first, then right column, instead of interleaving lines from both columns
- For complex semiconductor tables with merged headers and multi-line cells, raw text flows naturally and modern LLMs handle it well
- The text file is a navigation aid, not the final extraction method — the agent has `inspect_page` for when text isn't enough
- Column detection uses block geometry (width, height, gutter gap) with conservative thresholds to avoid false positives on table cells and diagram labels

```
--- PAGE 1 ---
[Cover page content...]

--- PAGE 2 ---
[Revision history...]

--- PAGE 22 ---
5 Electrical Characteristics
5.1 Absolute Maximum Ratings

Table 5-1: Absolute Maximum Ratings

Parameter
Symbol
Min
Max
Unit
Junction temperature
TJ
-40
150
°C
Supply voltage VS
VS
-0.3
65
V
...

Note(1): Stresses above those listed under Absolute Maximum Ratings
may cause permanent damage to the device.

--- PAGE 23 ---
5.2 Operating Conditions
...
```

**Critical requirement:** The `--- PAGE N ---` markers must correspond exactly to the page numbers in the JSON. When the agent reads in the JSON that "Absolute Maximum Ratings" is on pages 22-23, it can search for or read pages 22-23 in the text file and find the right content.

**Generated by:** PyMuPDF `get_text("blocks")` per page with column-aware reordering, reassembled with page markers.

### Running header/footer stripping

Every page of a datasheet repeats a header naming the part and a footer with
the document title, a revision string and a page number, and all of it used to
reach `search_text`, `get_section_text` and the LLM ToC fallback. The cost was
never really tokens; it was search precision. On the bundled 134-page PSoC 6,
`search_text("PSOC")` returned 209 matches -- over the 200 cap an agent sees, so
genuine hits were evicted -- of which 133 were the running header.

`core/furniture.py` holds the decision logic as pure functions over strings and
counts (no PyMuPDF, no environment, no I/O); `core/textfile.py` holds all the
geometry and the two-pass assembly in `scan_pages`. A block is dropped iff:

- it lies **wholly inside** the top or bottom 20% of its own page, so landscape
  and mixed-size pages need no special case;
- it is at most **200 characters** of raw text and does not open with a caption
  keyword (`figure`, `fig.`, `table`, `chart`);
- its key -- whitespace collapsed, digit runs masked to `#` -- **contains at
  least one letter**; and
- that key clears **either** recurrence route:
  - **overall**: it appears on at least `max(3, ceil(0.5 * pages))` pages; or
  - **one page parity**: within the odd- or even-page bucket alone it clears
    `max(6, ceil(0.5 * pages_in_bucket))` -- a higher floor than the overall
    route's, for the reason below -- **and** appears on at most `0.2 x that
    count` pages of the other parity. This is the alternating odd/even header,
    which no overall count can reach.

`DATASHEETINDEX_FURNITURE=0` (also `false`/`no`/`off`) disables it, and the
setting participates in the artifact-reuse key, so flipping it rebuilds rather
than serving a stale artifact. **The preamble is deliberately exempt**: it reads
`_extract_page_text`, which is unstripped, so its "raw text, zero heuristics"
contract holds by construction rather than by a flag. Do not move stripping into
`_extract_page_text`.

Measured on the PSoC: 200,584 -> 193,020 characters, 265 blocks, 3.8%. Search:
`Datasheet` 138 -> 6, `Rev. *S` 133 -> 1, `PSOC` 209 -> 76 (no longer capped).

#### Decisions already settled by measurement

The method is a simplified **Lin page-association** (*Header and footer
extraction by page-association*, SPIE 2003) -- the standard approach, not a new
one. What follows is what a survey of seven documents settled, so that the
constants above read as decisions rather than taste.

- **Block granularity, not lines.** A line-level scan finds `Table #` recurring
  89 times on the PSoC -- genuine captions that digit-masked matching would have
  deleted. At block granularity a `Table 43` caption is a body block and never
  enters the band. The caption-keyword rule is *insurance*, not the active
  mechanism: block granularity already keeps those captions out of the band.
- **No line-count rule.** An earlier draft excluded blocks of 3+ lines, copying
  PageIndex. That is wrong for PyMuPDF, whose `get_text("blocks")` groups a whole
  footer into one block of several short lines -- the PSoC footer is a single
  4-line, 41-character block, so the rule discarded the footer on 132 of 134
  pages. Across seven documents it missed genuine footers on five. PageIndex's
  equivalent guard is characters-*per*-line, because its blocks are paragraph
  clusters; that variant was measured too and also over-excludes.
- **A furniture key must contain a letter.** Digit masking can otherwise produce
  a key with no lexical evidence at all (`#`, `#.# #.#`). A bare page-number
  footer then makes `#` furniture, after which every bare-number block in either
  band is deleted document-wide -- reproduced, deleting table values `120`,
  `127`, `3.3 4.3`. The guard costs **zero** furniture keys on the bundled PDFs
  and across the survey. Its accepted price: a footer that is *only* a page
  number is no longer stripped, which converts a content-deletion mode into a
  miss.
- **The 0.5 page fraction.** Real furniture recurs on 52-100% of pages; the only
  keys below 92% are one document's two. Lowering to 0.33 starts deleting running
  *section headings* -- `6 Electrical specifications` on 47 of the PSoC's 134
  pages, `Register description` on 12 of 41 in another. 0.5 is the last value at
  which the corpus stays clean.
- **Parity dominance, not a bare parity threshold.** Live testing across a
  17-document, 8-vendor corpus found the odd/even case is not a corner: on
  `micro_atmega328.pdf` (294 pages) four genuine furniture keys sit at **exactly
  146 pages** each -- `ATmega328P [DATASHEET]` on 146 even pages and zero odd,
  its twin `3 ATmega328P [DATASHEET]` the mirror -- one page under the 147-page
  threshold, and the document stripped nothing at all. A *bare* per-parity
  threshold is the wrong fix: a bucket is half the document, so it would admit
  any key on roughly **25%** of the pages, which is the loosening the 0.5
  fraction was measured to reject one paragraph above. Requiring near-absence
  from the other parity (`PARITY_DOMINANCE = 0.2`) restricts the new route to
  the actual signature of an alternating header. The constant is not tuned:
  every key it recovers on this corpus is a clean split -- 146/0, 0/146, 14/0,
  0/13 -- so dominance rejects none of them. It has exactly one measured cost:
  `www.ti.com` on `ti_lm358.pdf` is 22 pages of one parity and 8 of the other,
  genuine furniture that dominance deliberately declines, because 8-against-22
  is an uneven recurrence rather than an alternating layout, and admitting it
  means admitting every similarly uneven key. Recovery measured, in dropped
  **blocks** (`_is_furniture_block`, the production predicate -- not lines; one
  block is typically several, and the PSoC's 265 blocks are 926 lines):
  `micro_atmega328.pdf` 0 -> 584 blocks (13,034 chars), `ti_ina219.pdf`
  55 -> 109, `tcan1044a-q1.pdf` 48 -> 101, and the other 14 documents --
  including the bundled PSoC 6 at its unchanged 265 -- byte-identical.
- **The parity route floors at `PARITY_MIN_PAGES = 6`, double the overall
  route's.** `MIN_PAGES` is absolute and does not scale, so sharing it let the
  parity route fire on a 10-page document from three pages of one parity -- 30%
  of the document, where the overall route demands 50%, and 23% at 13 pages.
  Dominance is weakest exactly there and cannot compensate: `0.2 * 3` is 0.6, so
  the other-parity count is forced to zero, which any three-page odd-only run
  satisfies. Odd-page section starts are a standard print convention, and the
  string that demonstrated it was `Register description` -- the content the 0.5
  fraction exists to protect. Doubling the floor cannot bind where the route
  earns its keep (146-against-74 and 14-against-10 bucket margins) and leaves
  every measured recovery above unchanged.
- **No fuzzy matching.** A similarity threshold can delete a genuine one-off line
  resembling its neighbours. The accepted cost is that furniture whose *letters*
  vary per page is missed, which fails safe.

#### Known limits, and why they are accepted

| Not detected | Why |
|---|---|
| Furniture whose letters vary per page (per-chapter running titles) | Exact-plus-digit-masked matching by design; fails safe by keeping text |
| Furniture recurring **unevenly** across the two parities | Largely fixed: a header alternating cleanly by odd/even page is now caught by the parity route. What remains is furniture whose recurrence is uneven rather than alternating -- `www.ti.com` at 22-versus-8 on `ti_lm358.pdf` -- which the dominance rule declines on purpose (above), since the alternative admits any key on ~25% of a document |
| Furniture confined to one **part** of a document | `ti_lm358.pdf` is a 32-page datasheet bound ahead of 36 pages of package-option and mechanical appendices with their own furniture. Each header variant covers 16 of its parity's 34 pages, so neither the overall nor the parity threshold is met, and the document still strips nothing. Not a parity problem; reaching it needs a lower fraction, which is worse (above) |
| Page-number-only footers | A letterless key would delete numeric content (above) |
| A repeated **table header row** on a table-heavy document | It is in-band, caption-free and recurrent. On the PSoC the repeated `Spec ID Parameter Description Min Typ Max Unit` row reaches 38/134 -- under threshold there, but not necessarily elsewhere. No cheap guard exists; `DATASHEETINDEX_FURNITURE=0` is the escape |
| A product title on a cover page | Dropped when the same string also runs as a header on most other pages (observed on the PSoC). Still reachable via the unstripped preamble and `search_text` |

#### Alternatives measured and rejected

- **`pymupdf.layout`**, which classifies `page-header`/`page-footer` directly and
  is what 0.32.0 uses to clean `extract_table_markdown`, cannot serve the text
  file: ~0.95s/page (~128s for the PSoC against an ~8s build) and it sits behind
  the optional `[layout]` extra that a plain `uv sync` excludes. It earns its
  keep instead as the **oracle**: a `layout`-marked test asserts that every block
  we delete is one the model also calls furniture, at >= 0.95 precision.
- **Existing libraries.** `refinedoc` (Apache-2.0, pure stdlib, page-association
  by name) was the only zero-dependency candidate and was tested rather than read
  about: working on text lines with no coordinates it cannot apply a position
  band, and on the PSoC it classifies four `Table N (continued)` captions as
  headers -- exactly the captions `TocNode.continued_tables` is built from. It is
  also slow (7.15s on the PSoC) and prints warnings to stdout, which under MCP
  stdio is the JSON-RPC channel. `docling` and `unstructured` are the same ML
  class as `pymupdf.layout`, already rejected above on cost.
- **LLM-driven detection.** Genuinely cheap -- the task needs only the distinct
  candidate strings and their page counts, so a prompt is 345-5,678 tokens
  regardless of document length. Rejected because *precision* is what matters
  when the operation is deletion: on one document a self-hosted `qwen3.6-27b`
  flagged 73 of 198 candidates including a table header row and several section
  headings, and `mixtral` flagged a document's own title. The deterministic
  rule's failures are misses; the LLM's are false deletions. It would also put an
  LLM in the path of a deliverable that must build with no credentials.

---

## Agent Tools

Beyond the built-in file reading capabilities of the Claude Agent SDK, the agent has a custom PDF-native inspection tool: `inspect_page` (visual inspection). Text-to-coordinate grounding (`locate_text`) is a Python API rather than an agent tool -- see below for why.

### `inspect_page`

Renders a PDF page as an image for **visual inspection** by the multimodal LLM. The agent sees the page exactly as a human would — with aligned columns, clear table structure, legible formulas, and visible figures.

```python
def inspect_page(
    page: int,
    region: dict | None = None,
    dpi: int | None = None,
    detail: Literal["low", "medium", "high"] = "high",
) -> list[dict]:
    """Visually inspect a PDF page by rendering it as an image.

    Args:
        page: 1-indexed page number (matching JSON and text file markers).
        region: Optional crop region as percentage of page dimensions:
                {"top": 0.0, "bottom": 0.5, "left": 0.0, "right": 1.0}
                Values are 0.0-1.0 fractions. This example crops to the
                top half of the page.
                If omitted, renders the full page.
        dpi: Explicit render resolution. Power-user override that wins
                over ``detail`` when set. Default ``None``.
        detail: Vision-token-cost tier -- "low" (75 dpi, ~650 tokens),
                "medium" (100 dpi, ~1150 tokens), or "high" (150 dpi,
                ~2580 tokens) per US-letter page on the Anthropic
                ``(W*H)/750`` formula. The library primitive defaults to
                "high" for backward compatibility; agent-surface
                wrappers (``DatasheetTools.inspect_page``, MCP tools)
                default to "medium" to halve cost on long loops.

    Returns:
        Library-internal content block list:
        [{"type": "image", "data": "<base64 PNG>", "mime_type": "image/png"}]

        The neutral tool envelope is NOT this block verbatim -- the
        inspect_page handler in tools/defs.py re-emits the media type
        under both "mime_type" and "mimeType". See "The image block
        carries two media-type keys" below.
    """
```

**Region uses percentage-based coordinates (0.0-1.0)** rather than PDF points. This is intentional: the agent doesn't know page dimensions in points, but it can reason about "top half", "bottom third", or "left two-thirds" naturally. The library converts percentages to PDF point coordinates internally.

Common region patterns:
- Top half: `{"top": 0.0, "bottom": 0.5, "left": 0.0, "right": 1.0}`
- Bottom half: `{"top": 0.5, "bottom": 1.0, "left": 0.0, "right": 1.0}`
- Full page: omit region (default)

**Return format** follows the Claude Agent SDK tool response convention: a list of content blocks. The image is base64-encoded PNG. The calling code in `tools/registry.py` handles this wrapping so the agent receives the image directly.

**Why only one *vision/table* tool?** We evaluated and dropped three other tools during design:
- `get_table` — PyMuPDF `find_tables()` produces worse results than raw text on complex semiconductor tables. Visual inspection is more reliable.
- `get_figure` — `get_images()` can't detect vector graphics (which is how 95% of datasheet diagrams are rendered). Visual inspection shows figures in full page context.
- `get_page_tables_overview` — Returns row/column counts from an unreliable detector. The text file already gives richer information: table titles, column headers, "(continued)" markers, and actual values.

### `locate_text` — a Python API, not an agent tool

Maps a query string to its bounding box(es) on a page. It returns one
result per match, each carrying `region` (a bounding rectangle) and `boxes`
(one or more per-line rectangles, with `region` their union), expressed in
**both** normalized percentages (0.0-1.0, clamped to the page so they feed
straight into `inspect_page(region=...)`) and raw, unclamped PDF points (for
annotating the PDF directly), plus page dimensions.

**It is deliberately not exposed as an agent tool.** It was, until measurement
showed the workflow does not pay off: a hit covers 0.07-0.58% of a page (a
heading match is a 1.6%-tall sliver), so cropping `inspect_page` to it renders
a picture of the query string and nothing else. To see the table *under* a
heading the agent would have to expand the region by an amount nothing tells it.
`inspect_page(page, detail="low")` is a cheaper overview from which the agent can
crop to what it actually observed, and no consumer was ever found calling the
tool — every real use is Python code calling the method directly.

Its real consumer is **source grounding**: a deterministic post-pass that
re-finds an extracted value's `source_text` in the live PDF and attaches page +
bounding box for citation and for highlight overlays in a UI. That is library
code, not an agent decision, which is why the method stays and the tool does not.

Matching is hybrid: `page.search_for` on the verbatim query (fast path), with a
normalized word-level fallback (`page.get_text("words")`) that tolerates the
dash/case/whitespace variation endemic to datasheets (`-0.3` vs `−0.3`, `±2%`).
The fast path returns one result per rectangle it finds (a single-line hit is
one single-box result); the normalized fallback groups a multi-line match's
words by `(block_no, line_no)` into a single result whose `region` is their
union.

It is stateless: the direct `DatasheetTools(pdf).locate_text(...)` Python API
works off the live PDF with no `build_datasheet` call. A document must first be loaded via `DatasheetTools(pdf)` or `build_datasheet`
(which binds the PDF), after which `locate_text` reads it directly without
needing the built text/JSON artifacts.

Grounding is string-level, not hit-level: a string appearing multiple times on a
page returns multiple results; disambiguate with a more specific query.

---

## What the Library Does NOT Do

- **Does not decide which parameters to extract** — the agent does
- **Does not decide when to escalate to vision** — the agent does
- **Does not implement extraction strategies** — the agent reasons about this
- **Does not validate parameter values** — the agent self-checks during extraction (min ≤ typ ≤ max, units present, footnotes captured). A library-level `validator.py` module is deferred until real extraction data reveals which error patterns are most common. Domain-specific plausibility checks (e.g., "is this voltage range reasonable for this IC type?") require per-device-class knowledge and don't scale.
- **Does not manage conversations or prompts** — that's the agent SDK's job. The system prompt guidance in this document is reference material for the consuming agent, not something the library generates or owns.

The library is a **pre-processor and toolbox**, not an extraction engine.

---

## Common Challenges (and How the Agent Handles Them)

The following challenges exist in datasheet parameter extraction. With the enriched JSON + text file + tools, here's how the agent can address each:

### 1. "Where is the parameter?" (Navigation)

The agent reads the JSON, sees that "Electrical Characteristics" is on pages 22-45 with 12 tables, and "Absolute Maximum Ratings" is a subsection on pages 22-23. It navigates directly to the right section in the text file. No embedding search needed.

### 2. "The table is garbled in text" (Table Quality)

The agent reads page 24 from the text file and the values aren't clear — columns run together or it's unsure which value belongs to which column. It calls `inspect_page(page=24)` and visually reads the table as a human would. This is more reliable than programmatic table parsing, which often garbles complex semiconductor tables even further.

### 3. "The value depends on conditions" (Context)

The agent sees "(1)" next to a value in the text file. It searches the text file for the footnote "(1)" on the same or nearby pages. If conditions are in a sub-header row, the agent reads the surrounding text context. The agent's LLM reasoning handles this naturally — it understands datasheet conventions.

### 4. "Same parameter, different sections" (Disambiguation)

The agent finds "VCC max = 65V" in "Absolute Maximum Ratings" and "VCC max = 60V" in "Recommended Operating Conditions". Because the JSON tells it which section each value came from, it correctly reports both with their contexts — one is a damage threshold, the other is the safe operating limit.

### 5. "Multi-page table" (Continuity)

The agent reads pages 24-26 from the text file and sees a table that continues. Because the text file preserves page markers but keeps the content flowing, the agent reads across page boundaries. If the table headers are lost on page 25, the agent can refer back to page 24's text or call `inspect_page(page=25)` for visual confirmation.

### 6. "Parameter is in a graph, not a table" (Visual Data)

The agent reads the text and finds "see Figure 3-2 for efficiency vs. load current". It calls `inspect_page(page=35)` and uses vision to read the graph in context. It reports the value as "approximately 92% at 500mA load (from Figure 3-2)" with a note that the value is read from a graph.

### 7. "Poor quality PDF / scanned document" (Degraded Input)

The text file for certain pages comes back nearly empty or garbled. The agent notices this (very little text for a page that should have content per the JSON). It calls `inspect_page` for those pages and uses vision for the entire extraction on those pages.

### 8. "This datasheet covers a product family" (Multi-Product Extraction)

A single PDF covers multiple product variants (e.g., TPS651/652/653, AD7606/7606-6/7606-4). Common patterns:

- **Variant columns** — one table with a column per product. Agent picks the right column.
- **Suffix variants** — ordering table maps part number suffixes to the few parameters that differ. Most specs are shared.
- **Conditional rows** — values gated by product name in the condition column. Agent filters by the target product.
- **Separate sections** — each variant gets its own spec section. JSON tree shows these as sibling nodes; agent navigates to the right one.
- **Ordering code encodes specs** — suffix determines which rows apply. Agent cross-references ordering info with spec tables.

The agent handles all five patterns through reasoning — it knows which product the user asked about and filters accordingly. For variant column tables, `inspect_page` is particularly useful since column alignment is often lost in raw text extraction.

---

## Implementation

### Inputs and Outputs

**Input:** A path to a local PDF file, or a URL to a datasheet PDF.

**Output:** The `build()` method writes two files to an output directory and returns a `DatasheetArtifacts` object:
- `{output_dir}/{filename}.json` — the enriched ToC JSON
- `{output_dir}/{filename}.txt` — the page-matched text file
- The returned `DatasheetArtifacts` also holds in-memory references for immediate use

### The `DatasheetIndex` Class

```python
class DatasheetIndex:
    """Pre-processes a datasheet PDF into agent-ready artifacts."""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = pymupdf.open(pdf_path)

    def build(self, output_dir: str | None = None,
              include_summaries: bool = False,
              llm_callable: Callable = None,
              output_stem: str | None = None,
              caption_figures: bool = True,
              max_figure_captions: int = 20,
              regenerate_toc: bool = False) -> DatasheetArtifacts:
        """Build the enriched ToC JSON and page-matched text file.

        Args:
            output_dir: Directory to write output files. None resolves to
                       $DATASHEETINDEX_OUTPUT_DIR or a UID-namespaced tempdir.
            include_summaries: Whether to generate LLM summaries per section.
                              Recommended only for large (300+ page) datasheets
                              or datasheets with poor ToC quality.
            llm_callable: Optional LLM function for ToC fallback, summaries,
                         and figure captioning.
                         Signature: (system: str, user: str) -> str
                         If not provided, low-quality ToC fallback and figure
                         captioning can still use a default client when
                         credentials are available.
            output_stem: Optional override for the deliverables' filename stem.
            caption_figures: Name raster figure regions with a vision model,
                            bounded by max_figure_captions. Default True, but
                            it is a no-op without a vision-capable client --
                            unlike include_summaries there is no client guard.
            max_figure_captions: Per-document ceiling on VLM caption calls.
                                 Must be an integer >= 0; raises ValueError
                                 otherwise.
            regenerate_toc: Force the LLM ToC fallback regardless of the
                            baseline's quality score. Requires an LLM client;
                            raises RuntimeError if none can be obtained,
                            rather than silently keeping the old ToC.

        Returns:
            DatasheetArtifacts with .json_path, .text_path, and in-memory data.
        """
        filename = Path(self.pdf_path).stem

        # Step 1: Generate the page-matched text file and the figure index in
        # one pass (the text is needed by all later steps)
        scan = scan_pages(self.doc)
        text_content = scan.text

        # Step 2: Generate the page-marked front matter and its per-page signals
        front_matter = build_front_matter(self.doc)
        preamble = front_matter.text

        # Step 3: Extract ToC (PyMuPDF get_toc(), instant)
        raw_toc = extract_toc(self.doc)

        # Step 4: Build tree with end_page computation and table counts
        tree = build_tree(raw_toc, len(self.doc))
        tree = enrich_with_table_counts(tree, self.doc)
        tree = enrich_with_continued_tables(tree, text_content)
        tree = enrich_with_footnote_markers(tree, text_content)
        tree = enrich_with_cross_references(tree, text_content)
        # `breadcrumb` and `boilerplate_category` are populated inside
        # `build_tree` above -- they only need the title and tree shape,
        # so they ride along with `assign_node_ids`.

        # Step 5: Assess ToC quality
        toc_quality = assess_toc_quality(tree, len(self.doc))

        # One client, shared by three branches -- but WHICH branch created it
        # is recorded, because summaries are gated on that (see
        # `_SUMMARY_CLIENT_ORIGINS`): a client self-created only to caption
        # figures must not turn `include_summaries=True` into per-section calls.
        active_llm_callable = llm_callable
        llm_client_origin = "caller" if llm_callable is not None else None
        needs_toc_fallback = regenerate_toc or toc_quality.score < 0.3
        has_caption_candidates = eligible_caption_count(scan.figures, ...) > 0
        if active_llm_callable is None and (
            needs_toc_fallback or has_caption_candidates
        ):
            active_llm_callable = self._try_create_default_llm_client()
            if active_llm_callable is not None:
                llm_client_origin = (
                    "toc_fallback" if needs_toc_fallback else "figure_captions"
                )

        # Step 6: If ToC is missing/poor and LLM is available, fall back
        if active_llm_callable and needs_toc_fallback:
            tree = generate_toc_from_text(text_content, len(self.doc), active_llm_callable)
            tree = enrich_with_table_counts(tree, self.doc)
            tree = enrich_with_continued_tables(tree, text_content)
            tree = enrich_with_footnote_markers(tree, text_content)
            tree = enrich_with_cross_references(tree, text_content)
            toc_quality = assess_toc_quality(tree, len(self.doc))

        # Step 7: Optionally add summaries (requires a sanctioned LLM client)
        if include_summaries and llm_client_origin in _SUMMARY_CLIENT_ORIGINS:
            add_summaries(tree, text_content, active_llm_callable)

        # Step 7b: Caption raster regions the text layer never named
        caption_figures_in_place(
            self.doc, scan.figures,
            vision_client=get_vision_client(active_llm_callable),
            max_figure_captions=max_figure_captions if caption_figures else 0,
        )

        # Step 8: Write output files
        json_data = {
            "source": filename + ".pdf",
            "total_pages": len(self.doc),
            "preamble": preamble,
            "preamble_pages": [p.to_dict() for p in front_matter.pages],
            "toc": [node.to_dict() for node in tree],
            "figures": scan.figures,
            "figures_excluded": {...},
            "figure_captions_excluded": {...},
        }
        json_path = Path(output_dir) / f"{filename}.json"
        text_path = Path(output_dir) / f"{filename}.txt"
        json_path.write_text(json.dumps(json_data, indent=2))
        text_path.write_text(text_content)

        return DatasheetArtifacts(
            json_path=str(json_path),
            text_path=str(text_path),
            json_data=json_data,
            text_content=text_content,
            toc_quality=toc_quality,
        )

    def _assess_toc_quality(self, tree) -> TocQuality:
        """Score the ToC and recommend whether summaries are needed."""
        # Factors: section count, hierarchy depth, title descriptiveness,
        #          page coverage, known section pattern matching
        ...
```

### The `DatasheetTools` Class

Lives in `tools/bound.py` -- a framework-neutral leaf module (it imports no
agent-framework code). Both the neutral tool defs and the SDK adapter build on
it, giving a one-directional import graph: `registry -> defs -> bound`. It is
re-exported from `tools/registry`, `tools`, and the top-level package for
backward compatibility.

```python
class DatasheetTools:
    """Bound datasheet tools the consuming agent can call."""

    def __init__(self, pdf_path: str):
        self._index = DatasheetIndex(pdf_path)

    @property
    def doc(self) -> pymupdf.Document:
        return self._index.doc

    def close(self) -> None:
        self._index.close()

    def inspect_page(
        self,
        page: int,
        region: dict[str, float] | None = None,
        dpi: int | None = None,
        detail: Detail = "medium",  # agent-surface default
    ) -> list[dict]:
        return inspect_page(
            self.doc, page, region=region, dpi=dpi, detail=detail
        )
```

### MCP / SDK Integration

The tool surface is designed in two layers so the tool *logic* is decoupled from
any one agent framework:

1. **`tools/defs.py` — the framework-neutral source of truth.**
   `create_datasheet_tool_defs()` returns the five tools (`build_datasheet`,
   `get_section_text`, `search_text`, `inspect_page`,
   `extract_table_markdown`) as plain `DatasheetToolDef` records -- `name`,
   `description`, `input_schema` (JSON Schema dict), and an async `handler`
   returning the `{"content": [...], "is_error": bool}` envelope. It imports **no**
   `claude-agent-sdk`. Hosts that are not on the Claude Agent SDK (pydantic-ai,
   plain function-calling agents, custom MCP servers) wrap each `handler`
   directly.
2. **`tools/registry.py` — the Claude Agent SDK adapter.**
   `create_datasheet_tools_server()` is a thin wrapper that lazily imports the SDK
   and wraps each neutral def with the SDK `@tool` decorator. Because it derives
   from the same defs, the SDK surface exposes byte-identical tool names,
   descriptions, and schemas.

`build_datasheet` returns the enriched ToC manifest, including the bounded
`figures` digest described under "Figure indexing"; `search_text` accepts a
single pattern or a list and tags each hit with its section breadcrumb;
`get_section_text` returns a page range prefixed with a position header.

`search_text` also carries the text-layer limitation on **both** surfaces its
consumer reads: in the tool description, and — when a search returns nothing on
a document that holds raster regions — as a `note` on the result itself. The
second is not redundancy. A description is read once at tool-registration time;
the inference "zero hits does not mean absent, because some of this document is
pixels" has to be available at the turn the agent draws the wrong conclusion.
The gate is `DatasheetTools.has_raster_figures()`, which counts `"raster"`
entries only: a `"caption"` entry comes from the text layer, so its words are
searchable and nothing is hidden. Both negatives are load-bearing — no note on
a successful search, none on a caption-only document — because a note that
appears unconditionally is one the agent learns to skip.

#### How the tool text is divided, and why it is short

A tool definition is re-sent on every request, so its length is a standing
cost. The division that keeps it honest:

- **The description answers two questions only** — when do I call this, and
  what comes back. Nothing else belongs there.
- **Everything about an argument lives on the argument**, in its JSON Schema
  `description`. Guidance stays attached to what it describes, and a reader
  scanning one parameter is not reading five paragraphs about the others.
  `tests/test_defs.py` fails if any parameter has no description.
- **No emphasis markers.** `IMPORTANT`, `CRITICAL`, `MUST`, `Do NOT` and
  `CALL THIS FIRST` are all rejected by a test. They were written to stop
  older models under-triggering; Anthropic's Claude 4.5/4.6 guidance is that
  the same language now pushes the other way, and ordinary prose is the fix.
- **Each description has a length budget**, set just above what the current
  text needs, so drift back toward an essay fails a test rather than quietly
  taxing every turn.

This cut the five descriptions from 5750 characters to 2900. The information
was preserved, not dropped: what left the prose moved into the parameters, so
the full serialized surface fell 8389 → 7000 characters. Checked against a
live model on the case the text exists for — an empty `search_text("SUMITOMO")`
beside a figures row naming a supplier table — both the old and the new surface
chose `inspect_page(page=5, detail="medium")` on 3 of 3 trials, and the new
one names the text-layer cause in its reasoning.

#### The image block carries two media-type keys

`inspect_page` is the only tool returning a non-text block, and its envelope
spells the media type **twice**:

```python
{"type": "image", "data": "<base64 PNG>",
 "mime_type": "image/png", "mimeType": "image/png"}
```

This is deliberate and must not be tidied down to one key. The envelope is the
Claude Agent SDK's envelope format, and that format is mixed-case *by
construction*: the SDK reads `is_error` (snake_case) for the result but
`item["mimeType"]` (camelCase) for an image block, from the same dict. So there
is no single spelling that satisfies every reader:

- Emitting only `mime_type` is what shipped through 0.21.0. Every `inspect_page`
  call through `create_datasheet_tools_server()` raised `KeyError('mimeType')`
  inside the SDK's own converter — see the gotcha in issue #13.
- Emitting only `mimeType` breaks the other direction:
  `mcp_server._envelope_to_content` and any host already reading the documented
  snake_case key.

Note the *library primitive* `tools/vision.py:inspect_page` still returns
`mime_type` alone; the dual key is added by the handler in `tools/defs.py` when
it builds the envelope. `DatasheetTools.inspect_page` returns the primitive's
block, not the envelope.

**Testing.** `create_datasheet_tools_server` is only as correct as the converter
it feeds, and the suite used to stub that converter out entirely — the fake
`create_sdk_mcp_server` accepted the envelope and never read a key from it, so
the envelope could spell a key any way it liked and every SDK test still passed.
That is how #13 survived two months in production. Two layers now cover it:
`tests/conftest.py:sdk_envelope_to_content` mirrors the real converter key for
key and runs in the default lane, and `tests/test_sdk_integration.py` runs the
genuine SDK and pins the mirror against it. The latter needs the optional `sdk`
dependency group (`uv sync --group sdk`) and skips without it — it is not in
`dev` because the wheel unpacks to ~263 MB, almost all of it a bundled `claude`
CLI binary the tests never invoke.

Per-session state lives in the `create_datasheet_tool_defs()` closure: the
current `DatasheetTools` is bound (and rebound) by the `build_datasheet` handler
and read by the others, so **one factory call == one session**. The server starts
**unbound and takes no arguments** -- the agent loads a document at runtime by
calling `build_datasheet` with a `pdf_source` (local path or URL) before using
any other tool. A failed switch to a bad source leaves the previously bound
document intact (the new source is built into a fresh instance and only swapped
in on success).

```python
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DatasheetToolDef:
    name: str
    description: str
    input_schema: dict[str, Any]                                    # JSON Schema
    handler: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]  # -> envelope


def create_datasheet_tool_defs() -> list[DatasheetToolDef]:
    """Framework-neutral: the five tools as plain defs, no claude-agent-sdk import."""
    tools_instance: DatasheetTools | None = None

    async def build_datasheet(args: dict[str, Any]) -> dict[str, Any]:
        nonlocal tools_instance
        ...  # binds/rebinds tools_instance for the current session

    async def inspect_page(args: dict[str, Any]) -> dict[str, Any]:
        ...  # the other handlers _require() the bound tools_instance

    return [
        DatasheetToolDef(
            name="inspect_page",
            description=(
                "Render a PDF page as a PNG image for visual inspection. Use when "
                "text extraction is insufficient (tables, figures, formulas)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "page": {"type": "integer", "minimum": 1},
                    "region": {"type": "object", "description": "Crop 0.0-1.0"},
                    "detail": {"type": "string", "enum": ["low", "medium", "high"]},
                    "dpi": {"type": "integer"},
                },
                "required": ["page"],
            },
            handler=inspect_page,
        ),
        ...  # build_datasheet, get_section_text, search_text, inspect_page, ...
    ]


def create_datasheet_tools_server():
    """Thin Claude Agent SDK adapter over create_datasheet_tool_defs().

    Requires claude-agent-sdk. Takes no arguments and starts unbound.
    """
    from claude_agent_sdk import create_sdk_mcp_server, tool

    return create_sdk_mcp_server(
        name="datasheetindex",
        version=package_version(),
        tools=[
            tool(d.name, d.description, d.input_schema)(d.handler)
            for d in create_datasheet_tool_defs()
        ],
    )
```

The server object is the thing a Claude Agent SDK host mounts. `claude-agent-sdk`
is intentionally optional -- it is needed only for `create_datasheet_tools_server`,
not for the tool logic -- so the core preprocessing path stays lightweight and
non-SDK hosts pull in nothing extra.

The consuming agent sets this up alongside the pre-processed artifacts:

```python
from datasheetindex import create_datasheet_tools_server

# No arguments; starts unbound. The agent calls the build_datasheet tool with a
# pdf_source before using any other tool.
datasheet_tools_server = create_datasheet_tools_server()

# Pseudocode: the exact agent wiring depends on the host runtime.
agent = SomeAgentRuntime(
    mcp_servers={"datasheet-tools": datasheet_tools_server},
    system_prompt=build_extraction_prompt(...),
)
```

A non-SDK host instead wraps the neutral defs directly. When the session needs
end-of-life cleanup (a URL source leaves a temporary file behind until closed),
use `create_datasheet_tool_session`, which returns the same defs plus a `close`:

```python
from datasheetindex import create_datasheet_tool_session

session = create_datasheet_tool_session()
for d in session.defs:
    register_with_your_host(d.name, d.description, d.input_schema, d.handler)
# ... on shutdown:
session.close()
```

**All three surfaces share one source of truth.** The local MCP server
(`mcp_server.py`, run via `datasheetindex-mcp-server`) is a third thin adapter:
it serves `create_datasheet_tool_session()` on a low-level `mcp` `Server`
(one `list_tools` + one `call_tool` that translates the neutral envelope into MCP
content blocks), and wires it onto the stdio / streamable-http / sse transports.
So the SDK server, the local MCP server, and non-SDK hosts all present identical
tool names, descriptions, and JSON schemas -- a change to a tool def propagates to
every surface from one place.

### Page-cut truncation signal

A section's ToC page range does not always contain the whole of its table. In
the TI TCAN1044A-Q1, `6.4 Recommended Operating Conditions` has ToC range pages
4-4, but its table continues onto page 5 -- so an agent reading the whole
section still loses rows (including `TJ`, the operating junction temperature).
The evidence of the cut, the publisher's `(continued)` marker, sits at the top
of the page the agent did not fetch.

`get_section_text` therefore probes both boundaries of the requested range and
inserts a `=== NOTE: ... ===` line under the position header when the range
cuts content marked as continuing. The `===` wrapper matters: real datasheets
(e.g. TI TCAN1044A-Q1 page 26) contain their own literal `NOTE:` lines in body
text, and a bare `NOTE:` prefix would collide with them, producing a false
truncation signal. The check is **range-relative**, not section-relative: the
TI case is a whole-section read, so a section-aware check would miss it.

A marker is honoured only if it appears within the first `_OPENING_BLOCK_LINES`
(5) nonblank lines of the following page -- a table that resumes does so at the
top of the page. Measured across the Infineon and TI datasheets, genuine
continuations sit at nonblank line 3, while the mid-page `NOTES: (continued)`
blocks on TI's mechanical-drawing pages sit at lines 19-48.

**The positional guard is the whole correctness property. Do not replace it with
a content check.** The obvious-looking alternative -- accept the marker only if
its title also appears on the *preceding* page, "proving" it is a continuation --
was measured and rejected: all ten markers in the corpus pass it, including all
six `NOTES:` false positives, so it discriminates nothing. It looks like a guard
and is not one. Both edges of `_OPENING_BLOCK_LINES` are pinned by tests in
`tests/test_continuation_boundary.py`; a drift in either direction fails them.

The title-length bound in `_CONTINUATION_RE` is **not** a second guard. It exists
only to say "this is a title, not a paragraph that happens to end in
(continued)". It must stay generous: vendors repeat the full parameterised
caption on the continuation page (`Table 12. Electrical characteristics (VDD =
3.3 V, TA = 25 degC, unless otherwise specified) (continued)` is ~100
characters), and a tight bound silently drops them -- a false negative in exactly
the failure class this signal exists to eliminate.

**Silence is not a completeness claim.** Two distinct cases are accepted as
invisible to this signal, and both are known limitations rather than bugs:

- Content can spill across a page break with no marker at all.
- A marker can exist but share its line with trailing text -- e.g. PyMuPDF
  merging a heading with the column-header row into one block-line, `"6.4
  Recommended Operating Conditions (continued) MIN NOM MAX UNIT"` -- which
  `_CONTINUATION_RE`'s trailing `$` anchor does not match. The anchor is kept
  deliberately: relaxing it to tolerate trailing text would let ordinary prose
  ending in "(continued) ..." false-positive.

The note therefore states only that the publisher marked the next page as
continuing; it asserts nothing about rows or column headers (a continuation
page often repeats its headers), and never claims a range is complete.

This is a different concept from `TocNode.continued_tables`, which keeps its own
narrower contract: tables captioned `Table N ... (Continued)`.

### Agent System Prompt Guidance

The system prompt below is **reference guidance for the consuming agent**, not part of this library. It is included here to document the intended usage pattern and inform tool design decisions:

```
You have access to a datasheet with:
1. A structured JSON map (enriched ToC) — sections, page ranges, table hints
2. A text file of the full document with page markers (--- PAGE N ---)
3. MCP tools that can build and read the artifacts, search the extracted text,
   call `inspect_page` for visual inspection, and re-extract a page's tables
   as Markdown (`extract_table_markdown`)

WORKFLOW:

Phase 1 — ORIENT (do this once, before any extraction):
  1. Read the JSON "preamble" — this is pages 1-2 of the datasheet,
     giving you the product name(s), key features, and performance summary.
     From this you can tell immediately if it's single or multi-product,
     and what the key differentiators are (voltage, speed, package, etc.)
  2. Scan the JSON tree to find relevant structural sections:
     - Operating ranges / functional range / recommended conditions
     - Ordering information
     These tell you how product variants map to VCC levels, temp grades, etc.
  3. If multi-product: read the operating ranges section from the text file
     to learn the VCC/temperature mapping for each variant:
     e.g., "1.8V device → VCC = 1.7-2.0V", "3.0V device → VCC = 2.7-3.6V"
  4. Now you have the context to correctly filter variant-specific parameters

Phase 2 — EXTRACT (for each query):
  1. Use the JSON to identify which sections are relevant
  2. Read those pages from the text file
  3. Apply your orientation knowledge to filter for the correct product
  4. Use inspect_page when you hit a situation listed below

WHEN TO USE inspect_page:
Use it — the visual is always more reliable than raw text for these cases:

• FORMULA VALUES split across lines
  Text shows: "VVREG" / "OUT -" / "0.3" — is this one value or three?
  Visual shows: "V_VREGOUT - 0.3" clearly in the Min column.

• MIN/TYP/MAX AMBIGUITY
  Text shows values on separate lines without column alignment.
  Visual shows values in clearly labeled Min/Typ/Max columns.

• FIGURES AND DIAGRAMS referenced in text
  "see Figure 3-2 for derating curve" — call inspect_page to read the graph.
  "Pin configuration in Figure 2" — call inspect_page to see the pin diagram.

• SPARSE OR GARBLED TEXT on a page
  If a page that should have content (per JSON) has very little text,
  it likely contains diagrams or has extraction issues — inspect visually.

• HIGH-STAKES VALUES you want to double-check
  For safety-critical parameters (absolute maximum ratings, thermal limits),
  inspect the page to verify what you read from text.

REGION CROPPING — use it to improve visual accuracy:
When inspecting a specific table or figure, crop to the region of interest
using the region parameter with percentage-based coordinates (0.0 to 1.0):
  inspect_page(page=24, region={"top": 0.0, "bottom": 0.5, "left": 0.0, "right": 1.0})
  This example crops to the top half of the page.
  • Top half: {"top": 0.0, "bottom": 0.5, "left": 0.0, "right": 1.0}
  • Bottom half: {"top": 0.5, "bottom": 1.0, "left": 0.0, "right": 1.0}
  • For full-page tables or when you're unsure of the layout, skip cropping
    and inspect the full page — that's always safe.
  • You can iteratively refine: inspect full page first to see the layout,
    then crop to the specific area if you need to re-read values precisely.

WHEN TEXT IS SUFFICIENT (no need to inspect):
• Simple parameter rows: "VVS_max -0.3 – 75 V" — clear from text
• Section headers, functional descriptions, notes — pure text content
• Table continuations with "(continued)" markers — structure is clear
• Footnotes referenced by markers like "1)" — usually on same/next page

MULTI-PRODUCT DATASHEETS:
Some datasheets cover a product family (e.g., TPS651/652/653) in one PDF.
When extracting for a specific product:
• Check early pages for an ordering table or product overview that maps
  part numbers to their differences (often just a few parameters differ)
• In tables with variant columns, pick the column for the target product
• In tables with conditional rows, filter by the target product name
• If sections are split per variant (e.g., "6.1 AD7606 Specs"), navigate
  to the right section using the JSON tree
• Report clearly which values are shared across the family vs specific
  to the requested product
• When in doubt, use inspect_page — variant column alignment is often
  lost in raw text extraction

SELF-CHECK after extracting parameters from each section:
• Are min ≤ typ ≤ max? If not, re-read the source — values may be swapped.
• Does every numeric value have a unit? If not, check the table header or
  column header for the unit.
• Did you capture footnotes? If a value has "(1)" or "Note 1" but your
  notes array is empty, go find the footnote text.
• Did you extract the same parameter twice with different values? Check
  whether they come from different sections (AMR vs operating conditions)
  and report both with context, or whether one is an error.

GENERAL PRINCIPLE:
Read text first. Most parameters can be extracted from text alone.
Call inspect_page when you're uncertain — it's cheap (renders one page)
and always gives you the ground truth.

OUTPUT FORMAT:
Return every extracted parameter as a structured object:
{
  "parameter": "Supply voltage VS",
  "symbol": "VVS_max",
  "min": -0.3,
  "typ": null,
  "max": 75,
  "unit": "V",
  "conditions": ["Tj = -40°C to +150°C"],
  "notes": [],
  "source_page": 9,
  "source_table": "Table 1 Absolute maximum ratings",
  "extraction_method": "text"
}

The extraction_method field records HOW the value was obtained:
- "text" — extracted from the text file (default, most common)
- "visual" — extracted via inspect_page (used when text was ambiguous)

EXAMPLE EXTRACTIONS (follow this pattern):

Input text: "VVS_max -0.3 – 75 V –  PRQ-486"
Output: {"parameter": "Supply voltage VS", "symbol": "VVS_max",
         "min": -0.3, "typ": null, "max": 75, "unit": "V",
         "conditions": ["Tj = -40°C to +150°C, all voltages w.r.t. GND"],
         "source_page": 9, "source_table": "Table 1", "extraction_method": "text"}

Input text (ambiguous): "VVS_rel_max VVREG OUT - 0.3 – – V"
Action: Call inspect_page(9) to verify → visual shows "V_VREGOUT - 0.3" as Min
Output: {"parameter": "Supply voltage VS relative", "symbol": "VVS_rel_max",
         "min": "VVREGOUT - 0.3", "typ": null, "max": null, "unit": "V",
         "source_page": 9, "source_table": "Table 1", "extraction_method": "visual"}
```

---

## Building on PageIndex

### What we keep:
- **Hierarchical tree structure** — the foundation of the JSON output
- **LLM-based ToC generation** as a fallback when PyMuPDF `get_toc()` returns empty/poor results
- **Recursive sub-section discovery** for large nodes without sub-ToC entries
- **Section summaries** — optional, for large or poorly-structured datasheets

### What we replace:
- **ToC detection** — PyMuPDF `get_toc()` instead of multi-LLM-call detection
- **Token counting** — simple approximation instead of tiktoken dependency
- **Text extraction** — PyMuPDF `get_text("blocks")` with column-aware reordering for the text file

### What we add:
- **Preamble** — page-marked pages 1-2 raw text embedded in JSON (up to 5000 characters, ~1250 tokens) plus per-page signals in `preamble_pages`, for agent orientation; zero heuristics, zero LLM calls
- **Table detection hints** — `has_tables` / `table_count` per node from PyMuPDF (best-effort heuristic; agent reads actual text and judges for itself)
- **`figures` / `figures_excluded`** — every raster image placement enumerated exactly (`get_image_info()`, not inferred), plus every `Figure N` / `Fig. N` text-layer caption; vector figures are still not detected by clustering drawing operations, but they leak their text so the agent is not blind there
- **`breadcrumb`** — pre-computed full ancestry path per node (e.g. `"5 Electrical Characteristics > 5.1 Absolute Maximum Ratings"`), so downstream agents and RAG indexers see structural context without re-traversing parents
- **`boilerplate_category`** — title-pattern flag for `legal` / `ordering` / `revision` / `contact` / `toc` / `glossary` sections, so agents can deprioritize disclaimers, revision histories, and similar admin content. Title-only regex matching (no LLM, no text scan); children of flagged parents inherit the category
- **ToC quality assessment** — auto-detect whether summaries are worth generating
- **Page-matched text file** — PyMuPDF `get_text("blocks")` with column-aware reordering and page markers
- **Running header/footer stripping** — a block inside the top/bottom 20% band whose whitespace-collapsed, digit-masked key contains at least one letter and recurs on at least half the pages -- or dominates one page parity, the alternating odd/even header -- is dropped from the page-matched text file. A simplified Lin page-association; block granularity is what keeps `Table N (continued)` captions intact. The preamble keeps raw text. See "Running header/footer stripping" under Deliverable 2 for the measurements behind each constant.
- **Vision as primary escalation** — `inspect_page` for when text isn't sufficient
- **`locate_text`** (Python API, not an agent tool) — text-to-coordinate source grounding (bounding boxes as
  percentages + PDF points), so an agent or review UI can turn a located string
  into a precise highlight or a tightly cropped `inspect_page` call
- **Agent tools** — `build_datasheet`, `get_section_text`, `search_text`,
  `inspect_page`, and `extract_table_markdown`, with text-first navigation,
  breadcrumb-tagged single- or multi-pattern search across wrapped/interleaved
  table text, position-headed section reads, and visual escalation when needed
- **Page alignment validation** — ensure JSON page numbers match text file markers
- **Zero extra dependencies** — only PyMuPDF needed for the happy path (no pymupdf4llm, no pdfplumber)
- **LLM as injectable callable** — the LLM fallback and summarizer accept a `llm_callable: (system, user) -> str` parameter rather than depending on a specific LLM client library. The consuming application provides its own LLM client (e.g., LiteLLM gateway with gpt-4.1 over Chat Completions). This keeps the library dependency-free for the happy path while allowing LLM features when needed.
- **Structured-output ToC fallback with candidate gating** — when the injected callable also exposes `structured_json(...)` (detected via `get_structured_output_client()`, an optional extension of the base callable protocol), `toc_fallback.py` requests Chat Completions' `response_format={"type": "json_schema", ...}` mode per chunk instead of best-effort-parsing free text. Failure is isolated per chunk: an incomplete or malformed chunk response is logged and skipped, never fatal to the chunks around it, and a structured path that yields nothing at all (a model that rejects `json_schema`) degrades to the free-text prompt for the whole document. Deciding whether the surviving entries are good enough is a separate job from parsing them, and it belongs to the gate below. Either way, the regenerated ToC is only a *candidate*: `index.py` scores it with the same `assess_toc_quality()` used for the original and only replaces the original ToC if `_accept_llm_toc_candidate()` judges it clearly not worse (a strictly better score, no page-coverage regression, and — only when there is a real ToC to protect — enough entries for the document's size). This exists because `assess_toc_quality()`'s page-coverage term can score a single fallback node deceptively well once `build_tree()` extends its `end_page` to the document's last page — without gating, that thin result would silently replace a working original ToC. The gate deliberately does *not* punish a candidate for having fewer entries than the baseline: the score already weights entry count, coverage, and depth, so rejecting a higher-scoring candidate for being smaller than a bloated pseudo-ToC would keep the junk it was called in to replace. An explicit `regenerate_toc=True` skips the score comparison and nothing else, because that comparison is precisely what the escalation routes around — see "ToC quality: what the score decides, and how a caller overrules it" above.

### Lessons from Google's LangExtract

LangExtract (Google, 2025) is a text extraction library that uses LLM-powered chunking, multi-pass extraction, and source grounding. While designed for unstructured narrative text (clinical notes, legal docs), three ideas transfer well to our domain:

1. **Source grounding** — every extraction maps to its exact source location. In our architecture, every parameter cites `source_page` and `source_table`, making the extraction auditable and traceable.

2. **Structured output schema** — LangExtract enforces a consistent extraction format via few-shot examples and controlled generation. Our agent prompt includes a defined output schema and example extractions so results are always structured and parseable.

3. **Extraction method tracking** — similar to LangExtract's character offset tracing, our `extraction_method` field records whether a value came from text parsing or visual inspection, providing transparency about extraction confidence.

What we don't adopt from LangExtract (and why):
- **Blind text chunking** — LangExtract chunks text positionally. We use ToC-based semantic navigation, which is far superior for structured documents like datasheets.
- **Multi-pass extraction** — useful for unstructured text where entities are scattered. Datasheets have parameters in clear tables; our challenge is text quality, not entity recall. `inspect_page` is a better solution for our domain.
- **Parallel chunk processing** — our agent navigates sequentially by design, reading the relevant sections identified by the JSON tree.

---

## Module Structure

```
datasheetindex/
├── core/
│   ├── structure.py       # ToC extraction → enriched tree JSON
│   │                      #   PyMuPDF get_toc() primary
│   │                      #   PageIndex LLM fallback
│   │                      #   Table count enrichment
│   │                      #   ToC quality assessment
│   ├── textfile.py        # PDF → page-matched text file
│   │                      #   Column-aware block extraction with page markers
│   │                      #   Page alignment validation
│   ├── furniture.py       # Running header/footer decision logic
│   │                      #   normalized-key recurrence within a page-edge band
│   ├── _textmatch.py      # Shared dash/token normalization + matcher
│   ├── locate.py          # locate_text: text -> bounding-box coordinates
│   ├── artifact_cache.py  # Build sidecar: fingerprint, validity, atomic writes
│   │                      #   <stem>.build.json beside the two deliverables
│   ├── figures.py         # raster_regions: exact raster placements, clipped
│   │                      #   to the page and normalized for inspect_page
│   ├── preamble.py        # Page-marked front matter + per-page signals
│   └── quality.py         # Page-level quality scoring
│                          #   (text density, extraction confidence)
├── tools/
│   ├── vision.py          # inspect_page (page → image for visual inspection)
│   ├── bound.py           # DatasheetTools (document-bound tool logic; neutral leaf)
│   ├── defs.py            # create_datasheet_tool_defs (framework-neutral tool defs)
│   └── registry.py        # Claude Agent SDK adapter (re-exports DatasheetTools)
├── llm/
│   ├── client.py          # Optional LLM client (free-text + structured_json + vision)
│   ├── toc_fallback.py    # PageIndex-style LLM ToC generation
│   │                      #   Page numbers validated against chunk markers
│   ├── summarizer.py      # Optional section summaries
│   ├── figure_captions.py # VLM captioning for raster figure regions,
│   │                      #   bounded by max_figure_captions
│   └── untrusted.py       # Framing for document text sent to an LLM
│                          #   (PDF text is untrusted input, not instructions)
├── index.py               # Main DatasheetIndex class
└── models.py              # Data models
```

---

## Implementation Priority

### Phase 1: Core (the two deliverables)
- `DatasheetIndex.build()` producing enriched JSON + text file
- PyMuPDF ToC extraction with tree building and table/figure metadata
- PyMuPDF text file generation with page markers
- Page alignment validation
- ToC quality scoring

### Phase 2: Agent Tools
- `inspect_page` — page rendering as image for visual inspection
- `locate_text` — text-to-coordinate grounding (bounding boxes for highlighting); Python API only, not an agent tool
- Tool registration for Claude Agent SDK

### Phase 3: LLM Fallbacks
- PageIndex-style ToC generation for PDFs with missing/poor ToC
- Optional section summaries (gated by ToC quality score)
- Recursive sub-section discovery for large nodes

### Phase 4: Refinement
- Multi-page table detection hints in the JSON
- Footnote marker detection in table cells (flagged in JSON)
- Cross-reference detection ("see Section X" → linked node_id)
- Batch processing for multiple datasheets
