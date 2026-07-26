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

The import must happen inside the lock, because the import *is* the hook installation. An import racing `classic_tables()` lets the guard save `None`, the import install the hook, and the guard restore its stale `None`. Since `pymupdf4llm._use_layout` stays `True`, `to_markdown()` then iterates a `None` `page.layout_information` and raises `TypeError` — permanently, because the module is cached in `sys.modules`. `build_datasheet` and `extract_table_markdown` both run under `asyncio.to_thread`, so this race is reachable.

The two engines are different heuristics, not better and worse. On a real 68-page datasheet the classic detector finds 75 tables to the ML engine's 39 and is ~4.4x faster; the ML engine's extra misses are the "Typical Characteristics" plot pages, where the classic detector false-positives on chart gridlines. Counts are defined as the classic detector's answer so they do not depend on which optional extras happen to be installed.

---

## The Two Deliverables

### Deliverable 1: Enriched ToC JSON

Not just a flat table of contents — a hierarchical tree with enough metadata for the agent to make informed navigation decisions. Includes a **preamble** — raw text from pages 1-2 — so the agent can orient itself before extraction.

The `preamble` is generated automatically with zero LLM calls. Rather than fragile heuristics to detect product names or classify ToC entries (which break across manufacturers), the library embeds the raw text of pages 1-2 as a `preamble` — giving the agent the context to orient itself.

Why not parse it programmatically? Because:
- Part number regex produces false positives ("JEDEC51", "AEC100") and misses wildcards ("TPS6513x")
- ToC keyword matching is manufacturer-specific ("Operating ranges" vs "Functional range" vs "Recommended operating conditions")
- An LLM call would work but adds a dependency to pre-processing (the happy path currently requires zero LLM calls)

The agent IS the LLM — let it reason about the preamble text directly.

```json
{
  "source": "infineon-tle9009dqu-datasheet-en.pdf",
  "total_pages": 73,
  "preamble": "TLE9009DQU\nLi-ion battery monitoring and balancing IC\n\nFeatures\n• Voltage monitoring of up to 9 battery cells connected in series\n• Hot plugging support\n• Dedicated 16-bit high precision delta-sigma ADC for each cell...",
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
  substance a picture is withholding from the text layer.
- **`"caption"`** -- a `Figure N` / `Fig. N` mention recognized in the
  column-aware page text, in either of two forms (same-line with a mandatory
  `.`/`:` separator, or split across two lines), including section-relative
  numbering (`"10-1"`). `figure_number` is always a string, never coerced to
  an int -- it is an identifier to display and match on, not an arithmetic
  value, and a union type would cost every consumer a branch for no benefit.

Every raster region above the threshold is also a candidate for VLM
captioning (`caption_source: "llm"`), which fills in a one-line description
for regions the text layer never named -- see `llm/figure_captions.py` and
the README for the cost, the cap, and the default-on behaviour.

**The agent is handed a digest, not the array.** `build_datasheet`'s manifest
(`tools/bound.py:get_artifact_manifest`) carries a bounded `figures` block --
`total` / `raster` / `captioned` counts, plus one `{page, figures, caption}`
row per page holding figures in ascending page order. The array itself stays in
the ToC JSON: the manifest is returned on every build, and a scanned document
can hold one full-page raster per page, so the digest is capped at 40 rows with
one 200-character caption each (`pages_with_figures` and `truncated` disclose
what was dropped). Carrying *something* is not optional -- the MCP agent
receives only the manifest, and per the WSL namespace gotcha `json_path` may
not even be readable from where the agent runs, so a digest is the difference
between the agent knowing a page holds a figure and never learning the figure
index exists.

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
              max_figure_captions: int = 20) -> DatasheetArtifacts:
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

        Returns:
            DatasheetArtifacts with .json_path, .text_path, and in-memory data.
        """
        filename = Path(self.pdf_path).stem

        # Step 1: Generate the page-matched text file and the figure index in
        # one pass (the text is needed by all later steps)
        scan = scan_pages(self.doc)
        text_content = scan.text

        # Step 2: Generate preamble (pages 1-2 raw text, ~600 tokens)
        preamble = generate_preamble(self.doc)

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
        needs_toc_fallback = toc_quality.score < 0.3
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
- **Preamble** — pages 1-2 raw text embedded in JSON (~600 tokens) for agent orientation; zero heuristics, zero LLM calls
- **Table detection hints** — `has_tables` / `table_count` per node from PyMuPDF (best-effort heuristic; agent reads actual text and judges for itself)
- **`figures` / `figures_excluded`** — every raster image placement enumerated exactly (`get_image_info()`, not inferred), plus every `Figure N` / `Fig. N` text-layer caption; vector figures are still not detected by clustering drawing operations, but they leak their text so the agent is not blind there
- **`breadcrumb`** — pre-computed full ancestry path per node (e.g. `"5 Electrical Characteristics > 5.1 Absolute Maximum Ratings"`), so downstream agents and RAG indexers see structural context without re-traversing parents
- **`boilerplate_category`** — title-pattern flag for `legal` / `ordering` / `revision` / `contact` / `toc` / `glossary` sections, so agents can deprioritize disclaimers, revision histories, and similar admin content. Title-only regex matching (no LLM, no text scan); children of flagged parents inherit the category
- **ToC quality assessment** — auto-detect whether summaries are worth generating
- **Page-matched text file** — PyMuPDF `get_text("blocks")` with column-aware reordering and page markers
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
- **LLM as injectable callable** — the LLM fallback and summarizer accept a `llm_callable: (system, user) -> str` parameter rather than depending on a specific LLM client library. The consuming application provides its own LLM client (e.g., LiteLLM gateway with gpt-4.1 via the Responses API). This keeps the library dependency-free for the happy path while allowing LLM features when needed.
- **Structured-output ToC fallback with candidate gating** — when the injected callable also exposes `structured_json(...)` (detected via `get_structured_output_client()`, an optional extension of the base callable protocol), `toc_fallback.py` requests the Responses API's `text.format=json_schema` mode per chunk instead of best-effort-parsing free text. Failure is isolated per chunk: an incomplete or malformed chunk response is logged and skipped, never fatal to the chunks around it, and a structured path that yields nothing at all (a model that rejects `json_schema`) degrades to the free-text prompt for the whole document. Deciding whether the surviving entries are good enough is a separate job from parsing them, and it belongs to the gate below. Either way, the regenerated ToC is only a *candidate*: `index.py` scores it with the same `assess_toc_quality()` used for the original and only replaces the original ToC if `_accept_llm_toc_candidate()` judges it clearly not worse (a strictly better score, no page-coverage regression, and — only when there is a real ToC to protect — enough entries for the document's size). This exists because `assess_toc_quality()`'s page-coverage term can score a single fallback node deceptively well once `build_tree()` extends its `end_page` to the document's last page — without gating, that thin result would silently replace a working original ToC. The gate deliberately does *not* punish a candidate for having fewer entries than the baseline: the score already weights entry count, coverage, and depth, so rejecting a higher-scoring candidate for being smaller than a bloated pseudo-ToC would keep the junk it was called in to replace.

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
│   ├── _textmatch.py      # Shared dash/token normalization + matcher
│   ├── locate.py          # locate_text: text -> bounding-box coordinates
│   ├── artifact_cache.py  # Build sidecar: fingerprint, validity, atomic writes
│   │                      #   <stem>.build.json beside the two deliverables
│   ├── figures.py         # raster_regions: exact raster placements, clipped
│   │                      #   to the page and normalized for inspect_page
│   ├── preamble.py        # Pages 1-2 raw text for agent orientation
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
