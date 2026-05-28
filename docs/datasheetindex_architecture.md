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

Note: `has_tables` and `table_count` are heuristic hints from PyMuPDF (false positives on block diagrams are expected). `source` is always the filename, not the full path. In this example, node "0001" has `table_count: 2` which are false positives from block diagram boxes.

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

**No `has_figures` field.** No library reliably detects vector diagrams. The agent infers figure presence from section titles ("Block diagram", "Pin configuration") and text references ("see Figure X"), which is more reliable than any programmatic detection.

**What's optionally included (LLM-powered, debatable):**
- `summary` per node — useful for very large datasheets (300+ pages) where the agent needs help deciding which of 50 sections to look at. For smaller datasheets (< 100 pages), the section titles alone are usually descriptive enough. This should be configurable.

**The decision on summaries:** If the ToC is high quality (descriptive titles, proper hierarchy, correct page numbers), summaries add cost without much value. If the ToC is sparse or uses cryptic section numbers, summaries become essential. The library should score ToC quality and recommend whether summaries are worth generating.

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

The agent has one custom tool beyond the built-in file reading capabilities of the Claude Agent SDK.

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
        Content block list for the Claude Agent SDK:
        [{"type": "image", "data": "<base64 PNG>", "mime_type": "image/png"}]
    """
```

**Region uses percentage-based coordinates (0.0-1.0)** rather than PDF points. This is intentional: the agent doesn't know page dimensions in points, but it can reason about "top half", "bottom third", or "left two-thirds" naturally. The library converts percentages to PDF point coordinates internally.

Common region patterns:
- Top half: `{"top": 0.0, "bottom": 0.5, "left": 0.0, "right": 1.0}`
- Bottom half: `{"top": 0.5, "bottom": 1.0, "left": 0.0, "right": 1.0}`
- Full page: omit region (default)

**Return format** follows the Claude Agent SDK tool response convention: a list of content blocks. The image is base64-encoded PNG. The calling code in `tools/registry.py` handles this wrapping so the agent receives the image directly.

**Why only one tool?** We evaluated and dropped three other tools during design:
- `get_table` — PyMuPDF `find_tables()` produces worse results than raw text on complex semiconductor tables. Visual inspection is more reliable.
- `get_figure` — `get_images()` can't detect vector graphics (which is how 95% of datasheet diagrams are rendered). Visual inspection shows figures in full page context.
- `get_page_tables_overview` — Returns row/column counts from an unreliable detector. The text file already gives richer information: table titles, column headers, "(continued)" markers, and actual values.

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

    def build(self, output_dir: str = "output",
              include_summaries: bool = False,
              llm_callable: Callable = None) -> DatasheetArtifacts:
        """Build the enriched ToC JSON and page-matched text file.

        Args:
            output_dir: Directory to write output files.
            include_summaries: Whether to generate LLM summaries per section.
                              Recommended only for large (300+ page) datasheets
                              or datasheets with poor ToC quality.
            llm_callable: Optional LLM function for ToC fallback and summaries.
                         Signature: (system: str, user: str) -> str
                         If not provided, low-quality ToC fallback can still use
                         the default client when credentials are available.

        Returns:
            DatasheetArtifacts with .json_path, .text_path, and in-memory data.
        """
        filename = Path(self.pdf_path).stem

        # Step 1: Generate page-matched text file (needed by all later steps)
        text_content = generate_text(self.doc)

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

        active_llm_callable = llm_callable
        if active_llm_callable is None and toc_quality.score < 0.3:
            active_llm_callable = self._try_create_default_llm_client()

        # Step 6: If ToC is missing/poor and LLM is available, fall back
        if active_llm_callable and toc_quality.score < 0.3:
            tree = generate_toc_from_text(text_content, len(self.doc), active_llm_callable)
            tree = enrich_with_table_counts(tree, self.doc)
            tree = enrich_with_continued_tables(tree, text_content)
            tree = enrich_with_footnote_markers(tree, text_content)
            tree = enrich_with_cross_references(tree, text_content)
            toc_quality = assess_toc_quality(tree, len(self.doc))

        # Step 7: Optionally add summaries (requires LLM)
        if active_llm_callable and (
            include_summaries or toc_quality.recommend_summaries
        ):
            add_summaries(tree, text_content, active_llm_callable)

        # Step 8: Write output files
        json_data = {
            "source": filename + ".pdf",
            "total_pages": len(self.doc),
            "preamble": preamble,
            "toc": [node.to_dict() for node in tree],
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

The `tools/registry.py` module exposes the concrete handoff point for a consuming
agent: `create_datasheet_tools_server(pdf_path)`. It creates a bound
`DatasheetTools` instance and registers `build_datasheet`, `get_toc`,
`get_page_text`, `search_text`, and `inspect_page` on a `ToolServer`.

```python
from claude_agent_sdk import Tool, ToolServer

def create_datasheet_tools_server(pdf_path: str):
    tools = DatasheetTools(pdf_path)

    server = ToolServer()
    server.register(
        Tool(
            name="inspect_page",
            description=(
                "Render a PDF page as a PNG image for visual inspection. "
                "Use when text extraction is insufficient (tables, figures, "
                "formulas). Returns base64-encoded image."
            ),
            parameters={
                "page": {
                    "type": "integer",
                    "description": "1-indexed page number to inspect",
                },
                "region": {
                    "type": "object",
                    "description": (
                        "Optional crop region with top/bottom/left/right "
                        "as percentages (0.0-1.0)"
                    ),
                    "properties": {
                        "top": {"type": "number"},
                        "bottom": {"type": "number"},
                        "left": {"type": "number"},
                        "right": {"type": "number"},
                    },
                },
                "detail": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": (
                        "Vision-token-cost tier. low=75 dpi, "
                        "medium=100 dpi (default), high=150 dpi."
                    ),
                },
                "dpi": {
                    "type": "integer",
                    "description": "Explicit override; wins over `detail`.",
                },
            },
            handler=lambda page, region=None, detail="medium", dpi=None: (
                tools.inspect_page(page, region=region, detail=detail, dpi=dpi)
            ),
        )
    )
    return server
```

This server object is the thing the consuming agent mounts. `claude-agent-sdk`
is intentionally optional so the core preprocessing path stays lightweight.

The consuming agent sets this up alongside the pre-processed artifacts:

```python
from datasheetindex import DatasheetIndex, create_datasheet_tools_server

artifacts = DatasheetIndex("datasheet.pdf").build(output_dir="output")
datasheet_tools_server = create_datasheet_tools_server("datasheet.pdf")

# Pseudocode: the exact agent wiring depends on the host runtime.
agent = SomeAgentRuntime(
    mcp_servers={"datasheet-tools": datasheet_tools_server},
    system_prompt=build_extraction_prompt(artifacts),
)
```

### Agent System Prompt Guidance

The system prompt below is **reference guidance for the consuming agent**, not part of this library. It is included here to document the intended usage pattern and inform tool design decisions:

```
You have access to a datasheet with:
1. A structured JSON map (enriched ToC) — sections, page ranges, table hints
2. A text file of the full document with page markers (--- PAGE N ---)
3. MCP tools that can build and read the artifacts, search the extracted text,
   and call `inspect_page` for visual inspection

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
- **No `has_figures`** — no library reliably detects vector diagrams; agent infers from section titles and text references
- **`breadcrumb`** — pre-computed full ancestry path per node (e.g. `"5 Electrical Characteristics > 5.1 Absolute Maximum Ratings"`), so downstream agents and RAG indexers see structural context without re-traversing parents
- **`boilerplate_category`** — title-pattern flag for `legal` / `ordering` / `revision` / `contact` / `toc` / `glossary` sections, so agents can deprioritize disclaimers, revision histories, and similar admin content. Title-only regex matching (no LLM, no text scan); children of flagged parents inherit the category
- **ToC quality assessment** — auto-detect whether summaries are worth generating
- **Page-matched text file** — PyMuPDF `get_text("blocks")` with column-aware reordering and page markers
- **Vision as primary escalation** — `inspect_page` for when text isn't sufficient
- **Agent tools** — `build_datasheet`, `get_toc`, `get_page_text`,
  `search_text`, and `inspect_page`, with text-first navigation, resilient
  search across wrapped/interleaved table text, and visual escalation when
  needed
- **Page alignment validation** — ensure JSON page numbers match text file markers
- **Zero extra dependencies** — only PyMuPDF needed for the happy path (no pymupdf4llm, no pdfplumber)
- **LLM as injectable callable** — the LLM fallback and summarizer accept a `llm_callable: (system, user) -> str` parameter rather than depending on a specific LLM client library. The consuming application provides its own LLM client (e.g., LiteLLM gateway with gpt-4.1 via the Responses API). This keeps the library dependency-free for the happy path while allowing LLM features when needed.

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
│   ├── preamble.py        # Pages 1-2 raw text for agent orientation
│   └── quality.py         # Page-level quality scoring
│                          #   (text density, extraction confidence)
├── tools/
│   ├── vision.py          # inspect_page (page → image for visual inspection)
│   └── registry.py        # Tool registration for Agent SDK / MCP
├── llm/
│   ├── toc_fallback.py    # PageIndex-style LLM ToC generation
│   └── summarizer.py      # Optional section summaries
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
