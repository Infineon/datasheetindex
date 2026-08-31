<a href="https://www.infineon.com">
<img src="./assets/images/Logo.svg" align="right" alt="Infineon logo">
</a>

# datasheetindex

Agent-first parameter extraction from technical datasheets.

## What it does

`datasheetindex` is meant to be handed to an external agent in two parts:

1. **Enriched ToC JSON** - Hierarchical section tree with page ranges, table hints, pre-computed breadcrumbs, boilerplate flags (revision history, disclaimers, etc.), a page-marked preamble (pages 1-2 raw text, with per-page signals in `preamble_pages`) for agent orientation, and a `figures` array indexing every raster image placement and text-layer figure caption (see "Figure indexing and captions" below)
2. **Page-matched text file** - Full document text with `--- PAGE N ---` markers aligned to the JSON, with column-aware reading order for two-column layouts (running headers and footers are omitted; set `DATASHEETINDEX_FURNITURE=0` to keep them)

All page numbers are **1-indexed** across the JSON, the text file markers, and
`inspect_page(page=...)`.

The library also exposes `create_datasheet_tools_server()`, which packages
artifact-building, ToC/text access, text search, `inspect_page`, and
`extract_table_markdown` as the MCP/tool-server surface the agent can mount. The server starts without a bound PDF; the agent loads one at runtime by
calling the `build_datasheet` tool with a `pdf_source`.

## Philosophy

The library doesn't extract parameters. The agent does. All intelligence lives in the agent; the library provides the best possible starting context and the right tools for edge cases.

## Supported products

`datasheetindex` is product-agnostic: it works with any PDF datasheet, including
the full [Infineon product portfolio](https://www.infineon.com/cms/en/product/)
(for example
[microcontrollers](https://www.infineon.com/cms/en/product/microcontroller/),
power, sensors, and connectivity devices). It has no dependency on a specific
product line or family.

## Benchmark

[`benchmark/`](./benchmark/) holds the chamber-grounded benchmark that
accompanies our EMNLP 2026 Industry Track paper
([arXiv:2608.28439](https://arxiv.org/abs/2608.28439)): the grading surface, the
archived model outputs behind every published number, and the analyses that
turn one into the other. It grades an extraction agent on two axes -- whether
it reported what the datasheet says, and whether that claim is physically true
as measured in a [Causal Chamber](https://causalchamber.org).

The two dispatch-level detector rules it ships are predicates over *this*
library's tool surface, which is why the benchmark lives here. They catch a
failure fidelity scoring cannot see: an agent that answers correctly without
ever opening the document.

It ships in two tiers. The first is fully offline: the grading surface, the
archive, and the analyses reproduce every published number with no model
client installed and no credentials. The second is the live agent harness
that produced the archive -- all three model arms, driven through a gateway
you supply, with a reference configuration and a per-artifact manifest saying
which script wrote each archived file.

It is a separate project under this repository, not part of the library. A
plain `uv sync` at the root installs none of it, and the published wheel
contains none of it. See [`benchmark/README.md`](./benchmark/README.md).

## Links

- [Infineon Developer Community](https://community.infineon.com/) - forums and
  knowledge base
- [Infineon Developer Center](https://softwaretools.infineon.com/) - tools and
  software packages
- [How to contribute](./CONTRIBUTING.md)
- [Code of Conduct](./CODE_OF_CONDUCT.md)
- [Support](./SUPPORT.md)
- [License](./LICENSE)

## Setup

```bash
uv sync
uv run pre-commit install
```

Optional integrations:

```bash
# LLM fallback / summaries (`create_llm_client`, `--model`, and automatic
# low-quality ToC fallback when credentials are available)
uv sync --extra llm

# Local MCP server testing (`datasheetindex-mcp-server`)
uv sync --extra mcp

# MCP server handoff to a consuming agent (`create_datasheet_tools_server`)
uv pip install claude-agent-sdk
```

For LLM-backed ToC fallback, summaries and figure captions, configure
`LITELLM_BASE_URL` and `LITELLM_MASTER_KEY` (see `.env.example`). Two optional
variables name the models, since which ones a gateway serves is a property of
your deployment rather than of this library:

| variable | names | default |
|---|---|---|
| `DATASHEETINDEX_MODEL` | summaries and the ToC fallback | `gpt-4.1` |
| `DATASHEETINDEX_VISION_MODEL` | figure captions only | follows the text model |

`build_datasheet`'s `model` argument overrides `DATASHEETINDEX_MODEL` for one
call; the vision model is deployment-level only, and deliberately not a tool
argument -- an agent has no way to know what a given gateway serves.

TLS verification is **on** by default. `LITELLM_TLS_VERIFY=false` (also `0`,
`no`, `off`) turns it off, for a gateway whose certificate the local trust store
cannot accept -- an internal CA missing from the image, or a self-signed cert on
localhost. Adding that CA to the trust store is the better fix: `LITELLM_MASTER_KEY`
travels on this channel. A certificate that fails verification raises
`LlmTlsVerificationError` (exported from the package root) naming the gateway
and both remedies, rather than degrading to an index with no ToC entries. Figure
captioning is the exception: it still degrades, because the artifact is complete
without captions.

`claude-agent-sdk` is only required for the SDK-flavored handoff
(`create_datasheet_tools_server`). The tool logic itself is framework-neutral --
non-SDK hosts get it via `create_datasheet_tool_defs()` with no SDK import (see
["Realize the tools without the Claude Agent SDK"](#realize-the-tools-without-the-claude-agent-sdk)).
The `mcp` extra is only required if you want to run a local stdio/HTTP MCP
server from this repository.

## Development

```bash
uv run pytest              # run tests
uv run ruff check src/     # lint
uv run ruff format src/    # format
uv run ty check            # type check
```

The pre-commit pytest hook runs the fast subset only:
`uv run pytest -q -m "not integration and not real_pdf"`. Run
`uv run pytest` for the full suite, including real-PDF and integration tests.

## Input sources

`DatasheetIndex` and `DatasheetTools` accept either:
- a local PDF file path, or
- an `http(s)` URL pointing to a PDF datasheet.

## Hand the MCP server to an agent

```python
from datasheetindex import create_datasheet_tools_server

# The server starts WITHOUT a bound PDF and takes no arguments. The agent loads a
# document at runtime by calling the build_datasheet tool with a pdf_source (local
# path or URL) before using any other tool.
datasheet_tools_server = create_datasheet_tools_server()

# Pass datasheet_tools_server into your agent runtime's MCP server configuration.
# The exact wiring depends on the host agent framework; this server object is the
# concrete handoff point from datasheetindex to the agent.
agent = SomeAgentRuntime(
    mcp_servers={"datasheet-tools": datasheet_tools_server},
    system_prompt="...tell the agent to call build_datasheet first...",
)
```

## Realize the tools without the Claude Agent SDK

Hosts that are not on the Claude Agent SDK (pydantic-ai, plain function-calling
agents, custom MCP servers) can get the same tools as framework-neutral
definitions -- **no `claude-agent-sdk` import required** -- via
`create_datasheet_tool_defs()`:

```python
from datasheetindex import create_datasheet_tool_defs

# One call == one session. build_datasheet binds the document; the other tools
# read it. Each def is a frozen dataclass: name, description, input_schema
# (JSON Schema), and an async handler.
tool_defs = create_datasheet_tool_defs()

for d in tool_defs:
    register_with_your_host(
        name=d.name,
        description=d.description,
        input_schema=d.input_schema,
        # async (args: dict) -> {"content": [...], "is_error": bool}
        handler=d.handler,
    )
```

Text blocks are `{"type": "text", "text": ...}`. `inspect_page` is the only tool
returning an image block, and it spells the media type twice on purpose --
`{"type": "image", "data": ..., "mime_type": "image/png", "mimeType": "image/png"}`
-- so hosts reading either convention work unchanged. Read whichever you prefer;
both always carry the same value.

When a handler raises, the envelope is `is_error: True` with a single text block
reading `"TypeName: message"` (e.g. `"ValueError: Page 9999 out of range..."`).
Messages the handler writes itself, like `"pdf_source is required"`, are not
prefixed.

`create_datasheet_tools_server()` is a thin adapter over this factory -- it wraps
each def with the SDK `@tool` decorator -- so the two surfaces expose identical
tool names, descriptions, and schemas.

If you want direct Python access instead of an MCP server, use `DatasheetTools`
to build artifacts, search text, and call `inspect_page()` on the bound
instance.

```python
from datasheetindex import DatasheetTools

with DatasheetTools("datasheet.pdf") as tools:
    artifacts = tools.build_datasheet(output_dir="output")
    toc = artifacts.json_data["toc"]  # enriched ToC tree (with breadcrumbs)

    # Search one term, or several in a single call. Each match carries the
    # breadcrumb of the section that contains its page; list searches also tag
    # each match with the `pattern` that produced it.
    matches = tools.search_text("supply voltage")
    matches = tools.search_text(["V_DS max", "junction temperature"])

    # Read a page range; the text opens with a "=== Page X of N ===" header
    # (singular for one page, "=== Pages X-Y of N ===" for a range). A
    # "=== NOTE: ... ===" line follows if the range cuts content marked
    # continuing onto an adjacent page -- its absence is not a completeness
    # guarantee.
    section_text = tools.get_section_text(12, 13)

    image = tools.inspect_page(
        page=12,
        region={"top": 0.15, "bottom": 0.55, "left": 0.05, "right": 0.95},
    )

    # Map a located string to its bounding box(es) - returned as percentages
    # (to crop inspect_page) and PDF points (to annotate the PDF). Matching is
    # hybrid: verbatim search with a normalized fallback for dash/case/spacing
    # variation. A string occurring multiple times yields multiple results.
    hits = tools.locate_text("supply voltage", page=12)
    if hits:
        tight = tools.inspect_page(page=12, region=hits[0]["region"]["pct"])
```

The optional `region` crop uses percentages from `0.0` to `1.0`.

## Run a local MCP server

You can run the local MCP server directly from the repository. It exposes these
tools for the bound PDF source:

- `build_datasheet` - build and save the `.json` / `.txt` artifacts, and return
  the manifest: source info, total pages, ToC quality, the full enriched ToC,
  and a bounded `figures` digest naming the pages that carry figure entries
  (see "Figure indexing and captions"). For a PDF with no usable ToC the
  manifest also carries a `hint` telling the agent to navigate by `search_text`
  instead (see "Datasheets without a ToC"). When the datasheet covers a product
  family rather than one part, the manifest also carries `multi_variant` (see
  "Datasheets that cover a product family")
- `get_section_text` - return extracted text for a page range from the latest
  build, opening with a position header (`=== Page X of N ===` for one page,
  `=== Pages X-Y of N ===` for a range) followed by zero or more
  `=== NOTE: ... ===` lines -- when the range cuts content the publisher marked
  as continuing onto an adjacent page, and when the datasheet covers a product
  family, in which case the note names the family and points at the per-part
  ordering table; absence of a note is not a completeness guarantee
- `search_text` - find page-aware text snippets in the latest build (pass a
  single pattern or a list of patterns), even when labels wrap across lines or
  table values interrupt the phrase; each hit carries the section breadcrumb.
  Searches the text layer only: when it finds nothing in a document that has
  raster figures, the result carries a `note` pointing at the `figures` digest
  and `inspect_page` (see "Figure indexing and captions")
- `inspect_page` - render a page image when visual confirmation is needed
- `extract_table_markdown` - re-extract a page as layout-aware Markdown tables
  (the page's running header and footer are omitted, same as `get_section_text`
  and the page-matched text file)

Build once, then use `get_section_text`, `search_text`, `inspect_page`, and
`extract_table_markdown` together. `search_text` prefers exact matches, then
falls back to whitespace-normalized and ordered-token matching for line-wrapped
table rows.

`locate_text` is **not** among the tools. It stays a supported Python API for
coordinate grounding (see above), but an agent has no reason to call it: the box
it returns covers well under 1% of a page, so cropping `inspect_page` to it
renders a picture of the query string. An agent that wants a closer look is
better served by `inspect_page(page, detail="low")` to see the layout, then
cropping to what it observed.

The server takes no PDF argument; it starts unbound and the client loads a
document at runtime by calling `build_datasheet` with a `pdf_source`.

```bash
# stdio transport (for Claude Code or another MCP client)
uv run --extra mcp datasheetindex mcp

# then call build_datasheet(pdf_source="datasheet.pdf", output_dir="output")
# from the MCP client
```

`datasheetindex mcp` and the `datasheetindex-mcp-server` console script are two
doors to the same server and take the same options; use whichever suits you.
The subcommand exists so a package-based MCP registry entry can name a real
distribution and a real command with the same string.

You can also expose it over HTTP:

```bash
# streamable HTTP transport (useful with MCP Inspector)
uv run --extra mcp datasheetindex mcp \
  --transport streamable-http --port 8000
```

With `streamable-http`, the default MCP endpoint is
`http://127.0.0.1:8000/mcp`.

Every tool returns its result as a single JSON string in a `TextContent` block
(except `inspect_page`, which returns an image); read it with
`json.loads(result.content[0].text)`. This matches the SDK tool surface exactly.

This local server is for direct MCP testing. If you need an in-process SDK
server object inside another Python runtime, use
`create_datasheet_tools_server()` instead; it exposes the same tool surface, with
the document loaded at runtime via the `build_datasheet` tool.

### Datasheets without a ToC

The enriched ToC comes from the PDF's own bookmarks, and plenty of real
datasheets ship without any -- two of the seven in our eval corpus have none,
and no printed contents page either. For those, `build_datasheet` returns
`"toc": []` with `"score": 0.0` unless the optional `[llm]` extra is installed
*and* gateway credentials are configured, which lets it reconstruct a ToC from
the page text.

Without that fallback the server still works. Only `build_datasheet`'s ToC
output degrades: `get_section_text`, `search_text`, `inspect_page`, and
`extract_table_markdown` address the document by page and raw text, and never
consult the ToC. The agent loses the section map, not the
document -- it navigates by searching instead of by planning a route, which is
how a person reads a datasheet with no bookmarks.

To make that explicit rather than leaving the agent to infer it, the manifest
carries a `hint` field whenever the returned ToC is empty:

```json
{
  "source": "current_sensor.pdf",
  "total_pages": 26,
  "toc_quality": { "score": 0.0, "entry_count": 0, ... },
  "toc_source": "none",
  "toc": [],
  "figures": { "total": 0, "raster": 0, "captioned": 0, "pages_with_figures": 0, "pages": [], "truncated": false },
  "hint": "This PDF has no usable table of contents, so there is no section map to plan from. Orient by reading pages 1-2 with get_section_text, then locate content with search_text and read around each hit with get_section_text. inspect_page renders a page as an image when the extracted text is unclear."
}
```

A document with a usable ToC has no `hint` key.

### Knowing where the ToC came from

`toc_source` sits beside `toc` in both the ToC JSON and the manifest, and is
one of:

| value | meaning |
|---|---|
| `pdf_outline` | The document's own PDF bookmarks. Page numbers are exact. |
| `llm_reconstructed` | The bookmarks were missing or too weak, and the ToC was rewritten from the body text. Every `start_page` is the model's inference from a `--- PAGE N ---` marker, so confirm a section with `search_text` before reading a range from it. |
| `none` | No ToC at all -- this is the case the `hint` above accompanies. |

`toc_quality` deliberately does not answer this. It scores the tree that came
out, not who wrote it, and a reconstruction that scores well is
indistinguishable from a good outline by score alone. The two calls for
different handling of the same `start_page`, so they are reported separately.

A rejected fallback candidate leaves the PDF's own outline in place, and
`toc_source` reports `pdf_outline` -- it names the tree you were given, not
whether the fallback ran.

### Asking for a better ToC

The quality score decides one thing only: whether to spend an LLM call
rebuilding the ToC. It is deterministic and cheap, so it can be wrong -- an
outline of `Page 1`, `Page 2`, ... covers the document perfectly while naming no
section at all. Since the manifest hands you the whole `toc`, you can see that
for yourself, and `build_datasheet(regenerate_toc=true)` acts on it: the ToC is
rewritten from the body text regardless of the score. `force_rebuild` is not the
same lever -- it re-runs the same deterministic scoring and reaches the same
decision.

It needs both the `[llm]` extra and gateway credentials, and raises if either is
missing rather than quietly returning the ToC you asked to replace. It is part
of the artifact-reuse key, so a normal build's artifacts are never *served* for a
regeneration request or the reverse. They do not coexist: there is one
`doc.json`, one `doc.txt` and one `doc.build.json` per document, so alternating
between the two rebuilds fully each time and overwrites the other's output.

Escalating is not a guarantee. The regenerated ToC is only a candidate, and the
page-coverage guard still rejects it if it covers less of the document than the
outline it would replace -- which is the tightest remaining barrier for the very
case this exists for, since a `Page 1..N` outline has `page_coverage == 1.0` and
any candidate must match it. A rejected candidate returns the same ToC with no
explanation attached; the way to detect it is `toc_source`, which stays
`pdf_outline` instead of becoming `llm_reconstructed`.

### Datasheets that cover a product family

About half of datasheets describe a family of parts rather than one part, and
their body text describes the family. A features list, a block diagram or a
peripheral section can name something a given part in the family does not have,
while the per-part answer sits in the selection or ordering table. An agent that
does not notice this answers confidently and wrongly -- observed: asked whether
one part carried a CORDIC accelerator, an agent answered "yes" from a
family-level features section, when the ordering table said "No" for every
variant of that part.

The library used to make this worse. `boilerplate_category: "ordering"` marks a
section as one to *deprioritize*, and on a family datasheet that is the one
section that can answer the question. Four things now address it:

1. The `ordering` category is **suppressed** when a family is detected -- and
   only that category. Single-part datasheets are unchanged, since ordering
   really is boilerplate there.
2. `multi_variant` appears in the ToC JSON and in `build_datasheet`'s manifest,
   naming the family and the rule that matched. It is present only when
   detected, so a single-part build pays nothing for it:

   ```json
   "multi_variant": { "family": "PSC3P5xD, PSC3M5xD", "rule": "wildcard" }
   ```

3. `get_section_text` prepends a note when the range may describe the family:

   ```
   === NOTE: this datasheet covers a product family (PSC3P5xD, PSC3M5xD). Text
   in this range may describe the family rather than the specific part you were
   asked about. Per-part differences are tabulated in "8 Ordering information"
   (page 69). ===
   ```

   It is suppressed inside the ordering section, where it would point at the
   page being read. The read-time note matters independently of the manifest
   field: in the observed failure the agent already held the ToC and the
   ordering section, and went wrong many turns later while reading a section.
4. `build_datasheet` and `get_section_text` carry a standing caution about
   per-part questions, phrased for every datasheet.

Detection reads the **title block** only -- page 1's largest-font text plus the
PDF metadata title -- and is deliberately conservative: measured precision 1.00
and recall 0.85 against hand-labelled ground truth over a 25-document, 6-vendor
corpus. A false positive costs some noise; a false negative costs a confident
wrong answer, so **the absence of `multi_variant` is not evidence that a
document covers a single part.** That is why the standing caution in the tool
descriptions is unconditional.

Page-1 *body* text was measured as an alternative and rejected: it drops
precision to about 0.41 while recovering two real families out of 57 misses,
because package order codes, companion parts, and tokens that are not part
numbers at all (`RGB888`, `PT100`, pin names) look exactly like feature
variants without vendor-specific grammars. The title works because a vendor
curates it -- writing `ADS111x` or `LM111, LM211, LM311` precisely when the
document covers a family.

### Figure indexing and captions

Alongside `toc`, the ToC JSON carries `figures` -- a page-then-position list
of every raster image placement (`kind: "raster"`, with `region` normalized
to the `inspect_page(region=...)` coordinate contract, `bbox` in raw PDF
points, `pixels`, `page_area_pct`, and `xref` -- the image XObject drawn,
which two placements of the same picture share) and every `Figure N` / `Fig. N`
caption the text layer names (`kind: "caption"`, with a string
`figure_number` -- `"10-1"` as readily as `"12"`). The two kinds are reported
as separate entries, never merged, even when a raster region and a caption
share a page. `figures_excluded` reports `{"below_min_area_pct": ...,
"min_area_pct": ...}` for placements dropped as decorative (a logo repeated
across every page). Both keys are always present -- `figures: []` on a
document with none -- so an empty result is distinguishable from an artifact
built before this feature existed.

`build_datasheet` (and `DatasheetIndex.build()`, `build_batch`) also take
`caption_figures: bool = True` and `max_figure_captions: int = 20`. When a
vision-capable client is available -- supplied explicitly, or self-created
the same way the ToC fallback is -- every raster region above the area
threshold gets a short VLM description (`caption_source: "llm"`), largest
regions first, up to the cap; `figure_captions_excluded` discloses what the
cap dropped, and `figure_captions_blocked` is true when *every*
attempted caption was rejected for a reason that will not change on retry --
a certificate that did not verify, or credentials the gateway refuses -- so
"no captions" can be told apart from "nothing to caption". One rejection
among successes is a blip and does not set it. Placements sharing an `xref` are one picture: the largest is
described, every placement receives the answer, and the cap counts pictures,
not placements. A header logo repeated on four pages is therefore one call
with four identical captions, not four calls -- measured at 33% of the calls
on a seven-document PCN corpus and 10% across nineteen documents. The caption names the kind of content (table, schematic, plot,
photo, block diagram, pinout) and then, immediately, its most identifying
labels -- for a table, its row labels first and then its column headings; for
a plot, its axes and plotted quantity -- under 60 words, and it never
transcribes cell values or numbers. This is what lets an agent tell, from the
manifest alone, that a page rendered entirely as a picture (no text layer at
all) is worth opening with `inspect_page`: `search_text` returning zero hits
on that page proves nothing, since there is no text there to search. That
inference is easy to miss pages later, so `search_text` states the limitation
in its own description and, when a search comes back empty on a document that
holds raster regions, attaches a `note` naming the digest and `inspect_page`
as the next step. Only then -- a hit-free search over a document with nothing
but text-layer figure captions has nothing hidden from it, and a note on a
search that succeeded is noise.
**Each captioned figure is one VLM call**, so raising the cap raises cost
proportionally. Without credentials configured, captioning is a no-op and the
deterministic `figures` array is unaffected either way. `caption_figures=False`
or `max_figure_captions=0` turns it off explicitly and restores the
pre-captioning artifact exactly.

Two knobs bound the cost of each call. `DATASHEETINDEX_VISION_MODEL` names the
model used for captioning alone -- unset, it follows the model used for
summaries and the ToC fallback -- which is worth setting when your gateway
serves a cheaper vision model, since captioning is the only per-figure cost in
a build. Name a **non-reasoning** model: a reasoning model spends the whole
output budget thinking and returns an empty caption. Each reply is capped at
300 output tokens, above every compliant answer measured (the prompt asks for
under 60 words, which lands near 90 tokens) and there to bound a model that
answers a 128-pin pinout by listing all 128 pins. Truncation is anticipated by
the prompt, which puts identifying labels before any description of structure.

Images are sent to the vision model at `detail: "high"`, not the API's
cheaper `"low"` (512x512-downscale) tier. Measured on a product-change-notice
datasheet whose "Product Attributes" table is rendered entirely as a raster
image: at `"low"` the model confidently invented several row headings that do
not exist in the table; at `"high"` it returned 19 of 20 row headings
verbatim correct, including the two rows naming a supplier. Cost, read from
`usage` on live responses: 120 input tokens per image at `"low"` versus 1074
at `"high"` -- about 9x, or roughly 2.4k to 21.5k input tokens per document at
the default cap of 20, paid once per document and then cached on disk by the
existing artifact reuse.

A caller who supplies, or has credentials for, a vision-capable client and asks
for summaries gets both. A **keyless** build runs no summaries: with
`llm_callable=None`, `include_summaries=True` produces summaries only when the
weak-ToC fallback needed a client of its own, so a document's figures never
switch summaries on by themselves. Captioning is unaffected -- it is the one
branch that self-creates a client for its own sake.

`build_datasheet`'s manifest does not repeat the `figures` array; it carries a
bounded **digest** of it, so an agent learns that raster content exists without
reading a file:

```json
"figures": {
  "total": 27,
  "raster": 20,
  "captioned": 7,
  "pages_with_figures": 12,
  "pages": [
    { "page": 3, "figures": 2, "caption": "Figure 1. Functional block diagram" },
    { "page": 9, "figures": 1, "caption": null }
  ],
  "truncated": false
}
```

`total` counts the entries in the ToC JSON's `figures` array that carry a
usable integer `page`, `raster` the `kind: "raster"` ones, and `captioned`
those carrying a caption from either source. `pages` lists one row per page
carrying figure entries, in ascending page order, with that page's entry count
and its **largest-area** caption (by `page_area_pct`, clipped to 350
characters), not merely its first in document order -- a small figure listed
ahead of a larger one in the ToC JSON must not shadow it in the digest. Ties
break on document order, never on dict or set iteration, so the digest is
byte-stable across runs. Because a `"caption"` entry
is created for any `Figure N` mention in the page text, a row can name a page
holding no raster image at all. Measured across a 14-document corpus, that
overwhelmingly means the figure is **drawn as vector art** -- which
`get_image_info()` cannot enumerate, so the index names the figure without being
able to offer a region for it. Such a page rewards a full-page `inspect_page`;
it is a signal, not noise. It is
capped at 40 rows -- `pages_with_figures` is the true count
and `truncated` says whether rows were dropped -- so the manifest's size does
not grow with a pathological document's figure count. The key is always
present, so `"total": 0` is distinguishable from an artifact predating the
figure index. Full detail, including each region's coordinates for
`inspect_page(region=...)`, stays in the ToC JSON at `json_path`.

## Python API

```python
from datasheetindex import DatasheetIndex, build_batch

artifacts = DatasheetIndex("datasheet.pdf").build(output_dir="output")

batch_result = build_batch(
    ["part-a.pdf", "part-b.pdf"],
    output_dir="batch-output",
)
```

In batch mode, output filenames are suffixed as needed to keep them unique when
multiple inputs would otherwise resolve to the same stem.

## CLI

```bash
# Local file
datasheetindex build path/to/datasheet.pdf --output-dir output

# Remote URL
datasheetindex build https://example.com/datasheet.pdf --output-dir output

# With explicit LLM model for ToC fallback and summaries
datasheetindex build datasheet.pdf --model gpt-4.1 --include-summaries

# Skip figure captioning, or raise its per-document cap (default 20)
datasheetindex build datasheet.pdf --no-caption-figures
datasheetindex build datasheet.pdf --max-figure-captions 40

# Run the local MCP server (stdio by default; needs the [mcp] extra)
datasheetindex mcp
```

By default, `datasheetindex` first uses native PDF ToC extraction. If ToC
quality is low, it automatically attempts LLM fallback with the default model
(`gpt-4.1`) when LLM credentials are available. Pass `--model` to choose the
LLM model explicitly; `--include-summaries` requires `--model`. Figure
captioning (see "Figure indexing and captions" above) runs by default under
the same credential rule and needs no `--model` of its own; `--no-caption-figures`
turns it off.

`datasheetindex build` needs no optional extras. `datasheetindex mcp` needs the
`[mcp]` extra and reports a single-line install hint if it is missing.

## Project structure

```
src/datasheetindex/
    core/
        structure.py       # ToC extraction + enriched tree JSON
        textfile.py        # PDF -> page-matched text file (column-aware)
        _textmatch.py      # Shared dash/token normalization + matcher
        locate.py          # locate_text: text -> bounding-box coordinates
        figures.py         # raster_regions: exact raster placements, clipped
                            #   to the page and normalized for inspect_page
        preamble.py        # Page-marked front matter + per-page signals
        variants.py        # Does this PDF cover a product family?
        quality.py         # ToC quality assessment
        annotations.py     # Footnote and cross-reference enrichment
        boilerplate.py     # Title-pattern boilerplate classification
    tools/
        vision.py          # inspect_page (page -> image)
        bound.py           # DatasheetTools (document-bound tool logic)
        defs.py            # create_datasheet_tool_defs (framework-neutral defs)
        registry.py        # Claude Agent SDK adapter (re-exports DatasheetTools)
    mcp_server.py          # Local stdio/HTTP MCP server entry point
    llm/
        client.py          # LLM client factory (free-text + structured_json + vision)
        toc_fallback.py    # LLM-based ToC generation fallback
        summarizer.py      # Optional section summaries
        figure_captions.py # VLM captioning for raster figure regions
    cli.py                 # CLI entry point
    index.py               # Main DatasheetIndex class
    models.py              # Data models
```

## License

Licensed under the [MIT License](./LICENSE).

Copyright (c) 2026 Infineon Technologies AG
