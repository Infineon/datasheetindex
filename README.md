<a href="https://www.infineon.com">
<img src="./assets/images/Logo.svg" align="right" alt="Infineon logo">
</a>

# datasheetindex

Agent-first parameter extraction from technical datasheets.

## Contents

- [What it does](#what-it-does) · [Philosophy](#philosophy) · [Supported products](#supported-products)
- [Benchmark](#benchmark) · [Links](#links)
- **Getting started** — [Setup](#setup) · [Development](#development) · [Input sources](#input-sources)
- **Using it from an agent** — [Hand the MCP server to an agent](#hand-the-mcp-server-to-an-agent) · [Realize the tools without the Claude Agent SDK](#realize-the-tools-without-the-claude-agent-sdk) · [Run a local MCP server](#run-a-local-mcp-server)
- **What the artifacts tell an agent** — [Datasheets without a ToC](#datasheets-without-a-toc) · [Knowing where the ToC came from](#knowing-where-the-toc-came-from) · [Asking for a better ToC](#asking-for-a-better-toc) · [Datasheets that cover a product family](#datasheets-that-cover-a-product-family) · [Figure indexing and captions](#figure-indexing-and-captions)
- **Reference** — [Python API](#python-api) · [CLI](#cli) · [Project structure](#project-structure) · [License](#license)

Design rationale — the measurements behind each decision, and the approaches
rejected on them — lives in
[`docs/datasheetindex_architecture.md`](./docs/datasheetindex_architecture.md).

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

About half of datasheets describe a family of parts, and their body text
describes the family: a features list or a peripheral section can name
something a given part does not have, while the per-part answer sits in the
selection or ordering table. An agent that misses this answers confidently and
wrongly -- observed, and reproducible 9 times out of 9 before this existed.

The library used to make it worse: `boilerplate_category: "ordering"` marks a
section as one to *deprioritize*, and on a family datasheet that is the one
section that can answer the question. Four things now address it:

1. **The `ordering` category is suppressed** when a family is detected, and
   only that category -- single-part datasheets are unchanged.
2. **`multi_variant`** appears in the ToC JSON and in `build_datasheet`'s
   manifest, present only when detected:

   ```json
   "multi_variant": { "family": "PSC3P5xD, PSC3M5xD", "rule": "wildcard" }
   ```

3. **`get_section_text` prepends a note** when the range may describe the
   family, naming the ordering section and suppressed inside it. It is phrased
   as an instruction ("Do NOT report a per-part answer from the text below ...
   Before answering, read *X*"), which is measured rather than stylistic: a
   descriptive phrasing scored 1/10 against this one's 10/10.
4. **A standing caution** in the `build_datasheet` and `get_section_text`
   descriptions, phrased for every datasheet.

Detection reads the **title block** only -- page 1's largest-font text plus the
PDF metadata title -- at a measured precision of 1.00 and recall of 0.85. So
**the absence of `multi_variant` is not evidence that a document covers a
single part**, which is why the standing caution is unconditional.

> Why the title rather than page-1 body text (precision falls to ~0.41), why
> this is not an LLM call, and why the exclusion list is kept deliberately
> short are in
> [the architecture doc](./docs/datasheetindex_architecture.md).

### Figure indexing and captions

Alongside `toc`, the ToC JSON carries a `figures` array with one entry per
raster image placement (`kind: "raster"` -- `region` in the
`inspect_page(region=...)` coordinate contract, plus `bbox`, `pixels`,
`page_area_pct`, and the `xref` two placements of one picture share) and one
per `Figure N` mention the text layer names (`kind: "caption"`, with a string
`figure_number`). The two kinds are never merged, even on the same page.
Three sibling keys disclose what is *not* there, and all are always present so
an empty result is distinguishable from an artifact predating the feature:

| key | says |
|---|---|
| `figures_excluded` | placements dropped as decorative, and the area threshold used |
| `figure_captions_excluded` | what `max_figure_captions` dropped |
| `figure_captions_blocked` | `true` only when *every* caption attempt was permanently rejected (bad certificate, refused credentials), so "no captions" can be told from "nothing to caption" |

**Captioning.** `build_datasheet` (and `DatasheetIndex.build()`, `build_batch`)
take `caption_figures: bool = True` and `max_figure_captions: int = 20`. With a
vision-capable client, each raster region above the area threshold gets a short
VLM description, largest first, up to the cap. Placements sharing an `xref` are
one picture: it is described once, every placement receives the answer, and the
cap counts pictures rather than placements. A caption names the kind of content
and then its most identifying labels -- a table's row labels and column
headings, a plot's axes -- in under 60 words, and never transcribes values.

**Each captioned figure is one VLM call**, so raising the cap raises cost
proportionally. Without credentials captioning is a no-op and the deterministic
`figures` array is unaffected; `caption_figures=False` or
`max_figure_captions=0` restores the pre-captioning artifact exactly.

`DATASHEETINDEX_VISION_MODEL` names a model for captioning alone -- unset, it
follows the model used for summaries and the ToC fallback -- which is worth
setting when your gateway serves a cheaper vision model. **Name a
non-reasoning model:** a reasoning model spends the whole output budget
thinking and returns an empty caption. Replies are capped at 300 output
tokens, and the prompt puts identifying labels first so truncation costs
description rather than identity.

**The manifest carries a bounded digest, not the array**, so an agent learns
that raster content exists without reading a file:

```json
"figures": {
  "total": 27, "raster": 20, "captioned": 7, "pages_with_figures": 12,
  "pages": [
    { "page": 3, "figures": 2, "caption": "Figure 1. Functional block diagram" },
    { "page": 9, "figures": 1, "caption": null }
  ],
  "truncated": false
}
```

`pages` lists one row per page holding figure entries, ascending, carrying that
page's **largest-area** caption. It is capped at 40 rows -- `pages_with_figures`
is the true count and `truncated` says whether rows were dropped -- so the
manifest cannot grow without bound. A row naming a page with no raster image
almost always means a **vector-drawn figure**, which cannot be enumerated for a
region: that page rewards a full-page `inspect_page`. Full detail, including
each region's coordinates, stays in the ToC JSON at `json_path`.

This is what lets an agent tell from the manifest alone that a page rendered
entirely as a picture is worth opening: `search_text` returning nothing there
proves nothing, because there is no text to search. `search_text` says so in
its own description, and attaches a `note` naming the digest and `inspect_page`
when a search comes back empty on a document that holds raster regions.

> Why images are sent at `detail: "high"` (measured: 19 of 20 row headings
> correct versus confabulation at `"low"`, for ~9x the input tokens), why the
> digest breaks ties on document order, and the rest of the measurements behind
> these defaults are in
> [the architecture doc](./docs/datasheetindex_architecture.md#figure-indexing).

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
