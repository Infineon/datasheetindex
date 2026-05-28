<a href="https://www.infineon.com">
<img src="./assets/images/Logo.svg" align="right" alt="Infineon logo">
</a>

# datasheetindex

Agent-first parameter extraction from technical datasheets.

## What it does

`datasheetindex` is meant to be handed to an external agent in two parts:

1. **Enriched ToC JSON** - Hierarchical section tree with page ranges, table hints, pre-computed breadcrumbs, boilerplate flags (revision history, disclaimers, etc.), and a preamble (pages 1-2 raw text) for agent orientation
2. **Page-matched text file** - Full document text with `--- PAGE N ---` markers aligned to the JSON, with column-aware reading order for two-column layouts

All page numbers are **1-indexed** across the JSON, the text file markers, and
`inspect_page(page=...)`.

The library also exposes `create_datasheet_tools_server(pdf_path)`, which packages
artifact-building, ToC/text access, text search, and `inspect_page` as the
MCP/tool-server surface the agent can mount.

## Philosophy

The library doesn't extract parameters. The agent does. All intelligence lives in the agent; the library provides the best possible starting context and the right tools for edge cases.

## Supported products

`datasheetindex` is product-agnostic: it works with any PDF datasheet, including
the full [Infineon product portfolio](https://www.infineon.com/cms/en/product/)
(for example
[microcontrollers](https://www.infineon.com/cms/en/product/microcontroller/),
power, sensors, and connectivity devices). It has no dependency on a specific
product line or family.

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

For LLM-backed ToC fallback and summaries, configure `LITELLM_BASE_URL` and
`LITELLM_MASTER_KEY` (see `.env.example`).

`claude-agent-sdk` is only required if you want the MCP/tool-server handoff.
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
from datasheetindex import DatasheetIndex, create_datasheet_tools_server

artifacts = DatasheetIndex("datasheet.pdf").build(output_dir="output")
datasheet_tools_server = create_datasheet_tools_server("datasheet.pdf")

# Pass datasheet_tools_server into your agent runtime's MCP server configuration.
# The exact wiring depends on the host agent framework; this server object is the
# concrete handoff point from datasheetindex to the agent.
agent = SomeAgentRuntime(
    mcp_servers={"datasheet-tools": datasheet_tools_server},
    system_prompt=build_prompt_from(artifacts),
)
```

If you want direct Python access instead of an MCP server, use `DatasheetTools`
to build artifacts, search text, and call `inspect_page()` on the bound
instance.

```python
from datasheetindex import DatasheetTools

with DatasheetTools("datasheet.pdf") as tools:
    tools.build_datasheet(output_dir="output")
    toc = tools.get_toc()
    matches = tools.search_text("supply voltage")
    page_text = tools.get_page_text(12)
    image = tools.inspect_page(
        page=12,
        region={"top": 0.15, "bottom": 0.55, "left": 0.05, "right": 0.95},
    )
```

The optional `region` crop uses percentages from `0.0` to `1.0`.

## Run a local MCP server

You can run the local MCP server directly from the repository. It exposes these
tools for the bound PDF source:

- `build_datasheet` - build and save the `.json` / `.txt` artifacts
- `get_toc` - return the enriched ToC JSON, including preamble and quality info
- `get_page_text` - return extracted text for one page from the latest build
- `search_text` - find page-aware text snippets in the latest build, even when
  labels wrap across lines or table values interrupt the phrase
- `inspect_page` - render a page image when visual confirmation is needed

Build once, then use `get_toc`, `get_page_text`, `search_text`, and
`inspect_page` together. `search_text` prefers exact matches, then falls back
to whitespace-normalized and ordered-token matching for line-wrapped table
rows.

```bash
# stdio transport (for Claude Code or another MCP client)
uv run --extra mcp datasheetindex-mcp-server datasheet.pdf

# then call build_datasheet(output_dir="output") from the MCP client
```

You can also expose it over HTTP:

```bash
# streamable HTTP transport (useful with MCP Inspector)
uv run --extra mcp datasheetindex-mcp-server datasheet.pdf \
  --transport streamable-http --port 8000
```

With `streamable-http`, the default MCP endpoint is
`http://127.0.0.1:8000/mcp`.

This local server is for direct MCP testing. If you need an in-process SDK
server object inside another Python runtime, use
`create_datasheet_tools_server(pdf_path)` instead; it exposes the same tool
surface for the bound PDF.

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
```

By default, `datasheetindex` first uses native PDF ToC extraction. If ToC
quality is low, it automatically attempts LLM fallback with the default model
(`gpt-4.1`) when LLM credentials are available. Pass `--model` to choose the
LLM model explicitly; `--include-summaries` requires `--model`.

## Project structure

```
src/datasheetindex/
    core/
        structure.py       # ToC extraction + enriched tree JSON
        textfile.py        # PDF -> page-matched text file (column-aware)
        preamble.py        # Pages 1-2 raw text extraction
        quality.py         # ToC quality assessment
        annotations.py     # Footnote and cross-reference enrichment
        boilerplate.py     # Title-pattern boilerplate classification
    tools/
        vision.py          # inspect_page (page -> image)
        registry.py        # MCP/tool-server factory for agent runtimes
    mcp_server.py          # Local stdio/HTTP MCP server entry point
    llm/
        client.py          # LLM client factory
        toc_fallback.py    # LLM-based ToC generation fallback
        summarizer.py      # Optional section summaries
    cli.py                 # CLI entry point
    index.py               # Main DatasheetIndex class
    models.py              # Data models
```

## License

Licensed under the [MIT License](./LICENSE).

Copyright (c) 2026 Infineon Technologies AG
