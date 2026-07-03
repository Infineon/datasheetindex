# Changelog

All notable changes to this project will be documented in this file.

## [0.17.0] - 2026-07-03

### Changed
- **The local MCP server now serves the framework-neutral tool defs -- single source of truth across all three surfaces.** `mcp_server.py` was rebuilt as a thin adapter over `create_datasheet_tool_session()` on a low-level `mcp` `Server` (one `list_tools` + one `call_tool` that translates the neutral `{"content": [...], "is_error": bool}` envelope into MCP content blocks), replacing its hand-maintained, drifted copies of the six tools' descriptions/handlers. The SDK server (`create_datasheet_tools_server`), the local MCP server, and non-SDK hosts now present **identical** tool names, descriptions, and JSON schemas; a change to a tool def propagates everywhere from one place. The stdio and streamable-http transports are exercised end-to-end by `integration`-marked tests with a real MCP client. Closes #5.
- **Wire-schema change on the local MCP server (behavioural, not a Python API break).** Its tool input schemas are now the canonical neutral JSON schemas (e.g. `oneOf` unions, `minimum`, per-field `description`) rather than FastMCP's signature-derived ones (`anyOf`/`title`/`$ref`). Property names and `required` are unchanged. Tool results are also now a single JSON-in-`TextContent` block (matching the SDK surface) rather than FastMCP's auto-derived `structuredContent`/`outputSchema`; a client that read `result.structuredContent` must `json.loads(result.content[0].text)` instead. The server is documented as "for direct MCP testing", so these only affect local clients.
- **`create_local_mcp_server()` now returns a `LocalMcpServer`** (with the same `run(transport=...)` method) instead of a `FastMCP` instance. Callers that only build the server and call `.run()` (the documented pattern) are unaffected.

### Added
- **`create_datasheet_tool_session()` + `DatasheetToolSession`.** A session bundles the neutral tool defs with a `close()` that releases the bound document -- fixing a latent gap where a document loaded from a URL left a temporary file behind until process exit. `create_datasheet_tool_defs()` is unchanged (now a thin wrapper over the session). Exported from `datasheetindex` and `datasheetindex.tools`.

## [0.16.1] - 2026-07-03

### Changed
- **Internal: `DatasheetTools` moved to a neutral `tools/bound.py` module.** The document-bound tool class now lives in its own leaf module that imports no agent-framework code, so the tool modules form a one-directional import graph (`registry -> defs -> bound`). Previously the framework-neutral `tools/defs.py` imported `DatasheetTools` from the SDK adapter `tools/registry.py`, which in turn imported the defs lazily inside a function purely to avoid the resulting import cycle; that workaround is gone. **No public API change** -- `DatasheetTools` is still importable from `datasheetindex`, `datasheetindex.tools`, and `datasheetindex.tools.registry` (verified by test). Closes #6.

## [0.16.0] - 2026-07-03

### Added
- **Framework-neutral tool-definitions factory.** New `create_datasheet_tool_defs()` (and the `DatasheetToolDef` frozen dataclass) realize the six datasheet tools -- `build_datasheet`, `get_section_text`, `search_text`, `inspect_page`, `locate_text`, `extract_table_markdown` -- as plain definitions (`name`, `description`, `input_schema`, async `handler`) **without importing `claude-agent-sdk`**. Hosts that are not on the Claude Agent SDK (pydantic-ai, plain function-calling agents, custom MCP servers) can wrap each `handler` directly; the `{"content": [...], "is_error": bool}` envelope already matches what most hosts expect. Per-session state (the `DatasheetTools` bound by `build_datasheet` and read by the other tools) lives in the factory's closure -- one call == one session -- exactly as before. Exported from both `datasheetindex` and `datasheetindex.tools`.

### Changed
- **`create_datasheet_tools_server()` is now a thin adapter over `create_datasheet_tool_defs()`.** It wraps each neutral def with the SDK `@tool` decorator and hands them to `create_sdk_mcp_server`, deleting ~260 lines of duplicated handler/metadata code. Tool names, descriptions, and JSON schemas are **identical** to before (locked by a parity test), so existing SDK consumers see zero behavior change; `claude-agent-sdk` becomes a wrapper-only dependency. Closes #3.
- **Tool handlers are now testable without the SDK.** Because the logic no longer requires `claude-agent-sdk` (which is not a declared dependency), `tests/test_defs.py` drives every tool handler end-to-end in CI -- including the "no datasheet loaded" guard, rebind-on-new-source, and per-factory-call session isolation.

### Fixed
- **A failed document switch no longer destroys the working session.** `build_datasheet` now builds a new `pdf_source` into a fresh `DatasheetTools` instance and only closes the previously bound document and rebinds *after* the build succeeds. Previously it closed the old document before validating the new source, so switching to a bad/unavailable path stranded every subsequent `get_section_text`/`search_text`/`locate_text` call with "No datasheet loaded" until a full re-build. (This latent bug also existed in the pre-refactor SDK-only handler.)

### Security
- Fixes GHSA (medium) in `pydantic-settings` (< 2.14.2): `NestedSecretsSettingsSource` followed symlinks outside `secrets_dir`, enabling local file reads and bypassing `secrets_dir_max_size`. Pulled in transitively via `mcp`; resolved by the dependency refresh below.

### Dependency upgrades
- Refreshed the lock to the latest compatible versions (`uv lock --upgrade`, all extras synced). Notably: `pydantic-settings` 2.14.1 -> 2.14.2 (security fix above), `mcp` 1.27.2 -> 1.28.1, `openai` 2.41.1 -> 2.44.0, `pymupdf` 1.27.2.3 -> 1.28.0 (and `pymupdf4llm`/`pymupdf-layout` to 1.28.0), `numpy` 2.4.6 -> 2.5.0, `onnxruntime` 1.26.0 -> 1.27.0, `ruff` 0.15.17 -> 0.15.20, `ty` 0.0.49 -> 0.0.56, `pytest` 9.1.0 -> 9.1.1, plus minor transitive bumps. Full suite, ruff, and ty all pass on the upgraded set.

## [0.15.0] - 2026-06-15

### Added
- **`locate_text` source grounding.** New tool that maps a query string to its bounding box(es) on a page, returning a result per match with `region` (a bounding rectangle) and `boxes` (one or more per-line rectangles; `region` is their union) in both normalized percentages and PDF points, plus page dimensions. Matching is hybrid: the verbatim `page.search_for` fast path returns one result per rectangle it finds, and a normalized word-level token fallback (dash/case/whitespace tolerant) groups a wrapped match's lines into a single result. The direct `DatasheetTools` Python API needs no `build_datasheet`; the Agent SDK and local MCP tool surfaces expose it once a document is loaded via `build_datasheet`.

### Changed
- **`inspect_page` on the local MCP server now raises a clean "No datasheet loaded" error** (via `_require_tools`) instead of an `AttributeError` when called before `build_datasheet`.
- **Shared text normalization extracted to `core/_textmatch.py`** (dash translation, token normalization, subsequence matcher), used by both `search_text` and `locate_text`. No behavior change to `search_text`.

### Dependency upgrades
- Refreshed the lock to the latest compatible versions (`uv lock --upgrade`, all extras synced). Notably: `openai` 2.40.0 -> 2.41.1, `pytest` 9.0.3 -> 9.1.0, `ruff` 0.15.15 -> 0.15.17, `ty` 0.0.42 -> 0.0.49, `cryptography` 48.0.0 -> 49.0.0, `typer` 0.26.5 -> 0.26.7, `starlette` 1.2.1 -> 1.3.1, `uvicorn` 0.48.0 -> 0.49.0, plus minor transitive bumps. Full suite, ruff, and ty all pass on the upgraded set.

## [0.14.0] - 2026-06-02

### Added
- **Multi-pattern `search_text`.** `query` now accepts a single string or a list of strings searched in one call. List searches tag each match with the `pattern` that produced it and dedupe by `(page, start, end)` (first pattern wins), with `max_results` as a global cap across patterns. The MCP and Agent SDK tool schemas accept a string or an array of strings. Single-string behavior is byte-identical to before.
- **ToC breadcrumb on every search hit.** Each `search_text` match is enriched with the `breadcrumb` of the section whose page range contains the match, so agents can disambiguate hits without a separate ToC lookup. Resolved via the new `find_breadcrumb_for_page()` over the typed `TocNode` tree, once per distinct page per call (deepest covering section wins; on equal-depth overlapping siblings the first in document order wins).
- **Position header on `get_section_text`.** Output now opens with a `=== Pages X-Y of N ===` (or `=== Page X of N ===` for a single page) header so the agent knows its position in the document and how much remains.

### Changed
- **`DatasheetArtifacts` retains the typed `TocNode` tree** (new `nodes` field). Breadcrumb resolution walks `TocNode` attributes instead of reaching into the serialized `json_data["toc"]` dicts, so the lookup is type-checked by `ty` and fails loudly on schema drift rather than silently dropping (or mis-attributing) breadcrumbs. The `.json` artifact is unchanged; the tree is in-memory only. New `TextSearchMatch` keys (`pattern`, `breadcrumb`) are `NotRequired`, so existing consumers are unaffected.
- **Tool descriptions adopt a "what vs. where" framing.** `search_text` is for when you know *what* to look for; `get_section_text` is for when you know *where* to read. Both the MCP and Agent SDK descriptions document the new multi-pattern, breadcrumb, and position-header behavior.
- **Documentation corrected.** The README and architecture doc referenced `get_toc`/`get_page_text` tools that do not exist; the tool surface (`build_datasheet`, `get_section_text`, `search_text`, `inspect_page`, `extract_table_markdown`) and the new behaviors are now documented accurately.

### Dependency upgrades
- Refreshed the lock with all extras (`llm`, `layout`, `mcp`) installed. `openai` 2.38.0 -> 2.40.0, `ty` 0.0.40 -> 0.0.42, `typer` 0.26.4 -> 0.26.5, `python-multipart` 0.0.29 -> 0.0.30, `virtualenv` 21.4.1 -> 21.4.2.

## [0.13.0] - 2026-05-15

### Added
- **`breadcrumb` field on every ToC node.** Pre-computed full ancestry path joined by `" > "` (e.g. `"5 Electrical Characteristics > 5.1 Absolute Maximum Ratings > 5.1.1 Junction Temperature"`). Computed in `assign_breadcrumbs()` and wired into `build_tree()`, so every code path -- including the LLM ToC fallback -- gets breadcrumbs without extra calls. Verified at 100% coverage on Infineon (66/66 nodes), TI (45/45), and NXP (108/108) live datasheets. Omitted from JSON when empty (e.g. legacy code constructing bare `TocNode` instances).
- **`boilerplate_category` field on ToC nodes.** New `core/boilerplate.py` module classifies titles into one of six categories -- `legal`, `ordering`, `revision`, `contact`, `toc`, `glossary` -- via title-only regex matching (no LLM, no text scanning). Title normalization strips leading section numbering (`12.3.4`, `Appendix A:`, `Chapter 3 `) and trailing punctuation before pattern matching, so `"12 Revision History"`, `"Appendix A: Ordering Information"`, and `"REVISION HISTORY"` all classify correctly. Substantive sections that merely mention boilerplate keywords (e.g. `"Trademark Licensing Strategy"`, `"Order of Operations"`, `"Glossary of Register Names"`) intentionally do not match. Bare ambiguous words (`"Information"`, `"Notice"`, `"Liability"`) require an explicit qualifier (`legal`, `important`, `product`) to match, since the bare forms are common substantive titles. Classification rule: a node's own classification wins; a node with no own classification inherits its parent's category (so an unlabelled "Rev 1.2 changes" subsection under "Revision History" inherits `revision`, but a substantive "Electrical Characteristics" subsection nested under a misclassified parent keeps its empty classification). English-only by design. Empty when no match -- omitted from JSON in that case.

### Changed
- `flag_boilerplate` is invoked inside `build_tree` alongside `assign_node_ids` and `assign_breadcrumbs`, so every code path that produces a tree -- happy path, LLM ToC fallback, future variants -- gets the classification for free without separate wiring.

### Dependency upgrades
- `uv-pre-commit` 0.11.13 -> 0.11.14, `ruff-pre-commit` v0.15.12 -> v0.15.13, plus minor transitive bumps (`ruff` 0.15.12 -> 0.15.13, `uvicorn` 0.46.0 -> 0.47.0, `sse-starlette` 3.4.3 -> 3.4.4, `idna` 3.14 -> 3.15, `virtualenv` 21.3.1 -> 21.3.3, `python-discovery` 1.3.0 -> 1.3.1).

## [0.12.0] - 2026-05-12

### Added
- **`detail` argument on `inspect_page`** - New `Literal["low", "medium", "high"]` parameter selects a vision-token-cost tier without the caller needing to know the underlying dpi. The tiers map to 75 / 100 / 150 dpi, producing roughly 650 / 1150 / 2580 input tokens per US-letter page on the Anthropic `(W*H)/750` formula. Available on `datasheetindex.tools.vision.inspect_page`, `DatasheetTools.inspect_page`, the `create_datasheet_tools_server` MCP tool, and the FastMCP `inspect_page_tool` handler. The MCP tool schema declares the enum so clients can validate before dispatch.

### Changed
- **Agent-surface wrappers default to `detail="medium"` (~1150 vision tokens/page) instead of 150 dpi.** Long agent loops -- the common case -- now pay roughly half the input-token cost per page without a visible drop in fidelity for body text and table cells. Callers that need footnote/subscript/dense-schematic resolution pass `detail="high"` (or `dpi=150` explicitly). The library primitive `datasheetindex.tools.vision.inspect_page` keeps its `detail="high"` default for backward compatibility with direct callers; only the agent-facing wrappers shifted.
- **`dpi` argument is now `int | None`** on every layer. Passing `dpi` still wins over `detail` when both are supplied (power-user override); passing neither falls back to the layer's `detail` default. `dpi=150` continues to produce identical bytes to the pre-0.12 default render (verified by `test_detail_high_matches_legacy_dpi_150`).

### Fixed
- **Invalid `detail` values are now rejected even when `dpi` is supplied.** Previously the membership check ran only inside the `if dpi is None:` branch, so a typo like `detail="hi", dpi=150` was silently accepted and would later erupt as a confusing `ValueError` the moment a caller dropped the `dpi` override. Validation is now unconditional. Test added: `tests/test_vision.py::test_unknown_detail_raises_even_with_explicit_dpi`.
- **`DatasheetTools.inspect_page` drops `# type: ignore[arg-type]`.** The wrapper now declares `detail: Detail` instead of `detail: str` and re-exports the `Detail` literal from `datasheetindex.tools.vision`, so static checkers see the constraint at the wrapper boundary.
- **`tests/test_cli.py::_CapturingIndex.build` matches `_FakeIndex.build`'s signature** so `ty check` reports a clean tree. (Pre-existing Liskov violation; surfaced while fixing the new diagnostics.)

### Dependency upgrades
- `uv-pre-commit` 0.11.8 -> 0.11.13.
- `ty` 0.0.34 -> 0.0.35, `mcp` 1.27.0 -> 1.27.1, `openai` 2.33.0 -> 2.36.0, `pydantic` 2.13.3 -> 2.13.4, `onnxruntime` 1.25.1 -> 1.26.0, `cryptography` 47.0.0 -> 48.0.0, plus minor transitive bumps.

## [0.11.1] - 2026-05-04

### Fixed
- **`output_dir` default is now safe in read-only container environments** - `DatasheetIndex.build()`, `DatasheetTools.build_datasheet`, and the MCP server entry points previously defaulted `output_dir` to `"output"` (a relative path), which fails on a read-only container root when no explicit path is provided. A new `resolve_default_output_dir()` function now handles the resolution: it honours `$DATASHEETINDEX_OUTPUT_DIR` if set, and falls back to `<tempdir>/datasheetindex-<uid>` (UID-namespaced for multi-tenant safety). The CLI and batch entry points still pass `"output"` explicitly to preserve the dev/interactive UX.

## [0.11.0] - 2026-04-09

### Added
- **Column-aware text extraction** - Text extraction now uses `page.get_text("blocks")` with block-level column detection instead of `page.get_text(sort=True)`. Two-column datasheet layouts (common in TI, NXP, STMicro) are read left column first, then right column, producing coherent prose instead of interleaved lines. Single-column datasheets are unaffected. Column detection uses conservative thresholds (block width, height, gutter gap, gutter consistency) to avoid false positives on table cells and diagram labels.
- **Column-aware preamble extraction** - The preamble (pages 1-2 raw text for agent orientation) now also uses column-aware extraction, so two-column cover pages produce readable output.

## [0.10.7] - 2026-04-01

### Fixed
- **`get_artifact_manifest` now returns the ToC list, not the full JSON blob** - The `toc` key was being set to `artifacts.json_data` (the entire document dict) instead of `artifacts.json_data.get("toc")`. Agents calling `get_artifact_manifest` now receive the correct hierarchical ToC list.
- **`extract_table_markdown` JSON Schema is now a valid MCP schema object** - The tool's parameter schema was a bare `{"page": int}` dict; it is now a proper JSON Schema object with `"type": "object"`, `"properties"`, and `"required"`, matching all other tools in the registry.
- **Temp file cleaned up on `KeyboardInterrupt` and other non-Exception signals** - `_download_pdf` now has a `BaseException` handler so the temp file is removed even if the caller interrupts the download mid-stream.

### Changed
- **`_iter_page_text` LRU cache reduced from 16 to 4** - Shrinks memory footprint; large text strings held in the cache are now evicted sooner.
- **`_has_env` fixture consolidated into `conftest.py`** - The duplicate fixture that was copy-pasted across `test_llm_client.py`, `test_summarizer.py`, and `test_toc_fallback.py` now lives in a single shared location. Unused `importlib` and `os` imports removed from the individual test files.
- **Test function names corrected** - Three test functions in `test_summarizer.py` were missing the underscore separator (`testextract_section_text` etc.) and were not collected by pytest; renamed to `test_extract_section_text` and variants.
- **Dependency upgrades** - `uv-pre-commit` bumped from 0.10.9 to 0.11.2; `ruff-pre-commit` bumped from v0.15.5 to v0.15.8.

## [0.10.6] - 2026-03-19

### Changed
- **MCP tool schemas use proper JSON Schema objects** - All four MCP tools (`build_datasheet`, `get_section_text`, `search_text`, `inspect_page`) now declare their parameters as JSON Schema objects with `"type": "object"`, `"properties"`, and `"required"` instead of bare Python-type dicts. This allows MCP clients to validate inputs before calling the tools.
- **Page parameters enforce `minimum: 1`** - All page-related integer parameters (`start_page`, `end_page`, `page`) now carry a `"minimum": 1` constraint in their schema, matching the documented 1-indexed page convention.
- **`search_text` schema marks only `query` as required** - The `page`, `case_sensitive`, and `max_results` parameters are now correctly treated as optional, with the description updated to note that omitting `page` searches all pages.

## [0.10.5] - 2026-03-13

### Changed
- **`build_datasheet` tool descriptions warn against unnecessary `include_summaries` use** - Both the MCP server and Agent SDK registry now explicitly advise agents to leave `include_summaries` as False unless the user explicitly requests it, since summaries make one LLM call per ToC section and are slow and expensive.
- **`build_datasheet` tool descriptions guide model selection** - Agents are now directed to use only models available on the LiteLLM gateway (gpt-4.1, gpt-5-mini, gpt-5-nano, gpt-4.1-nano, gpt-4o-mini, gpt-5, gpt-5.1, gpt-5.2) instead of inventing or guessing model names.

## [0.10.4] - 2026-03-12

### Fixed
- **Summaries no longer run when `include_summaries=False`** - The build logic was using `include_summaries or toc_quality.recommend_summaries`, which silently triggered LLM summarization even when the caller explicitly passed `include_summaries=False`. The condition is now `include_summaries` only, so summaries run only when explicitly requested.

### Changed
- **Build stage timing logs** - `DatasheetIndex.build()` now logs elapsed time at each major stage (PDF open, text extraction, table counting, ToC quality assessment, LLM fallback, LLM summaries, and total) at `INFO` level for performance diagnostics.

## [0.10.3] - 2026-03-12

### Fixed
- **Parallel table counting now active under MCP and Agent SDK** - Removed the `sys.stdout.isatty()` guard that was disabling multiprocessing whenever stdout was a pipe. Worker subprocesses already redirect their stdin/stdout to devnull via `_subprocess_init`, so parallel scanning was always safe. This restores the ~3x speedup on large PDFs when the library is called from an MCP server or Agent SDK host.
- **Increased LLM retry robustness** - Retry attempts raised from 3 to 5, base delay from 2s to 4s, and max delay from 30s to 60s to better absorb rate-limit bursts on long PDF builds.
- **Inter-call delay for LLM summarization** - A 0.5s pause is now inserted between consecutive section summarization requests to reduce the chance of hitting rate limits when summarizing datasheets with many sections.

## [0.10.0] - 2026-03-12

### Changed
- **`build_datasheet` now returns the full enriched ToC** - The artifact manifest returned by `build_datasheet` now includes the complete ToC JSON alongside source info, page count, and quality score. Agents no longer need a separate call to retrieve the ToC after building.
- **`get_toc` and `get_artifact_manifest` tools removed** - Both standalone tools have been retired. Their functionality is now covered by `build_datasheet` in a single call. The tool surface shrinks from 7 to 5 tools: `build_datasheet`, `get_section_text`, `search_text`, `inspect_page`, and `extract_table_markdown`.

## [0.9.0] - 2026-03-12

### Added
- **SSL fallback for PDF downloads** - URL downloads now retry without certificate verification when the server returns an `SSLCertVerificationError`. Covers semiconductor vendor sites (e.g. mxic.com.tw) that use self-signed or improperly chained certificates. Secure path is always tried first.
- **LLM retry with exponential backoff** - LLM API calls now retry automatically on 429 and 5xx responses, with delays of 2s and 4s (capped at 30s), before propagating the error.
- **Inter-chunk delay in ToC fallback** - A 1-second pause between consecutive LLM chunk calls prevents rate-limit bursting when generating a ToC from long PDFs.
- **pymupdf4llm ONNX model preloading** - The MCP server now loads the layout model at startup so the first `extract_table_markdown` call does not stall for ~2s during model initialization.

### Changed
- **`build_datasheet` and `extract_table_markdown` MCP tools are now async** - Both tools run their blocking work in a thread via `asyncio.to_thread`, preventing event loop stalls that could cause MCP client timeouts on large PDFs.
- **LLM ToC fallback failure is now non-fatal** - If the LLM fallback raises during `build()`, the original (possibly poor-quality) ToC is kept and a warning is logged instead of propagating the exception.

## [0.8.0] - 2026-03-11

### Added
- **`extract_table_markdown` tool** - Re-extracts a single page as layout-aware Markdown with proper `|`-delimited table rows using `pymupdf4llm`. Cheaper than `inspect_page` (text tokens vs vision tokens) and useful when `get_section_text` returns garbled whitespace-aligned tables. Available in both the Agent SDK server and the local MCP server.
- **`[layout]` optional extra** - New `pymupdf4llm>=1.27.0` dependency group that enables the `extract_table_markdown` tool. Install with `uv sync --extra layout`.
- **Parallel table counting** - `enrich_with_table_counts` now spawns a `ProcessPoolExecutor` to scan pages concurrently when a `pdf_path` is available. Cuts build time from ~30s to ~11s on a 122-page PDF. Falls back to sequential scanning automatically if multiprocessing fails or the path is unavailable.

### Changed
- **`create_local_mcp_server` no longer requires a PDF at startup** - The server starts without a bound document. Agents call `build_datasheet` with a `pdf_source` argument to load a document; calling it again with a different source switches documents without restarting the server.
- **`create_datasheet_tools_server` no longer requires a PDF at startup** - Same dynamic binding behaviour as the MCP server: pass `pdf_source` to `build_datasheet` at call time.
- **`build_datasheet` tool accepts `pdf_source` parameter** - Both the Agent SDK and MCP server variants of `build_datasheet` now take an explicit `pdf_source` (local path or URL) instead of reading it from a constructor argument.

## [0.7.0] - 2026-03-11

### Added
- **`get_artifact_manifest` tool** - Exposed as a standalone MCP tool and agent SDK tool so agents can check build status and document scope without re-reading the full ToC.
- **`get_section_text` replaces `get_page_text`** - Reads a page range (inclusive, 1-indexed) in one call and returns text with `--- PAGE N ---` markers for agent orientation. Accepts `start_page`/`end_page` matching ToC node fields directly.

### Changed
- **`create_datasheet_tools_server` rewritten to Agent SDK conventions** - Replaced the `Tool`/`ToolServer` pattern with `@tool` + `create_sdk_mcp_server`, matching the current Claude Agent SDK API. All six tools are now async and return structured `content`/`is_error` response envelopes.
- **Workflow-oriented tool descriptions** - All MCP and agent SDK tool descriptions updated to guide agents through the intended workflow: build first, plan with ToC, read sections, search, then inspect visually.
- **`DatasheetTools.get_page_text` removed** - Replaced by `get_section_text(start_page, end_page)` with stricter range validation and page-marker output.

## [0.6.0] - 2026-03-11

### Added
- **Local MCP server support** - Added the optional `mcp` extra and `datasheetindex-mcp-server` console command so a bound datasheet can be exposed over stdio or HTTP for direct MCP client testing.
- **Expanded MCP tool surface** - MCP integrations can now build datasheet artifacts, read the enriched ToC, fetch per-page text, search extracted text, and render pages for visual inspection.

### Changed
- **Artifact-aware tool workflows** - `DatasheetTools` now caches built artifacts and exposes `build_datasheet()`, `get_toc()`, `get_page_text()`, and `search_text()` for local and in-process MCP usage.
- **Faster broad text search** - `search_text()` now reuses cached page and token search structures with staged fallback logic so broad datasheet queries return much faster.

### Fixed
- **Wrapped and interleaved text matching** - Text search now keeps finding parameters when phrases wrap across lines or table values interrupt the label text.
- **Optional LLM test guards** - LLM-related tests now skip cleanly when optional client dependencies are not installed.

## [0.5.1] - 2026-03-10

### Added
- **Explicit test markers** - Added `real_pdf` and `integration` pytest markers so slower fixture-backed and external-integration tests are clearly labeled and easier to include or exclude.

### Changed
- **Faster pre-commit test selection** - The pre-commit pytest hook now skips slow tests by marker instead of by test name, making the fast local validation path more reliable.
- **CLI and architecture guidance** - Updated the README, architecture guide, and CLI help text to match the current auto-fallback LLM behavior, optional MCP handoff dependency, and batch output naming behavior.

### Fixed
- **Batch output collisions** - Batch builds now keep output filenames unique when multiple input PDFs would otherwise write the same stem.
- **LLM client cleanup** - Internally created LLM HTTP clients are now closed explicitly after use to avoid leaking owned resources.
- **Optional dependency handling in tests** - LLM-related tests and code paths now import optional dependencies more defensively so missing extras fail gracefully instead of during module import.

## [0.5.0] - 2026-03-06

### Added
- **MCP server handoff API** - Added public `create_datasheet_tools_server(...)`
  exports from both `datasheetindex` and `datasheetindex.tools` so consuming
  agents can mount the datasheet inspection server directly.

### Changed
- **Agent handoff documentation** - Updated the README and architecture guide to
  show the concrete MCP/tool-server wiring pattern using built artifacts and a
  bound datasheet tool server.
- **Tool server messaging** - Clarified that the registry factory returns the
  concrete MCP/tool server object and documented the optional
  `claude-agent-sdk` install path.

## [0.4.0] - 2026-03-03

### Added
- **LLM client runtime controls** - Added environment-driven timeout and retry settings
  (`LITELLM_TIMEOUT_SECONDS`, `LITELLM_MAX_RETRIES`) and explicit TLS verification
  control (`LITELLM_TLS_VERIFY`) for safer and more configurable endpoint usage.
- **URL body signature validation** - URL downloads now validate PDF file signatures
  before indexing, preventing HTML or non-PDF payloads from being treated as datasheets.

### Changed
- **Architecture docs key naming** - Updated architecture examples and pseudocode to use
  `source` and `toc` keys, matching the actual JSON output schema.
- **Pre-commit test selection** - Tightened pytest hook selection to exclude slower
  integration/real-PDF tests during pre-commit runs for faster local feedback.

### Fixed
- **Regression coverage for hardening paths** - Expanded test coverage for URL download
  validation and LLM client environment parsing to catch invalid timeout/retry settings
  and non-PDF payload handling.

## [0.3.0] - 2026-02-17

### Added
- **URL source support** - `DatasheetIndex` and `DatasheetTools` now accept `http(s)` URLs
  in addition to local file paths. PDFs are downloaded to a temp file and cleaned up on close.
- **CLI** - `datasheetindex build <path-or-url>` command with `--output-dir`, `--model`,
  and `--include-summaries` options. Registered as a console script entry point.
- **Auto LLM fallback** - When ToC quality is low and no `llm_callable` is provided,
  `build()` automatically attempts to create a default LLM client (`gpt-4.1`) if
  credentials are available. Falls back gracefully if they are not.
- **ToC entry validation** - `validate_toc_entry()` checks entry shape, level >= 1,
  and start_page >= 1 with clear error messages. Used by both native and LLM-fallback
  tree builders.
- **Region validation in `inspect_page`** - Rejects unknown keys, out-of-bounds values
  (must be 0.0-1.0), and inverted ranges (top >= bottom, left >= right).
- **Context manager protocol** - Both `DatasheetIndex` and `DatasheetTools` support
  `with` statements for automatic resource cleanup.
- **Download safety** - SSRF protection (validates final URL after redirects), 100 MB
  size limit, and temp file cleanup in all error paths.

### Fixed
- **End-page clamping** - Last child nodes in malformed ToCs no longer get
  `end_page < start_page` when the child starts after the parent's inferred end.
- **Missing re-enrichment after LLM fallback** - Continued tables, footnote markers,
  and cross-references are now applied to LLM-regenerated ToC nodes (previously only
  table counts were re-enriched).
- **Console script exit code** - The `datasheetindex` entry point now correctly
  propagates non-zero exit codes via `SystemExit`.

### Changed
- **`DatasheetTools` internals** - Now delegates to `DatasheetIndex` rather than
  managing its own `pymupdf.Document`, inheriting URL support and temp file handling.
- **`toc_fallback.py`** - Eliminated duplicated tree builder; now calls `build_tree()`
  from `structure.py` directly.
- **Public API naming** - Renamed `_validate_toc_entry`, `_compute_end_pages`, and
  `_assign_node_ids` to drop the underscore prefix since they are used across modules.
- **Exception handling** - `_try_create_default_llm_client` now catches only
  `(ImportError, ValueError, OSError)` instead of bare `Exception`.
- **Filename sanitization** - Output filenames are truncated to 200 characters to stay
  within OS path length limits.

## [0.2.0] - 2025-07-20

### Added
- Phase 4 refinement features: multi-page table detection (`continued_tables`),
  footnote marker detection (`footnote_markers`), cross-reference detection
  (`cross_references`), and batch processing.
- LLM fallback ToC generation for PDFs with missing or poor native ToC.
- Optional LLM-powered section summaries gated by ToC quality score.
- ToC quality assessment with scoring and summary recommendations.
- Preamble extraction (pages 1-2 raw text) for agent orientation.

## [0.1.0] - 2025-07-15

### Added
- Initial project structure with uv, pre-commit, and module scaffolding.
- Core ToC extraction from PyMuPDF `get_toc()` with hierarchical tree building.
- Page-matched text file generation with `--- PAGE N ---` markers.
- `inspect_page` tool for visual page rendering with region cropping.
- Table count enrichment per ToC node.
- `DatasheetTools` registry for Agent SDK integration.
