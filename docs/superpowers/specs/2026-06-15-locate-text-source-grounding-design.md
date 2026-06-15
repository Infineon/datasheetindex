# Design: `locate_text` — text-to-coordinate source grounding

- **Date:** 2026-06-15
- **Status:** Approved (review comments incorporated; pending re-review)
- **Scope:** `datasheetindex` library only

## Context

The "Baseline Enterprise RAG: From PDF to Highlighted Answer" article (Towards
Data Science) demonstrates a "highlighted answer" feature: every cited line is
mapped to its bounding box (`x0, y0, x1, y1`) so the source region can be drawn
on the rendered PDF. The article builds this from a persisted `line_df`.

`datasheetindex` has a structural gap that this idea exposes. The current tool
set can *find* text and can *render* a region, but nothing connects the two:

- `search_text` answers "your string is on page 22 at char offset 1400" —
  a character offset into the text artifact, not a coordinate.
- `inspect_page(region=...)` accepts a percentage region and renders it — but
  the caller must supply that region by eye.

There is no way to go from "this text" to "this region." `locate_text` is that
missing edge. It turns the existing tools into a composable chain:
**find (`search_text`) -> locate (`locate_text`) -> render (`inspect_page`)**.

Unlike the article, we do **not** persist a `line_df`/`word_df`. PyMuPDF can
resolve coordinates on demand (`page.search_for`, `page.get_text("words")`),
so the primitive stays stateless and adds no artifact and no dependency.

## Philosophy fit

The architecture doc defines what the library does NOT do: decide which
parameters to extract, decide when to escalate, implement extraction
strategies, validate values, manage prompts. That is the agent's intelligence.

`locate_text` makes none of those decisions. Given a string and a page, it
returns where the string sits. It is purely mechanical PDF geometry — the same
category as `inspect_page` (coords -> image) and `search_text` (text -> char
offset). It is PyMuPDF-only, reuses existing normalization, and is therefore
more cohesive inside the library than bolted on outside.

The boundary, mirroring how `inspect_page` is in-library but "deciding when to
use vision" is the agent's:

- **In scope (this library):** text -> coordinates. A fact about the PDF.
- **Out of scope:** value -> coordinates (stamping a bbox onto an extracted
  parameter result) is provenance about an extraction and belongs to the
  consuming agent (`datasheet-agent`). Rendering — drawing boxes, writing
  annotated PDFs, any UI — belongs to the consumer.

## Goals

- A stateless `locate_text` primitive that maps a query string (or list of
  strings) on a page to its bounding box(es).
- Robust against the transformations the text artifact applies (ASCII-dash
  normalization, whitespace collapsing, casing), so a string the agent took
  from `search_text`/`get_section_text` reliably resolves to coordinates.
- Coordinates returned in two forms: normalized percentages (round-trip into
  `inspect_page(region=...)`) and raw PDF points (PDF-native annotation), plus
  page dimensions.
- Exposed as a tool on both server surfaces (Agent SDK and local MCP),
  alongside the existing five.

## Non-goals (explicit, for this cut)

- No `datasheet-agent` changes. Carrying coordinates onto `ParameterResult`
  is a separate follow-up in that repository.
- No rendering. No highlighted images, no annotated PDF writer, no UI.
- No persisted `word_df`/`line_df` artifact. Resolution is on-demand.
- No new runtime dependency. PyMuPDF-only holds.

## Public API and data contract

```python
def locate_text(
    doc: pymupdf.Document,
    query: str | Sequence[str],
    *,
    page: int | None = None,
    max_results: int = 20,
) -> list[TextLocation]:
    ...
```

`TextLocation` is a `TypedDict`, in the style of `TextSearchMatch`:

```python
class _Box(TypedDict):
    pct: dict[str, float]      # {"top","bottom","left","right"}, each 0.0-1.0
    points: dict[str, float]   # {"x0","y0","x1","y1"}, PDF points

class TextLocation(TypedDict):
    page: int                  # 1-indexed
    match_method: str          # "search_for" | "tokens"
    page_width: float          # PDF points
    page_height: float         # PDF points
    region: _Box               # union of `boxes`; the inspect_page round-trip input
    boxes: list[_Box]          # >= 1; a multi-line match yields one box per line
    pattern: NotRequired[str]  # which query produced this hit (list queries only)
```

Notes:

- `boxes` is a list because that is the natural output of both match paths and
  it is honest about line wrapping; use it for precise per-line highlighting.
  `region` is the convenience union over `boxes` — `top=min`, `left=min`,
  `bottom=max`, `right=max`, in both `pct` and `points` — and is the single
  rectangle to feed the round-trip:
  `inspect_page(region=loc["region"]["pct"])`. For the common single-line match,
  `region == boxes[0]`.
- Percentages use the exact inverse of the `inspect_page` region math so the
  round-trip is pixel-consistent. For a box rect `(bx0, by0, bx1, by1)` in page
  coordinates and `rect = page.rect`:

  ```
  left   = (bx0 - rect.x0) / rect.width
  right  = (bx1 - rect.x0) / rect.width
  top    = (by0 - rect.y0) / rect.height
  bottom = (by1 - rect.y0) / rect.height
  ```

  `page_width = rect.width`, `page_height = rect.height`,
  `points = {x0: bx0, y0: by0, x1: bx1, y1: by1}`.
- Boxes always have positive area (both PyMuPDF paths return non-degenerate
  rects), so the percentages always satisfy `inspect_page`'s strict
  `top < bottom` / `left < right` requirement.

## Grounding semantics (v1 scope)

v1 grounds a **query string**, not a specific `search_text` hit. `locate_text`
returns *all* candidate boxes for the string on the page (up to `max_results`);
it takes no `start`/`end` or occurrence anchor. When the same string appears
multiple times on a page, it returns one `TextLocation` per occurrence and
leaves disambiguation to the caller — pass a longer, more specific query (e.g.
include surrounding context).

This is deliberate, not an oversight. Exact one-hit-to-one-region grounding
would require a char-offset <-> word <-> bbox bridge (plus the column-aware word
ordering noted under the matching algorithm), i.e. reconstructing the text
artifact's geometry. That is a materially larger feature; it is recorded as a
follow-up rather than built here. So the reader should not infer stronger
"exact highlighted answer" semantics than the API provides: it highlights *a
string on a page*, and is reliable precisely when the query is unique enough on
that page.

## Matching algorithm (hybrid)

Per page, for each query pattern:

1. **Fast path — `page.search_for(query)` with the original, verbatim query.**
   `search_for` matches against the raw PDF text, so the *unnormalized* query is
   correct here; passing the ASCII-translated string would discard exact matches
   and misclassify easy hits as `tokens`. It returns rects directly and covers
   the easy majority, including PyMuPDF's ligature and dehyphenation handling.
   These hits get `match_method == "search_for"`.
2. **Fallback — word-level matching (only when step 1 returns nothing).**
   Normalization applies *here only*. Normalize both the query and each word
   with the shared helpers: `_translate_search_text` plus
   `_normalize_token(..., case_sensitive=False)` (`locate_text` exposes no
   `case_sensitive` option, so the fallback is always case-insensitive).
   `page.get_text("words")` gives per-word `(x0, y0, x1, y1, word, block_no,
   line_no, word_no)`; match the normalized query tokens against the normalized
   word sequence:
   - single-token query: direct word-equality;
   - multi-token query: the `_match_query_tokens` subsequence search with a max
     gap (interleaving-tolerant).
   This catches `-0.3` vs Unicode-minus (`−`, U+2212), `±2%`, casing, and
   collapsed whitespace — the symbol/dash mismatch class endemic to datasheets
   and that the article documented as a retrieval failure. These hits get
   `match_method == "tokens"`.

**Occurrence grouping — one `TextLocation` per occurrence:**

- **Fast path:** each rect `search_for` returns is its own `TextLocation` (a
  single-box hit, so `region == boxes[0]`). `search_for` returns a flat,
  occurrence-unstructured list of rects, and clustering those into logical
  multi-line occurrences is ambiguous and PyMuPDF-version-dependent, so v1 does
  not attempt it. (In practice the fast path serves single-line atoms and
  values, where rect == occurrence.)
- **Token path:** the word fragments of a *single* subsequence match form one
  occurrence; group them into `boxes` keyed by `(block_no, line_no)` (PyMuPDF's
  `line_no` is block-scoped, not globally unique, so grouping by `line_no` alone
  would merge unrelated blocks sharing a line index), then set `region` to their
  union. A repeated phrase yields one `TextLocation` per subsequence match. This
  is the path that delivers a multi-box, unioned `region` for a wrapped phrase.

The fallback deliberately drops `search_text`'s "ignore queries shorter than 3
tokens" guard: for grounding we must locate single atoms like `VVS_max`.

**Native-geometry limitation.** Both paths operate on the PDF's *native* page
geometry, not on the column-reordered text artifact produced by
`_extract_page_text`. Within-column phrases and single atoms resolve reliably.
A "phrase" that is contiguous only because column-linearization joined the
bottom of the left column to the top of the right column is not guaranteed to
resolve as a single match — and such a span is not a real contiguous region to
highlight anyway. Callers that hit this should ground the within-column
sub-string. Aligning the fallback to the artifact's column order (so artifact
phrases round-trip faithfully) is a follow-up; see Out of scope.

This layered structure mirrors the existing fallback ladder in
`_search_single` (literal -> collapsed-whitespace -> token-sequence), so it is
stylistically native to the codebase.

## Module structure and shared-helper extraction

New module `src/datasheetindex/core/locate.py` holds `locate_text` and
`TextLocation`. It sits in `core/` (it returns data and reuses text
normalization), not in `tools/vision.py` (which renders images).

The matcher needs the same normalization `search_text` uses. Those helpers are
currently private to `textfile.py`. Reaching across modules into underscore
names is the coupling smell; instead, lift the shared pieces into a new
`src/datasheetindex/core/_textmatch.py` and import them from both modules:

- `_DASH_TRANSLATION`, `_translate_search_text`
- `_TOKEN_EDGE_PUNCTUATION`, `_normalize_token`
- `_TOKEN_RE` (tokenizing the query and the per-word stream)
- `_TokenSpan`, `_match_query_tokens`

`textfile.py`'s public and private behavior is unchanged; it re-imports the
moved names. This is a focused extraction in service of the feature, not an
unrelated refactor. (Alternative, not recommended: `locate.py` imports the
privates from `textfile.py` directly.)

`locate.py` tokenizes the query (and the word stream) with `_TOKEN_RE` +
`_normalize_token`, then drives matches through `_match_query_tokens` directly.
It must **not** reuse `_find_token_sequence_spans` as-is: that helper's
`len(query_tokens) < 3` guard would silently refuse single- and two-token
grounding targets such as `VVS_max` or `-0.3`. `_find_token_sequence_spans`
stays in `textfile.py`, unmoved.

Bypassing that helper, `locate.py` must still preserve the rest of its proven
behavior so the word path does not drift from the search ladder or regress on
large pages:

- pass `max_gap_tokens = max(8, len(query_tokens) * 2)` to `_match_query_tokens`;
- short-circuit to no-match when the page has fewer normalized tokens than the
  query;
- short-circuit when the query's token set is not a subset of the page's token
  set.

The only deliberate divergence from `_find_token_sequence_spans` is dropping the
`< 3` token guard.

## Tool wiring

The core lives in one module and is exposed on every consumer surface. This repo
has **two independent server registries**, and both must gain the tool or it is
silently missing on one:

- `core/locate.py` — `locate_text(doc, query, *, page=None, max_results=20)`.
- `DatasheetTools.locate_text(query, *, page=None, max_results=20)` in
  `registry.py`, delegating to core with the bound document. Like
  `inspect_page` (and unlike `search_text`/`get_section_text`), it reads
  `self.doc` and does **not** call `_require_artifacts()`: locate works on the
  live PDF, not the text artifact, so a pure-Python
  `DatasheetTools(pdf).locate_text(...)` succeeds with no prior
  `build_datasheet`. This asymmetry is intentional — do not add an artifact
  guard.
- **Agent SDK surface** (`tools/registry.py`): an `@tool("locate_text", ...)`
  handler using `_require().locate_text(...)` (mirrors `inspect_page`), added to
  the `create_datasheet_tools_server` `tools=[...]` list.
- **Local MCP surface** (`mcp_server.py`): a `locate_text_tool(query, page=...,
  max_results=..., ctx=...)` registered via `server.tool(name="locate_text",
  ...)` in `create_local_mcp_server`. It follows the `get_section_text_tool` /
  `search_text_tool` pattern — resolve the bound `DatasheetTools` via
  `_require_tools(ctx)` for a clean "no datasheet loaded" error. (Do **not**
  copy `inspect_page_tool`, which dereferences the lifespan context directly
  and does not guard.) Also extend the `instructions=` string in
  `create_local_mcp_server` to advertise `locate_text`; otherwise the surface
  exposes the tool without describing it in the server guidance.
- **In-scope cleanup (same file):** switch the existing `inspect_page_tool` to
  resolve its bound tools via `_require_tools(ctx).inspect_page(...)`. It
  currently dereferences `ctx.request_context.lifespan_context.tools` directly
  and raises an unhelpful `AttributeError` when no datasheet is loaded;
  `_require_tools` yields the same clean "no datasheet loaded" error as the
  other tools. Behavior-preserving otherwise.

On both server surfaces a document must already be bound, which happens only via
`build_datasheet(pdf_source=...)`; but `locate_text` itself never reads the
written text/JSON artifacts.

Tool description (both surfaces) points at both uses: feed `pct` back into
`inspect_page(region=...)` to crop to the exact region; use `points` to annotate
the PDF. It recommends passing `page` (the agent usually knows it from the
search hit or extraction provenance) so large documents stay cheap.

## Error handling and edge cases

Consistent with the existing tools:

- Not found -> `[]` (not an error), like `search_text`.
- `page` out of range -> `ValueError` (as `inspect_page` / `get_section_text`).
- Empty query (or all-empty list) -> `ValueError`.
- List query -> each match tagged with `pattern`; `max_results` is a global cap
  across patterns; first pattern wins on duplicates (as `search_text`). The
  dedup key is `(page, boxes_key)`, where `boxes_key` is the tuple of each box's
  points rounded to the nearest integer point — `(round(x0), round(y0),
  round(x1), round(y1))` — collected into a sorted tuple so box order does not
  affect identity. Integer-point rounding collapses sub-point jitter between the
  two match paths while keeping distinct occurrences (which differ by far more
  than 1 pt) separate.
- `page=None` -> scan all pages, capped by `max_results`. The tool description
  recommends passing `page` to bound cost on long documents.
- Multi-line match -> one box per line (see algorithm step 4).
- Rotated pages -> `search_for`/`words` return rects in page space and the pct
  math uses `page.rect`, the same basis as `inspect_page`. Consistent, but
  rotation is noted as a known edge case rather than special-cased in v1.
- Empty page / no words -> `[]`.

## Testing plan

A mix of a deterministic synthetic PDF (built in-test with PyMuPDF, following
`tests/test_vision.py::_make_test_doc`: `pymupdf.open()` + `new_page()` +
`TextWriter`, placing known strings at known coordinates) plus one real-fixture
smoke test. Real-fixture tests skip when the PDF is absent, following the
established `if not TLE9350_PATH.exists(): pytest.skip(...)` pattern.

Cases:

1. Exact `search_for` hit returns the correct rect; `match_method == "search_for"`.
2. **Dash-mismatch fallback:** a query with ASCII `-` locates Unicode-minus
   text and reports `match_method == "tokens"`.
3. Single-token symbol locate (e.g., `VVS_max`).
4. Multi-word phrase locate.
5. Not found -> `[]`.
6. `page` out of range -> `ValueError`; empty query -> `ValueError`.
7. **pct <-> points <-> page-dims consistency:** `pct.left * page_width`
   approximately equals `points.x0 - rect.x0`, etc.
8. List query: `pattern` tagging, dedup, and global `max_results` cap.
9. **Round-trip:** `locate_text` -> `inspect_page(region=loc["region"]["pct"])`
   renders without error and yields a smaller image than the full page.
10. Real-fixture smoke test (`TLE9350_PATH`): locate a known string, assert at
    least one box on the expected page.
11. **Both tool surfaces register `locate_text`.** Two tests assert an exact
    tool set and both must be updated to include `locate_text`:
    `tests/test_mcp_server.py` (`set(server.registered_tools) == {...}`) and
    `tests/test_registry.py::test_create_server_registers_tools` (which fakes
    `claude-agent-sdk` via `sys.modules` and asserts `set(server.tools) ==
    {...}`). Also add a direct `DatasheetTools.locate_text` test, and smoke-test
    that each surface's registered handler returns boxes for a known string.
12. **No-build behavior:** `DatasheetTools(pdf).locate_text(...)` returns boxes
    without a prior `build_datasheet` call — asserts the absence of an artifact
    guard.
13. **Occurrence grouping.** (a) A repeated single-line string yields one
    `TextLocation` per occurrence, each single-box. (b) A wrapped multi-word
    phrase resolved via the token path (forced through the fallback with a
    dash/symbol variant) yields one `TextLocation` whose `boxes` has length > 1
    (grouped by `(block_no, line_no)`) and whose `region` is their union
    (`region != boxes[0]`).
14. **`inspect_page_tool` guard (in-scope cleanup):** calling it with no
    datasheet loaded raises the clean `RuntimeError` ("No datasheet loaded..."),
    not `AttributeError`.

## Out of scope / follow-ups

- `datasheet-agent`: carry located coordinates onto `ParameterResult`
  (value -> coords provenance), and any rendering/UI. Separate work in that
  repo.
- **Exact one-hit grounding:** a `start`/`end` or `occurrence` anchor that maps
  a single `search_text` hit to a single region, plus column-aware word ordering
  so artifact phrases (including cross-column ones) round-trip faithfully. Both
  need a char-offset <-> word <-> bbox bridge over `_extract_page_text`'s
  geometry; deferred from v1 (see Grounding semantics).
- Possible later refinement: a persisted word index if profiling shows
  on-demand resolution is too slow on very large documents. Not needed for v1.
- CHANGELOG entry and version bump at implementation time.
