# Section reads keep their tail when the next heading is mid-page

Status: design, 2026-11-02. To be folded into `docs/datasheetindex_architecture.md`
and removed once shipped, as the header/footer and front-matter specs were.

## Why

`core/structure.py::compute_end_pages` ends every section on the page before its
next sibling starts. That is only right when the next heading is the first thing
on its page. When it sits partway down, the text above it belongs to the earlier
section, and reading that section's range leaves it out with no warning.

Reproduced on Bosch BME280 (`.superpowers/corpus/bosch_bme280.pdf`). Node
"1.3 Pressure sensor specification" is 10-10. Page 11 holds the rest of its spec
table above the "1.4 Temperature sensor specification" heading: absolute and
relative accuracy, resolution, noise, solder drift, long-term stability, sampling
rate. `get_section_text(10, 10)` returns none of them and emits no NOTE.

Corpus scale, measured against our own furniture-stripped text on 2026-11-02:
- 667 of 2,744 sibling boundaries (24%) have 150 or more characters above the
  next heading on its page.
- 639 of 1,849 later-page boundaries (35%) have the next heading 10 or more lines
  down.
- Worst on Bosch, Espressif, ATmega, PSoC and MSP430. The next heading was found
  in the page text at 2,744 of 2,750 boundaries.

The only safety net today is the `(continued)` NOTE, which needs the vendor to
print a marker.

## Intent

When the agent reads a section, it gets all of that section's content.
**Completeness over tightness:** a read may carry a few lines that belong
elsewhere, clearly marked, but it must never silently lose rows that belong to
it. That trade was the user's call.

## Why the fix is at read time (decided with the user)

The first draft of this spec extended `end_page` onto the shared page. Two
independent reviews (a fresh builder check, and Copilot CLI on gpt-6.1-sol) plus
a verification run showed that was the wrong layer:

- **Overlapping ranges have a large blast radius.** Search breadcrumbs, the
  multi-variant suppression check, evidence-lead de-duplication (which infers
  nesting from range containment), the summarizer and artifact persistence each
  needed their own rule.
- **It would regress the multi-variant note.** Read literally, the note would
  fire on the agent's full read of a comparison table whose last page became
  shared: MSP430 "6 Device Comparison" 9-9 would become 9-10.
- **Strict "the heading opens its page" fails on TI.** The TI part-list running
  header survives furniture stripping (verified on LM358: it is the first content
  of every page). About 44% of corpus boundaries would extend by a full page,
  almost all of them on TI.

Fixing the read appends only the text above the next heading: a few lines of
header residue on TI, the missing rows on Bosch. Nothing else changes.

## Goals

1. `get_section_text(start, end)` includes the text at the top of page `end + 1`
   that precedes the first section heading starting partway down that page.
2. Every section heading that starts partway down a page inside the returned text
   is preceded by a marker line, so the agent can tell where one section's rows
   end and the next begins. This covers the first page of a read, which may open
   with the previous section's tail.
3. A search hit above the first mid-page heading on its page gets the breadcrumb
   of the section that continues onto that page, not the one that starts there.

## Non-goals

- No change to the artifact: the ToC JSON (`end_page`, `table_count`,
  `has_tables`, annotations), the page-matched text file, the preamble, or the
  artifact-reuse key.
- No change to multi-variant suppression or lead selection; they keep reading
  `end_page` as today.
- Bookmark hygiene and title-on-page validation. Both were measured and rejected:
  0 non-monotonic outlines, and 3,967 of 4,022 titles found on their target page.

## Anti-goals: fail if any of these happens, even if it otherwise works

- A read returns less text than today.
- A read appends more than the head of the single page `end + 1`, except in the
  heading-not-found case below.
- Any NOTE that fires today stops firing, or vice versa.
- Markers or appended text leak into the cached text file or `text_content`:
  reads must not mutate artifacts, and repeated reads must be identical.

## Behaviour

### Locating headings (shared by all three goals)
- For every ToC node, locate its heading in the page text of its `start_page`.
  The page text is exactly what `extract_page_text` returns, the same string
  whose offsets `TextSearchMatch.start` uses.
- The result per node is one of:
  - **opens the page:** only whitespace precedes it;
  - **mid-page:** at offset `h > 0` in that same string, never a stripped or
    normalized copy;
  - **not found.**
- Derived from `text_content` and `nodes`, both already in memory on a cold build
  and a disk reuse alike, so nothing is persisted. Compute lazily once per loaded
  artifact and cache in memory.

### `get_section_text(start, end)`, with `end < total_pages`
Let the next page be `end + 1`, and consider the nodes whose `start_page` is
that page.

- **No such node:** append nothing. Either the section continues and the ranges
  already cover it, or the next page belongs to no new section.
- **Some node opens the next page:** append nothing.
- **The earliest located heading is mid-page at `h`:** append that page's text
  `[0:h)`, headed by a framing line that names the page and the heading it stops
  at. The NOTE's job is to keep the agent from treating the appended lines as a
  whole page.
- **Such nodes exist but none was located:** append the whole next page, with a
  framing line saying the next section's heading could not be located on it.
  This is completeness over tightness. It happened at 6 of 2,750 boundaries.

Framing lines use the `===` tool-framing convention, never `--- PAGE N ---`,
which must keep meaning a whole page. The header line (`=== Pages X-Y of N ===`)
keeps naming the requested range.

### Section-start markers
- Inside every page the read returns, the requested pages plus any appended
  head, insert `=== SECTION STARTS: <breadcrumb> ===` immediately before each
  mid-page heading at its exact offset.
- No marker for a heading that opens its page or was not found.
- The breadcrumb is used rather than the bare title because titles repeat.

### NOTEs
- The variant note and `_continuation_notes` are computed exactly as today, on
  `(start, end)`.
- `_continuation_notes`' tail check reads page `end`'s trailing text, which is
  unchanged by appending page `end + 1`'s head.

### Search breadcrumbs
- A hit on page `p` at `start < h_min(p)` gets the breadcrumb of the deepest node
  covering page `p - 1`. Here `h_min(p)` is the earliest mid-page heading offset
  among nodes starting on `p`.
- Every other hit keeps today's `find_breadcrumb_for_page(nodes, p)`.
- The per-page breadcrumb cache in `DatasheetTools.search_text` becomes keyed on
  `(page, before_first_heading)`, so two hits on either side of a heading in one
  search resolve independently.
- Ancestry is handled for free: if chapter A's last child A.1 ends on page 2 and
  chapter B starts mid-page on 3, a hit on page 3 above B's heading resolves to
  A.1. Today's tiling makes A.1 the deepest node covering page 2, and B's
  descendants start after B's heading.

## Decisions marked as my calls
1. **Heading-not-found appends the whole next page.** That follows from
   completeness over tightness; the alternative, appending nothing, is today's
   silent loss.
2. **Only one page of look-ahead.** The tail of a section cannot span more than
   the shared page under the tiling rule, because a full page with no new heading
   is already inside the range.
3. **Behaviour version, not artifact version:** a minor bump to 0.44.0, since tool
   output changes. Artifacts stay valid. The version bump invalidates reuse
   anyway, which is harmless.
4. **The summarizer is out of scope.** It is optional, LLM-side and range-based,
   and can adopt the same helper later if summaries are found to miss tails.

## Verification

### Unit (CI)
- **Synthetic PDF and text fixtures** covering:
  - a heading that opens its page (nothing appended);
  - a mid-page heading (exact `[0:h)` appended, marker at `h`);
  - an unlocatable heading (whole page plus NOTE);
  - two headings on the next page (stops at the earliest);
  - a deeper previous subtree (breadcrumb resolves to the last descendant);
  - two hits in one search straddling a heading (different breadcrumbs);
  - leading spaces and tabs before a heading (offsets match `TextSearchMatch`
    exactly);
  - a heading title mentioned in body text above the real heading (the matcher
    must not stop early).
- **Immutability:** two identical reads return identical strings, and
  `text_content` is byte-identical before and after.
- **NOTE parity:** for every read in the existing variant and continuation tests,
  the NOTE lines are unchanged.

### Real PDF (local, `real_pdf`)
- PSoC 6 "3.1.1 CPUs" (12-12): the read includes page 13's head up to the next
  heading, and the marker.

### Corpus script (gitignored corpus, all documents including RP2040 and the
ESP32 TRM)
- BME280 1.3 contains "Absolute accuracy" and "Long term stability".
- Every boundary where the next heading is mid-page gets its tail.
- No read shrinks.
- Report median and p90 appended characters per read, and the
  heading-not-found count.

## Release
- 0.44.0 CHANGELOG entry: what changed, the BME280 example, the corpus numbers.
- Update in `docs/datasheetindex_architecture.md`:
  - the `get_section_text` contract and its three-part output;
  - the `end_page` section (~line 289): one sentence that `end_page` stays page
    tiling and reads add the tail;
  - the search breadcrumb semantics.
- `tools/defs.py` tool text gets at most one short clause, if the framing line is
  not self-explanatory.
- Fold this spec into the architecture doc and delete it before merge.
- datasheet-agent: no required change. Its `_spec_pages` page union still misses
  the tail page; noted as a possible follow-up there, not here.

## Left to the builder
- The heading matcher: whitespace and line-wrap tolerance, numbering prefixes,
  and choosing the occurrence most likely to be the heading. 2,744 of 2,750 found
  shows a simple matcher suffices.
- Exact framing and marker wording.
- Where the lazy heading map lives, likely beside `artifacts.nodes` in
  `tools/bound.py`, and how the search lookup is threaded.
