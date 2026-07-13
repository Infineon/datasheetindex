# Design: page-cut truncation signal for `get_section_text`

Status: proposed
Date: 2026-07-13

## Problem

An agent reads a page range, gets a table that ends cleanly at the bottom of the
last page, and answers from it. It has no way to know the table continued onto
the next page, because the evidence of the continuation -- the `(continued)`
marker -- sits at the top of the page it did not fetch. The truncation is
systematically located outside the fetched context.

This is not hypothetical. Measured on three real datasheets:

| Doc | continuation markers found today | tables cut by a leaf ToC section boundary |
| --- | --- | --- |
| Infineon TLE9350BSJ | 2 | 0 |
| TI TCAN1044A-Q1 | 0 | 3 |
| NXP TJA1051 | 0 | 0 (genuinely none) |

### Two defects, and why they hid each other

**Defect 1 -- the continuation regex is vendor-specific.**
`_CONTINUED_TABLE_RE` in `core/structure.py` requires a literal `Table N` prefix:

```python
re.compile(r"(Table\s+[\d\-\.]+\s+.+?)\s*\((?:[Cc]ontinued|[Cc]ont\.)\)")
```

Infineon writes `Table 9 Electrical characteristics transmitter (Continued)`, so
it matches. TI writes `6.4 Recommended Operating Conditions (continued)` -- no
`Table` token, lowercase `c` -- so it does not. `TocNode.continued_tables` is
therefore **empty for every TI and NXP document**. That is a live bug in the
shipped library, independent of this feature.

**Defect 2 -- nothing tells the reader when its range cuts a table.**
`get_section_text` emits only `=== Pages X-Y of N ===`. It never consults the
continuation data, so it cannot warn.

The two defects are *anti-correlated*, which is why neither surfaced. Where the
marker is detected (Infineon), the ToC section already contains the whole table
-- 7.4 spans pages 20-21 and the cut is at 20->21, inside the section -- so no
reader is harmed. Where a reader *is* harmed (TI), the marker is not detected.

### The concrete harm

TI section `6.4 Recommended Operating Conditions` has ToC range **pages 4-4**.
Its table continues onto page 5. An agent doing exactly what the tool
description tells it -- read the whole section, `get_section_text(4, 4)` --
silently loses:

- `TJ  Operating junction temperature  -40 ...`
- `IOH(RXD) ... Devices without VIO  -2 mA`
- `IOL(RXD) ... Devices without VIO   2 mA`

Worse than omission: page 4 *does* carry `IOH(RXD) ... Devices with VIO ...
-1.5 mA`. An agent asked for the RXD high-level output current reads a table
that ends cleanly, finds a plausible value, and reports `-1.5 mA` -- never
learning the "without VIO" variant exists. Nothing in the fetched context hints
at truncation.

`6.8 Electrical Characteristics` (ToC pages 6-7) has the same shape: its table
continues onto page 8.

## Design

Two changes, both in the pre-processing/toolbox layer. No extraction
intelligence is added; the library only surfaces a fact it already derives.

### 1. Generalize the continuation regex

Replace `_CONTINUED_TABLE_RE` with one that matches any heading-ish line ending
in a continuation marker, with an optional `Table N` label on the preceding line:

```python
_CONTINUED_TABLE_RE = re.compile(
    r"^[ \t]*(?:(Table\s+[\d.\-]+)[ \t]*\r?\n)?"   # optional "Table 9" label line
    r"[ \t]*(\S[^\n]{2,90}?)[ \t]*"                # the title text
    r"\((?:continued|cont\.)\)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
```

The title is the two groups joined and whitespace-normalized. This matches all
three conventions observed:

- `Table 1 Electrical Specs (Continued)` -> `Table 1 Electrical Specs`
- `Table 9\n  Electrical characteristics transmitter (Continued)` ->
  `Table 9 Electrical characteristics transmitter`
- `6.4 Recommended Operating Conditions (continued)` ->
  `6.4 Recommended Operating Conditions`

**Guard against false positives.** A bare `(continued)` line is not enough. TI's
mechanical-drawing pages (33-41) carry `NOTES: (continued)`. A match is accepted
only if its normalized title also appears on the **preceding** page -- which is
what makes it a *continuation* rather than a heading that happens to end in that
word. This guard is the correctness property of the whole design: when the
signal fires, the table provably spans the boundary.

**Compatibility.** `TocNode.continued_tables` titles become whitespace-normalized
(`"Table 9 \n  Electrical..."` -> `"Table 9 Electrical..."`), and TI/NXP docs
start reporting continuations where they previously reported none. Both are
intended corrections. Existing unit tests in `tests/test_continued_tables.py`
use single-line markers and are unaffected.

### 2. Peek one page past the requested range in `get_section_text`

New helper in `core/structure.py`, range-relative rather than section-relative,
so it protects narrow page reads and whole-section reads alike:

```python
def continuation_at_boundary(text_content: str, page: int) -> list[str]:
    """Titles of tables that span the page -> page+1 boundary.

    Empty when `page` is the last page, when page+1 carries no continuation
    marker, or when the marker fails the preceding-page guard.
    """
```

`DatasheetTools.get_section_text` calls it for `end_page` (tail cut) and for
`start_page - 1` (head cut), and appends notes under the existing position
header:

```
=== Pages 4-4 of 42 ===
NOTE: "6.4 Recommended Operating Conditions" continues on page 5, outside this
range. Re-read with end_page=5 to see the remaining rows.
```

For a head cut, the range opens mid-table and the agent is missing the column
headers that make the rows readable:

```
=== Pages 5-5 of 42 ===
NOTE: this range opens inside "6.4 Recommended Operating Conditions", which
starts on page 4. The column headers are on the earlier page.
```

The head-cut start page is found by walking back while consecutive pages carry
the same title.

### Wording is one-directional, by design

The note states only what is proven. It must **never** claim the converse --
no "this section is complete", and no `complete=true` field. Tables can spill
across a page break with no marker at all, so silence means "no continuation was
detected", not "the context is whole". Emitting a completeness claim would
convert a false negative into a false assurance, which is worse than the status
quo.

## What this is not

- Not a typed answer schema. The library stays a pre-processor; the answer
  contract belongs to the consuming agent.
- Not a geometric table-boundary detector. A `find_tables()` bbox check
  independently corroborated the TI cut, but the classic detector
  false-positives on plot gridlines (see `CLAUDE.md`), so it would be noisiest
  exactly where it is wrong. A truncation banner the agent learns to distrust is
  worth less than none. Revisit only if a vendor is found that spills tables
  with no marker at all.
- Not a change to `search_text` snippets or to the ToC JSON schema.
  `continued_tables` keeps its shape; only its detection improves.

## Testing

Unit tests use synthetic `text_content` (no PDF needed), matching the existing
style in `tests/test_continued_tables.py`:

- regex: Infineon two-line, TI single-line, and `Table N` single-line forms
- guard: `NOTES: (continued)` with no matching title on the preceding page is
  rejected
- `continuation_at_boundary`: fires at a cut, silent at the last page, silent
  when the marker is absent
- `get_section_text`: tail-cut note, head-cut note, both, neither; the header
  stays intact and no completeness claim is emitted

Regression fixture: a TI-style section whose ToC range ends one page before its
table does -- the exact `6.4 Recommended Operating Conditions` shape. This is
the case the current code cannot see.

The real PDFs are not in the repo (`tests/conftest.py` points at a sibling
`data2page/` directory), so PDF-backed assertions stay opt-in and skip when
absent, as they do today.
