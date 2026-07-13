# Design: page-cut truncation signal for `get_section_text`

Status: proposed
Date: 2026-07-13

## Problem

An agent reads a page range, gets a table that ends cleanly at the bottom of the
last page, and answers from it. It has no way to know the table continued onto
the next page, because the evidence of the continuation -- the `(continued)`
marker -- sits at the top of the page it did not fetch. The truncation is
systematically located outside the fetched context.

`get_section_text` emits only `=== Pages X-Y of N ===`. It never checks whether
the range it is handing back cuts content that continues past `end_page`.

### The concrete harm

Measured on the TI TCAN1044A-Q1. Section `6.4 Recommended Operating Conditions`
has ToC range **pages 4-4**, but its table continues onto page 5. An agent doing
exactly what the tool description tells it -- read the whole section,
`get_section_text(4, 4)` -- silently loses:

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

The Infineon TLE9350BSJ, by contrast, has two continued tables that both sit
*entirely inside* their leaf ToC section (7.4 spans 20-21, cut at 20->21). No
reader is harmed there. The harm requires the cut to fall on the boundary of the
range actually requested -- which is why the signal must be **range-relative**,
not section-relative. A section-aware check would miss the TI case entirely,
since that read *is* a whole-section read.

## Design

One change, in the toolbox layer. No extraction intelligence is added; the
library surfaces a fact derivable from text it already holds.

### `continued_tables` is left alone

`TocNode.continued_tables` keeps its current contract: tables captioned
`Table N ... (Continued)`, matched by the existing `_CONTINUED_TABLE_RE`. It is
Infineon-shaped, and under its own definition TI has no such tables -- that is a
narrow definition, not a bug.

The boundary signal is a **separate concept** with its own matcher: "does content
continue across this page break", not "which tables in this section are
captioned as continued". Broadening `continued_tables` to match TI's
`6.4 Recommended Operating Conditions (continued)` would put *section* titles in
a field named for *table* titles -- a semantic change to the ToC JSON. Nothing
consumes the field today (grepped: no usage in `datasheet-agent`), so there is
no reason to touch its contract. The two stay decoupled.

### The boundary matcher

New, private to the boundary check:

```python
_CONTINUATION_RE = re.compile(
    r"^[ \t]*(\S.{2,90}?)[ \t]*\((?:continued|cont\.)\)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_OPENING_BLOCK_LINES = 5
```

**The guard is positional.** A continuation marker is only honoured if it appears
within the first `_OPENING_BLOCK_LINES` **nonblank** lines of the page -- a table
that resumes does so at the top of the page, below the running header.

This is the whole correctness property, and it is what the data supports. Across
the Infineon and TI documents:

| | genuine continuations | `NOTES:` false positives |
| --- | --- | --- |
| position on page | nonblank line **#3** (4 of 4) | nonblank line **#19-48** (6 of 6) |

TI's mechanical-drawing pages (33-41) carry mid-page `NOTES: (continued)` blocks.
The positional guard rejects all six with a wide margin (3 vs 19), and keeps all
four real ones.

An earlier draft guarded instead by requiring the marker's title to also appear
on the *preceding* page. That check was measured and **rejected**: all ten
matches pass it, including all six false positives, so it has no discriminating
power. It is not carried forward.

### The boundary check

New helper in `core/structure.py`, range-relative:

```python
def continuation_at_boundary(text_content: str, page: int) -> list[str]:
    """Titles of content that continues from `page` onto `page + 1`.

    A title is returned when page+1 opens with a continuation marker inside its
    opening block. Empty when `page` is the last page, or no such marker exists.
    """
```

`DatasheetTools.get_section_text` calls it for `end_page` (the range's tail cut)
and for `start_page - 1` (the range opens mid-continuation), and appends notes
under the existing position header.

### Note copy states only what is proven

What the heuristic establishes is exactly this: *the publisher marked content on
the next page as continuing from this one.* It does not establish that the
content is a table, that rows are missing, or where the column headers live. (In
the TI case the continuation page in fact **repeats** its `MIN NOM MAX UNIT`
headers -- so a note claiming "the column headers are on the earlier page" would
be false for the very document that motivated this design.)

The copy stops there:

```
=== Pages 4-4 of 42 ===
NOTE: "6.4 Recommended Operating Conditions" is continued on page 5, which is
outside this range.
```

```
=== Pages 5-5 of 42 ===
NOTE: this range opens inside "6.4 Recommended Operating Conditions", which is
continued from page 4.
```

No claim about rows, headers, or what re-reading would reveal. The agent has the
page number and can decide.

### Silence is not a completeness claim

The note is one-directional. It must **never** assert the converse -- no "this
section is complete", no `complete=true` field. Content can spill across a page
break with no marker at all, so silence means "no continuation was detected", not
"the context is whole". A completeness claim would convert a false negative into
a false assurance, which is worse than the status quo.

## What this is not

- Not a typed answer schema. The library stays a pre-processor; the answer
  contract belongs to the consuming agent.
- Not a geometric table-boundary detector. A `find_tables()` bbox check
  independently corroborated the TI cut, but the classic detector
  false-positives on plot gridlines (see `CLAUDE.md`), so it would be noisiest
  exactly where it is wrong. Revisit only if a vendor is found that spills
  tables with no marker at all -- which would also be invisible to this design,
  and is the known limitation to accept for now.
- Not a change to `search_text`, to `continued_tables`, or to the ToC JSON.

## Testing

Unit tests use synthetic `text_content` (no PDF needed), matching the existing
style in `tests/test_continued_tables.py`:

- matcher: TI single-line (`6.4 Recommended Operating Conditions (continued)`),
  Infineon (`Electrical characteristics transmitter (Continued)`), and
  `Table N ... (Continued)` forms
- positional guard: a marker at nonblank line 3 fires; the same marker pushed to
  line 19 does not -- the `NOTES:` shape
- `continuation_at_boundary`: fires at a cut, silent on the last page, silent
  when no marker is present
- `get_section_text`: tail-cut note, head-cut note, both, neither; the position
  header stays intact and no completeness claim is emitted
- `continued_tables` is unchanged by this work: existing tests must pass untouched

Regression fixture: a section whose ToC range ends one page before its table does
-- the `6.4 Recommended Operating Conditions` shape, ToC pages 4-4 with the table
running onto page 5.

The real PDFs are not in the repo (`tests/conftest.py` points at a sibling
`data2page/` directory), so PDF-backed assertions stay opt-in and skip when
absent, as they do today.
