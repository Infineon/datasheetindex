# Preamble: page-marked front matter with per-page signals

Design, 2026-07-25. Status: approved, not implemented.

Independent of `2026-07-25-on-disk-artifact-reuse-design.md`. Different concern,
ships separately.

## Problem

`generate_preamble` reads pages 1-2 and truncates at a line boundary near
`max_chars=2400`. Three defects follow, all measured on the two datasheets to
hand.

### It silently drops half the front matter, keeping the wrong half

| document | pages 1-2 | emitted | dropped |
|---|---|---|---|
| Infineon PSoC 6 (CY8C62x8) | 4747 ch | 2385 ch | 2362 ch (50%) |
| TI PCN 20251008002.1 | 2689 ch | 2398 ch | 291 ch (11%) |

Both hit the cap, so truncation is the normal case, not an edge case. On the
PSoC the dropped 50% is the whole of page 2, which continues the `Features`
section: 13 serial communication blocks, the audio subsystem, 32 TCPWMs, the
12-bit 2-Msps SAR ADC, "up to 102 programmable GPIOs", CAPSENSE. Those are
specifications.

What it *kept* is the tell: the emitted text ends on page 1's legal footer
(`...d "Warnings" at the end of this document`). A character budget applied in
document order retains boilerplate and discards specifications, because the
boilerplate happens to sit earlier. The knob is wrong, not just its value.

Nothing signals the loss. Compare `get_section_text`, which inserts a
`=== NOTE: ... ===` line when a requested range cuts marked content, under the
principle CHANGELOG 0.18.0 states outright: "Silence is not a completeness
claim." The preamble violates it.

Page 3 is never read regardless of what it holds -- 742 ch on the PSoC, 2071 ch
on the TI PCN.

### It does not record which page anything came from

Pages are joined with a bare `"\n"`. Unlike the page-matched text file there are
no `--- PAGE N ---` markers, so an agent cannot tell whether a feature it read
is on page 1 or 2, cannot cite it, and cannot navigate back to it.

### Front matter is not always front matter

A Product Change Notification's page 1 is a cover letter. Measured on the TI
PCN, the whole preamble is "TI requires acknowledgement of receipt of this
notification within 30 days..." with zero specifications. An agent has no way to
tell that orientation found nothing, and any consumer relying on the preamble to
propose comparison parameters would silently get none.

## Design

Report evidence; do not act on it. The library computes per-page signals and the
agent decides what to ignore. This is the architecture's own division of labour:
the library is a pre-processor and toolbox, the agent supplies the intelligence.
A skip heuristic would put intelligence in the library; a signal puts evidence
there.

### 1. Page markers

Emit `--- PAGE N ---` before each page's text, the same format the text file
uses. Every line becomes attributable and citable.

### 2. Budget

Default `max_chars` rises from 2400 to 5000, which fits the PSoC's 4747-char
front matter with headroom, and both `max_pages` and `max_chars` become
parameters. A caller doing parameter discovery can ask for more; we do not guess
its token economics.

The default is chosen from two documents and should be sanity-checked against a
wider corpus before release. The direction is what matters: retaining a legal
footer while dropping the specifications is worse than spending the tokens.

**`max_chars` bounds document text, not the returned string.** Markers and any
`NOTE` line are tool framing, and counting them against the budget would make the
amount of *content* a caller receives depend on how long the framing happens to
be. Worse, it is circular: the note names the truncation point, so its length is
not known until after truncating, and reserving space for it would mean
truncating twice.

The overhead is bounded and computable in advance -- at most
`max_pages * 17` characters of markers plus two `NOTE` lines of about 120
characters each -- so a caller sizing a token budget can account for it. State the
bound in the docstring.

This changes an existing guarantee. `test_respects_max_chars` and
`test_real_pdf_respects_max_chars` assert `len(preamble) <= max_chars` on the
whole string, and they must be rewritten to assert it on the document-text
portion, which `FrontMatter.chars_shown` reports directly. Rewriting a test to
match a deliberate semantic change is legitimate; doing it without saying so is
not, hence this paragraph and the Compatibility note below.

### 3. Truncation is marked

When a cap bites, append a line naming what was lost. The two caps fail
differently and cannot share a message, because the extractor's knowledge differs
in each case.

**`max_chars` bit.** All `max_pages` pages were extracted, so the total is known:

```
=== NOTE: preamble truncated at 5000 characters; 5000 of 7118 characters from
pages 1-2 shown, ending mid-page on page 2 ===
```

`M` is **characters on the pages read**, never "total front matter in the
document" -- that would require knowing where front matter ends, which is exactly
what cannot be determined. The `ending mid-page on page P` clause is dropped when
the cut coincides with a page boundary.

**`max_pages` bit.** The document has pages beyond the ones read, and whether they
are front matter is unknown, so the message claims only the mechanical fact:

```
=== NOTE: preamble covers pages 1-2 of 134; later pages were not examined ===
```

It deliberately does not say "front matter continues on page 3" or count
characters it never extracted. Reading one page past the limit merely to describe
what was skipped would defeat the limit.

Both caps can bite on one document, in which case both lines are emitted, the
character note first. The `===` wrapping matches the framing `get_section_text`
already uses and marks the line as tool output rather than document content, since
real datasheets contain their own literal `NOTE:` lines.

### 4. Per-page signals

A new top-level `preamble_pages` array in the ToC JSON, one entry per page read:

| field | meaning |
|---|---|
| `page` | 1-based page number |
| `chars` | extracted character count |
| `bullets` | lines opening with a bullet glyph or dash |
| `legal_hits` | matches of the legal-boilerplate vocabulary |
| `has_features_heading` | a line that is exactly `Features` or `General description` |

Measured discrimination:

| | chars | bullets | legal_hits | features heading |
|---|---|---|---|---|
| TI PCN p1 (cover letter) | 2160 | 0 | 4 | no |
| PSoC 6 p1 (real front matter) | 2432 | 34 | 0 | yes |
| PSoC 6 p2 (features continued) | 2314 | 43 | 0 | yes |

All three signals separate the two cases cleanly and independently.

#### `legal_hits` needs a prose matcher, which the existing pattern is not

`core/boilerplate.py`'s `legal` pattern cannot be used as it stands. It is
anchored `^(...)$` (`boilerplate.py:57,71`), so it matches a whole title and
nothing else: against page 1's footer sentence it scores **zero**, not the 4 hits
this design's own table reports -- those were measured with an ad-hoc prose
pattern. Reusing `classify_title` is doubly wrong here, since it also needs a ToC
and the document that needs this most has none.

Nor can a single vocabulary be compiled into both matchers unchanged. Several
title branches *deliberately* require a qualifier -- `legal\s+(disclaimer|
notices?|information)`, `important\s+(notices?|...)`, `product\s+liability` --
because bare `information`, `notice`, and `liability` are common substantive
section titles (`boilerplate.py:54-56` says so). In running prose the judgement
inverts: a bare "liability" or "warranty" in a footer sentence *is* the signal.
Forcing one list to serve both would either weaken the title matcher, which flags
boilerplate in a published artifact, or under-count prose.

So: share the vocabulary, keep two matchers, and let the difference between them
be explicit rather than accidental.

- `_LEGAL_VOCABULARY` -- one module-level constant, the terms and phrases that
  signal legal boilerplate: `disclaimer`, `warranty`/`warranties`, `liability`,
  `liable`, `trademark`, `copyright`, `patent`, `indemnif…`,
  `terms and conditions`, `limitation(s) of liability`, `export control`,
  `subject to change without notice`, `no license`, `as is`, `at your own risk`.
- **Prose matcher** -- unanchored, case-insensitive, word-boundary alternation over
  the whole vocabulary. `legal_hits` counts its matches in a page's text.
- **Title matcher** -- the existing anchored pattern, rebuilt from the same
  constant *minus* an explicit `_TITLE_UNSAFE` exclusion set (`liability`,
  `liable`, `as is`, `no license`) and plus the qualifier-requiring branches it
  already has.

**This refactor must be behaviour-preserving for `classify_title`.** It is a
new-feature spec touching tested classification code, so the bar is that the
existing `classify_title` tests pass untouched, and a new test asserts the
exclusion set explicitly: bare `liability` is a legal hit in prose and *not* a
legal title. If the refactor cannot be made behaviour-preserving, add the prose
matcher beside the title pattern with a comment explaining the asymmetry and
accept the partial duplication -- a drifted signal count is a cheaper failure than
a regressed boilerplate flag.

### 5. Unit density is deliberately omitted

A count of numeric-plus-unit tokens looks like the obvious signal and is left
out of the first version.

A naive ASCII pattern undercounts badly -- 5 matches on PSoC page 1 against 30
for a corrected one -- because it misses `150-MHz` (hyphen separator), `1.1-V`,
and `40 µA` (micro sign). Handling those requires both U+00B5 MICRO SIGN and
U+03BC GREEK SMALL LETTER MU, plus hyphen and slash separators. The corrected
pattern then produces false positives on part numbers, which datasheets are full
of: `8/A` from `CY8C62x8/A`, `4F` from `Cortex-M4F`.

So it is noisy in both directions and needs calibration against a corpus. The
three signals above already discriminate perfectly on both fixtures, so shipping
a fourth noisy one into a public artifact buys nothing. Add it later, calibrated
and tested against part-number forms, if a consumer needs it.

## Rejected: skipping the declaration page

Detecting a cover or legal page and dropping it was considered and rejected.

- **The error is asymmetric.** Wrongly skipping page 1 of a real datasheet costs
  the general description and half the features -- the most valuable page in the
  document. Wrongly keeping a cover page costs some tokens. A heuristic with a
  cheap failure in one direction and an expensive one in the other should not be
  the default.
- **It cannot be calibrated here.** Two documents is not a corpus. The relevant
  fixtures live in `datasheet-agent`'s golden set.
- **Precedent.** The table-engine gotcha in CLAUDE.md records the same shape of
  decision: classic finds more tables, the ML engine finds different ones,
  "Neither dominates. Stability is the point." Nobody debugs a preamble; a
  preamble that is cleverer 80% of the time and wrong 20% is worse than one that
  is predictable.
- **Reporting subsumes it.** A caller given the signals can implement skipping;
  a library that skips forecloses the alternative.

## API shape

`generate_preamble` returns a `str`, so it cannot carry the signals. Rather than
compute them in a second function that re-extracts every page:

```python
@dataclass(frozen=True)
class PreamblePage:
    page: int
    chars: int
    bullets: int
    legal_hits: int
    has_features_heading: bool

@dataclass(frozen=True)
class FrontMatter:
    text: str                     # page-marked, possibly NOTE-suffixed
    pages: list[PreamblePage]
    chars_shown: int              # document characters in text, framing excluded
    chars_extracted: int          # document characters on the pages read
    pages_read: int
    total_pages: int

    @property
    def char_truncated(self) -> bool:
        return self.chars_shown < self.chars_extracted

    @property
    def pages_omitted(self) -> int:
        return self.total_pages - self.pages_read

def build_front_matter(
    doc, *, max_pages: int = 2, max_chars: int = 5000
) -> FrontMatter: ...
```

**There is no single `truncated` flag.** An earlier draft had one, and it could
not be given an honest meaning: the two caps are independent, so one boolean must
either conflate them -- `truncated=True` telling a caller nothing about whether it
should raise `max_chars`, raise `max_pages`, or both -- or silently privilege one.
The four counts above are what the notes are rendered from, so the structured
fields and the prose cannot disagree, and `char_truncated` / `pages_omitted` give a
caller the two decisions separately.

Both new parameters are **keyword-only**. Every existing caller already passes
`max_chars` by keyword or omits it, so nothing breaks.

`index.py` switches to `build_front_matter`. `generate_preamble` is retained as a
one-line wrapper returning `.text`, since five tests in `tests/test_preamble.py`
target it directly and it costs nothing to keep.

**Implementation note.** `tests/test_index.py` monkeypatches
`datasheetindex.index.generate_preamble` with `lambda _doc: "pre"` in seven
places. Once `index.py` calls `build_front_matter` instead, those stubs patch a
function that is no longer called and must be repointed — they will not fail
loudly, they will silently stop stubbing, so this is the one change in this spec
that can quietly weaken existing tests.

## Compatibility

- `preamble` now contains `--- PAGE N ---` markers and is longer. A consumer
  treating it as opaque prose is unaffected; one parsing it line by line sees
  new marker lines. Verify nothing downstream parses it before release --
  `datasheet-agent` was checked and does not (it appears there only in
  `extract_chamber.py`, unrelated, and in a docstring).
- `preamble_pages` is a new top-level key in the ToC JSON. Purely additive.
- **`max_chars` now bounds document text rather than the whole returned string**
  (section 2), so the string can exceed it by a bounded amount of framing. This is
  the one non-additive change in this spec, and it is why two existing tests are
  rewritten. A caller sizing a token budget should subtract the stated overhead.

## Testing

- Page markers appear once per page read, in order, matching the text file's
  format.
- A front matter that fits the budget is emitted whole with no `NOTE` line.
- A front matter exceeding `max_chars` is truncated on a line boundary and carries
  the character note, whose `N` and `M` equal `chars_shown` and `chars_extracted`.
  Assert the numbers, not just the presence of the line, so the prose cannot drift
  from the fields.
- A document with pages beyond `max_pages` carries the page note naming
  `pages_read` and `total_pages`, and **does not** carry the character note when
  the text fit. The two notes are independent.
- A document that trips both caps carries both lines, character note first.
- `max_chars` bounds `chars_shown`, not `len(text)`: assert `chars_shown` is within
  the cap while the returned string legitimately exceeds it by the marker and note
  overhead, and assert that overhead is within the stated bound.
- `max_pages` and `max_chars` are both honoured, including `max_pages` larger
  than the document -- where `pages_omitted` is 0 and no page note is emitted.
- Signals, on synthetic pages so the assertions are exact: a bulleted feature
  page scores high bullets, zero legal hits, and a `Features` heading; a legal
  cover page scores zero bullets and non-zero legal hits.
- `legal_hits` counts matches of the prose matcher built from `_LEGAL_VOCABULARY`,
  asserted against that constant rather than a duplicated list.
- **A legal footer sentence scores non-zero `legal_hits`.** This is the test that
  the anchored title pattern would fail, so it pins the prose matcher's existence.
- Bare `liability` is a prose hit and not a legal *title*, pinning `_TITLE_UNSAFE`.
- Every existing `classify_title` test passes unmodified after the refactor.
- A one-page document yields one `preamble_pages` entry and does not error.
- The seven repointed stubs in `tests/test_index.py` still stub what `index.py`
  actually calls -- assert the stub value reaches the emitted JSON, so a stub
  that stops taking effect fails rather than passing quietly.
- No LLM and no network, so all of the above runs under a plain `uv sync`.

## Out of scope

- Skipping or reordering pages (above).
- Unit-density signals (above).
- Any change to the page-matched text file, which already carries markers.
- Extracting a *structured* feature list. That is parsing the front matter into
  fields, which is the extraction-engine direction the architecture rejects.
- Anything in `datasheet-agent`. The preamble-based axis strategy is filed there
  as issue #1 and is a consumer of this, not part of it.
