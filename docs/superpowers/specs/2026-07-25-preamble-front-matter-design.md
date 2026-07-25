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

### 3. Truncation is marked

When a cap does bite, append a line naming what was lost. The cut can fall
mid-page, so the message must not assume a page boundary:

```
=== NOTE: preamble truncated; N of M front-matter characters shown, ending
mid-page on page P ===
```

with the `mid-page on page P` clause dropped when the cut happens to coincide
with a page boundary or when `max_pages` rather than `max_chars` was the binding
limit. The `===` wrapping matches the framing `get_section_text` already uses and
marks the line as tool output rather than document content, since real datasheets
contain their own literal `NOTE:` lines.

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

`legal_hits` should reuse the vocabulary already in `core/boilerplate.py`'s
`legal` pattern (disclaimer, warranty, liability, trademark, ...) rather than
inventing a second list. Note that `classify_title` itself is not reusable here:
it classifies ToC section *titles* and needs a ToC, and the document this is
most needed for has none.

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
    truncated: bool

def build_front_matter(
    doc, *, max_pages: int = 2, max_chars: int = 5000
) -> FrontMatter: ...
```

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

## Testing

- Page markers appear once per page read, in order, matching the text file's
  format.
- A front matter that fits the budget is emitted whole with no `NOTE` line.
- A front matter exceeding the budget is truncated on a line boundary *and*
  carries the `NOTE` line naming what was dropped.
- `max_pages` and `max_chars` are both honoured, including `max_pages` larger
  than the document.
- Signals, on synthetic pages so the assertions are exact: a bulleted feature
  page scores high bullets, zero legal hits, and a `Features` heading; a legal
  cover page scores zero bullets and non-zero legal hits.
- `legal_hits` uses the same vocabulary as `boilerplate.classify_title`, asserted
  against a shared constant rather than a duplicated list.
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
