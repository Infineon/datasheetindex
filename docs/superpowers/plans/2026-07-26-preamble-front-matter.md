# Preamble Front Matter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `generate_preamble`'s silent character-budget truncation with page-marked front matter that discloses what it dropped and reports per-page signals, so an agent can attribute, cite, and judge the text it reads.

**Architecture:** `core/preamble.py` grows one new entry point, `build_front_matter(doc, *, max_pages, max_chars) -> FrontMatter`, which extracts pages 1..`max_pages`, emits `--- PAGE N ---` markers in the same format as the page-matched text file, appends `=== NOTE: ... ===` lines when either cap bites, and returns per-page signal counts alongside the text. `generate_preamble` stays as a one-line wrapper over `.text`. `core/boilerplate.py` gains a prose legal matcher (`count_legal_hits`) beside its existing anchored title patterns. `index.py` calls `build_front_matter` and emits a new additive top-level `preamble_pages` key.

**Tech Stack:** Python 3.11+, PyMuPDF (only runtime dependency), pytest, ruff, ty, uv.

**Source spec:** `docs/superpowers/specs/2026-07-25-preamble-front-matter-design.md`. Read it before starting; it carries the measurements behind every constant here.

## Global Constraints

- **No new dependencies.** PyMuPDF is the only runtime dependency. No LLM call, no network in any code or test added by this plan, so everything runs under a plain `uv sync`.
- **ASCII-only source.** Per `CLAUDE.md`: no emoji or literal Unicode symbols in scripts or tests. Bullet glyphs in regexes go in as `\uXXXX` escapes.
- **Ruff line-length 88**, ruff format, and `ty` type checking. Pre-commit hooks enforce all of it; never pass `--no-verify`.
- **No f-string without a variable** (project code style).
- **Run tests with `uv run pytest`.** Capture output to a log when it is large: `uv run pytest 2>&1 | tee /tmp/pre-test.log`.
- **`max_chars` bounds document text, not the returned string.** Markers and NOTE lines are framing and are excluded from the budget. This is the one non-additive change in the plan; two existing tests are rewritten because of it (Task 1).
- **Page text is emitted verbatim — no `rstrip`.** Today's `core/preamble.py:39` rstrips; dropping that is required for the framing-overhead formula to hold exactly.
- **Assembly format is fixed:** parts are `[marker_1, text_1, marker_2, text_2, ...]` joined with a single `"\n"`, no trailing newline — byte-identical in shape to `generate_text` (`core/textfile.py:248-259`). Each NOTE is appended as `"\n" + note`.
- **Deviation from the spec, taken deliberately:** the spec's section 4 preferred rebuilding `classify_title`'s anchored legal pattern from a shared `_LEGAL_VOCABULARY` constant, with a documented fallback — "add the prose matcher beside the title pattern with a comment explaining the asymmetry and accept the partial duplication". This plan takes the fallback. `classify_title` is not touched at all, so the boilerplate flag published in the artifact cannot regress. Task 3 pins the asymmetry with a test instead of a constant.

---

### Task 1: `build_front_matter` — page markers, caps as parameters, counts

**Files:**
- Modify: `src/datasheetindex/core/preamble.py` (full rewrite of a 39-line module)
- Test: `tests/test_preamble.py`

**Interfaces:**
- Consumes: `datasheetindex.core.textfile._extract_page_text(page) -> str` (existing).
- Produces:
  - `DEFAULT_MAX_PAGES: int = 2`, `DEFAULT_MAX_CHARS: int = 5000`
  - `PreamblePage(page: int, chars: int, bullets: int = 0, legal_hits: int = 0, has_features_heading: bool = False)`, frozen dataclass, with `to_dict() -> dict[str, object]`. **The three signal fields keep their defaults until Task 4** — do not assert on them before then.
  - `FrontMatter(text: str, pages: list[PreamblePage], chars_shown: int, chars_extracted: int, pages_read: int, total_pages: int)`, frozen dataclass, with properties `char_truncated: bool` and `pages_omitted: int`.
  - `build_front_matter(doc, *, max_pages: int = DEFAULT_MAX_PAGES, max_chars: int = DEFAULT_MAX_CHARS) -> FrontMatter`
  - `generate_preamble(doc, max_chars: int = DEFAULT_MAX_CHARS) -> str` (unchanged signature, now a wrapper).

Note: NOTE lines arrive in Task 2. This task emits markers and counts only.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_preamble.py` (keep the five existing tests in place for now; two of them are rewritten in Step 5):

```python
from datasheetindex.core.preamble import (
    DEFAULT_MAX_CHARS,
    build_front_matter,
    generate_preamble,
)
from datasheetindex.core.textfile import _extract_page_text


def _doc_with_lines(pages: int, lines: int = 6, width: int = 40):
    """A doc whose pages carry `lines` lines of `width` 'A' characters."""
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page()
        writer = pymupdf.TextWriter(page.rect)
        y = 72
        for _ in range(lines):
            writer.append((72, y), "A" * width)
            y += 14
        writer.write_text(page)
    return doc


def _framing_overhead(pages: int) -> int:
    """Marker + separator characters for pages 1..pages, per the spec."""
    return sum(13 + len(str(n)) for n in range(1, pages + 1)) + 2 * pages - 1


def test_page_markers_appear_once_per_page_in_order():
    doc = _doc_with_lines(2)
    fm = build_front_matter(doc)
    doc.close()

    assert fm.text.count("--- PAGE 1 ---") == 1
    assert fm.text.count("--- PAGE 2 ---") == 1
    assert fm.text.index("--- PAGE 1 ---") < fm.text.index("--- PAGE 2 ---")
    assert fm.text.startswith("--- PAGE 1 ---\n")


def test_front_matter_that_fits_is_emitted_whole():
    doc = _doc_with_lines(2)
    fm = build_front_matter(doc)
    doc.close()

    assert fm.chars_shown == fm.chars_extracted
    assert fm.char_truncated is False
    assert fm.pages_read == 2
    assert fm.total_pages == 2
    assert fm.pages_omitted == 0
    assert "NOTE" not in fm.text


def test_framing_overhead_matches_the_formula_on_two_pages():
    doc = _doc_with_lines(2)
    fm = build_front_matter(doc)
    doc.close()

    assert len(fm.text) - fm.chars_shown == _framing_overhead(2) == 31


def test_framing_overhead_matches_the_formula_at_three_digit_pages():
    """Blank pages extract to "", so the whole string is framing."""
    doc = pymupdf.open()
    for _ in range(105):
        doc.new_page()
    fm = build_front_matter(doc, max_pages=105)
    doc.close()

    assert fm.chars_shown == 0
    assert len(fm.text) == _framing_overhead(105)


def test_page_text_is_emitted_verbatim_without_rstrip():
    """The formula rests on this, so it gets its own test.

    Block text normally ends in a newline, so a page's segment must appear
    between its markers character for character -- no rstrip anywhere.
    """
    doc = _doc_with_lines(2)
    page_one = _extract_page_text(doc[0])
    fm = build_front_matter(doc)
    doc.close()

    assert f"--- PAGE 1 ---\n{page_one}\n--- PAGE 2 ---" in fm.text
    assert len(fm.text) - fm.chars_shown == _framing_overhead(2)


def test_max_pages_larger_than_document_reads_what_exists():
    doc = _doc_with_lines(1)
    fm = build_front_matter(doc, max_pages=9)
    doc.close()

    assert fm.pages_read == 1
    assert fm.total_pages == 1
    assert fm.pages_omitted == 0
    assert len(fm.pages) == 1
    assert fm.pages[0].page == 1
    assert fm.pages[0].chars == fm.chars_extracted


def test_max_chars_bounds_document_text_not_the_string():
    doc = _doc_with_lines(2, lines=45, width=80)
    fm = build_front_matter(doc, max_chars=200)
    doc.close()

    assert fm.chars_shown <= 200
    assert fm.char_truncated is True
    # The returned string legitimately exceeds the cap: markers are framing.
    assert len(fm.text) > fm.chars_shown


def test_generate_preamble_wraps_build_front_matter():
    doc = _doc_with_lines(1)
    text = generate_preamble(doc)
    expected = build_front_matter(doc).text
    doc.close()

    assert text == expected
    assert DEFAULT_MAX_CHARS == 5000


def test_pages_entries_cover_every_page_read_even_when_truncated():
    doc = _doc_with_lines(2, lines=45, width=80)
    fm = build_front_matter(doc, max_chars=200)
    doc.close()

    assert [p.page for p in fm.pages] == [1, 2]
    assert fm.chars_extracted == sum(p.chars for p in fm.pages)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_preamble.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_front_matter'`.

- [ ] **Step 3: Rewrite `src/datasheetindex/core/preamble.py`**

```python
"""Front matter extraction: page-marked pages 1-2 text for agent orientation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pymupdf

from datasheetindex.core.textfile import _extract_page_text

#: Pages read from the front of the document.
DEFAULT_MAX_PAGES = 2
#: Document characters emitted. Raised from 2400 in 0.25.0: the old value cut
#: the measured 4747-character front matter of the PSoC 6 in half and kept the
#: half holding the legal footer rather than the specifications.
DEFAULT_MAX_CHARS = 5000


@dataclass(frozen=True)
class PreamblePage:
    """Per-page evidence about one front-matter page.

    The signals are reported, never acted on: the library says what it saw and
    the agent decides what to ignore. A cover letter and a real features page
    are separated by these counts, but skipping either is the caller's call.
    """

    page: int
    chars: int
    bullets: int = 0
    legal_hits: int = 0
    has_features_heading: bool = False

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "page": self.page,
            "chars": self.chars,
            "bullets": self.bullets,
            "legal_hits": self.legal_hits,
            "has_features_heading": self.has_features_heading,
        }


@dataclass(frozen=True)
class FrontMatter:
    """The preamble text plus what it cost and what it left out.

    There is deliberately no single ``truncated`` flag. The two caps are
    independent, so one boolean could not tell a caller whether to raise
    ``max_chars``, ``max_pages``, or both. ``char_truncated`` and
    ``pages_omitted`` give the two decisions separately, and the NOTE lines in
    ``text`` are rendered from these same numbers, so prose and fields cannot
    disagree.
    """

    text: str
    pages: list[PreamblePage]
    chars_shown: int
    chars_extracted: int
    pages_read: int
    total_pages: int

    @property
    def char_truncated(self) -> bool:
        """True when ``max_chars`` cut document text that was extracted."""
        return self.chars_shown < self.chars_extracted

    @property
    def pages_omitted(self) -> int:
        """Pages the document holds beyond the ones read."""
        return self.total_pages - self.pages_read


def _truncate_on_line_boundary(text: str, budget: int) -> str:
    """Keep whole lines while they fit in ``budget``. No rstrip."""
    kept: list[str] = []
    total = 0
    for line in text.splitlines(keepends=True):
        if total + len(line) > budget:
            break
        kept.append(line)
        total += len(line)
    return "".join(kept)


def build_front_matter(
    doc: pymupdf.Document,
    *,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> FrontMatter:
    """Extract page-marked front matter with per-page signals. Zero LLM calls.

    ``max_chars`` bounds *document text* (``chars_shown``), not the returned
    string: the ``--- PAGE N ---`` markers and any ``=== NOTE: ... ===`` line
    are tool framing. Marker overhead for pages 1..P is exactly
    ``sum(13 + digits(n) for n in 1..P) + (2 * P - 1)`` -- 31 at the default
    ``max_pages=2``. Note lines add roughly 120 characters each; that is an
    estimate, not a bound, since they embed page numbers and counts.
    """
    total_pages = len(doc)
    pages_read = max(0, min(max_pages, total_pages))
    texts = [_extract_page_text(doc[i]) for i in range(pages_read)]

    parts: list[str] = []
    chars_shown = 0
    budget = max_chars
    for offset, page_text in enumerate(texts):
        page_num = offset + 1
        if budget <= 0 and page_num > 1:
            break
        kept = (
            page_text
            if len(page_text) <= budget
            else _truncate_on_line_boundary(page_text, budget)
        )
        parts.append(f"--- PAGE {page_num} ---")
        parts.append(kept)
        chars_shown += len(kept)
        budget -= len(kept)
        if kept != page_text:
            break

    pages = [
        PreamblePage(page=offset + 1, chars=len(page_text))
        for offset, page_text in enumerate(texts)
    ]

    return FrontMatter(
        text="\n".join(parts),
        pages=pages,
        chars_shown=chars_shown,
        chars_extracted=sum(len(t) for t in texts),
        pages_read=pages_read,
        total_pages=total_pages,
    )


def generate_preamble(doc: pymupdf.Document, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Return only the page-marked front-matter text.

    Retained for callers that want the string; ``build_front_matter`` carries
    the signals and the truncation counts.
    """
    return build_front_matter(doc, max_chars=max_chars).text
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `uv run pytest tests/test_preamble.py -v`
Expected: the nine new tests PASS. `test_respects_max_chars` and `test_real_pdf_respects_max_chars` now FAIL — they assert `len(preamble) <= max_chars` on the whole string, which markers deliberately break. Step 5 rewrites them.

- [ ] **Step 5: Rewrite the two tests the semantic change invalidates**

In `tests/test_preamble.py`, replace `test_respects_max_chars` with:

```python
def test_respects_max_chars_on_document_text():
    """max_chars bounds document text; markers and notes are framing.

    Rewritten in 0.26.0: this asserted len(preamble) <= max_chars on the whole
    returned string, a guarantee page markers deliberately give up.
    """
    doc = _doc_with_lines(2, lines=45, width=80)
    fm = build_front_matter(doc, max_chars=200)
    doc.close()
    assert fm.chars_shown <= 200
```

and replace `test_real_pdf_respects_max_chars` with:

```python
@pytest.mark.real_pdf
def test_real_pdf_respects_max_chars_on_document_text():
    """Real PDF: max_chars bounds document text, not the framed string."""
    if not TLE9350_PATH.exists():
        pytest.skip("Test PDF not found")
    doc = pymupdf.open(str(TLE9350_PATH))
    fm = build_front_matter(doc, max_chars=500)
    doc.close()
    assert fm.chars_shown <= 500
```

- [ ] **Step 6: Run the whole preamble suite plus lint and types**

Run: `uv run pytest tests/test_preamble.py -v && uv run ruff format src tests && uv run ruff check src tests && uv run ty check src`
Expected: all PASS, no lint or type errors.

- [ ] **Step 7: Commit**

```bash
git add src/datasheetindex/core/preamble.py tests/test_preamble.py
git commit -m "feat: page-marked front matter with parameterized caps

build_front_matter emits --- PAGE N --- markers, makes max_pages and
max_chars parameters, raises the character default 2400 -> 5000, and
returns the counts a caller needs to size a budget. max_chars now bounds
document text rather than the whole string; the two tests asserting the
old guarantee are rewritten against chars_shown."
```

---

### Task 2: Truncation is marked

**Files:**
- Modify: `src/datasheetindex/core/preamble.py`
- Test: `tests/test_preamble.py`

**Interfaces:**
- Consumes: `build_front_matter`, `FrontMatter` from Task 1.
- Produces: no new public names. `FrontMatter.text` may now end with one or two appended `=== NOTE: ... ===` lines, character note first.

Exact strings, from the spec:

```
=== NOTE: preamble truncated at 5000 characters; 5000 of 7118 characters from pages 1-2 shown, ending mid-page on page 2 ===
=== NOTE: preamble covers pages 1-2 of 134; later pages were not examined ===
```

The `, ending mid-page on page P` clause is dropped when the cut fell on a page boundary. The page phrase is `page 1` when one page was read and `pages 1-P` otherwise.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_preamble.py`:

```python
def test_char_truncation_note_carries_the_exact_counts():
    doc = _doc_with_lines(2, lines=45, width=80)
    fm = build_front_matter(doc, max_chars=200)
    doc.close()

    expected = (
        f"=== NOTE: preamble truncated at 200 characters; "
        f"{fm.chars_shown} of {fm.chars_extracted} characters from "
        f"pages 1-2 shown, ending mid-page on page 1 ==="
    )
    assert fm.text.endswith(expected)


def test_page_note_names_pages_read_and_total():
    doc = _doc_with_lines(4)
    fm = build_front_matter(doc)
    doc.close()

    assert fm.text.endswith(
        "=== NOTE: preamble covers pages 1-2 of 4; "
        "later pages were not examined ==="
    )
    # The caps are independent: the text fit, so no character note.
    assert "truncated at" not in fm.text


def test_both_notes_appear_with_the_character_note_first():
    doc = _doc_with_lines(4, lines=45, width=80)
    fm = build_front_matter(doc, max_chars=200)
    doc.close()

    notes = [ln for ln in fm.text.splitlines() if ln.startswith("=== NOTE:")]
    assert len(notes) == 2
    assert "truncated at 200 characters" in notes[0]
    assert "later pages were not examined" in notes[1]


def test_page_boundary_cut_omits_the_mid_page_clause():
    """A cap that lands exactly on a page boundary cut no page in half."""
    doc = _doc_with_lines(2, lines=4, width=40)
    page_one_chars = build_front_matter(doc, max_pages=1).chars_extracted
    fm = build_front_matter(doc, max_chars=page_one_chars)
    doc.close()

    assert fm.char_truncated is True
    assert "ending mid-page" not in fm.text
    assert fm.text.count("--- PAGE ") == 1


def test_single_page_document_note_uses_singular_page_phrase():
    doc = _doc_with_lines(1, lines=45, width=80)
    fm = build_front_matter(doc, max_chars=200)
    doc.close()

    assert "characters from page 1 shown" in fm.text
    assert "pages 1-1" not in fm.text


def test_notes_are_framing_and_excluded_from_chars_shown():
    doc = _doc_with_lines(4, lines=45, width=80)
    fm = build_front_matter(doc, max_chars=200)
    doc.close()

    assert fm.chars_shown <= 200
    assert len(fm.text) > 200
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_preamble.py -k "note or clause or boundary" -v`
Expected: FAIL — no NOTE line is emitted yet.

- [ ] **Step 3: Implement the notes**

In `src/datasheetindex/core/preamble.py`, add the three helpers below `_truncate_on_line_boundary`:

```python
def _page_phrase(pages_read: int) -> str:
    """``page 1`` for one page, ``pages 1-P`` otherwise."""
    if pages_read == 1:
        return "page 1"
    return f"pages 1-{pages_read}"


def _char_note(
    *, max_chars: int, chars_shown: int, chars_extracted: int,
    pages_read: int, cut_page: int,
) -> str:
    """Name what ``max_chars`` cut.

    ``chars_extracted`` is characters on the pages *read*, never "front matter
    in the document" -- that would require knowing where front matter ends,
    which is exactly what cannot be determined.
    """
    tail = f", ending mid-page on page {cut_page}" if cut_page else ""
    return (
        f"=== NOTE: preamble truncated at {max_chars} characters; "
        f"{chars_shown} of {chars_extracted} characters from "
        f"{_page_phrase(pages_read)} shown{tail} ==="
    )


def _page_note(*, pages_read: int, total_pages: int) -> str:
    """Claim only the mechanical fact: pages exist that were not examined.

    Deliberately does not say front matter continues, or count characters it
    never extracted -- reading one page past the limit to describe what was
    skipped would defeat the limit.
    """
    return (
        f"=== NOTE: preamble covers {_page_phrase(pages_read)} of "
        f"{total_pages}; later pages were not examined ==="
    )
```

In `build_front_matter`, track the cut page inside the loop by replacing the loop-terminating `break`:

```python
        if kept != page_text:
            cut_page = page_num
            break
```

and initialize `cut_page = 0` beside `chars_shown = 0`. Then, after the loop and before constructing `FrontMatter`, assemble the text with its notes:

```python
    chars_extracted = sum(len(t) for t in texts)
    text = "\n".join(parts)
    if chars_shown < chars_extracted:
        text += "\n" + _char_note(
            max_chars=max_chars,
            chars_shown=chars_shown,
            chars_extracted=chars_extracted,
            pages_read=pages_read,
            cut_page=cut_page,
        )
    if total_pages > pages_read:
        text += "\n" + _page_note(
            pages_read=pages_read, total_pages=total_pages
        )
```

and pass `text=text` / `chars_extracted=chars_extracted` to the `FrontMatter(...)` call instead of recomputing them.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_preamble.py -v`
Expected: all PASS, including Task 1's `test_front_matter_that_fits_is_emitted_whole` (no NOTE when nothing was lost) and both overhead-formula tests (which use documents that trip no cap).

- [ ] **Step 5: Lint, format, type check**

Run: `uv run ruff format src tests && uv run ruff check src tests && uv run ty check src`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/datasheetindex/core/preamble.py tests/test_preamble.py
git commit -m "feat: disclose preamble truncation instead of dropping text silently

Each cap gets its own NOTE line, since the extractor's knowledge differs:
max_chars knows the total on the pages it read, max_pages knows only that
later pages exist. Both can fire; the character note comes first. Silence
is not a completeness claim."
```

---

### Task 3: A prose legal matcher in `core/boilerplate.py`

**Files:**
- Modify: `src/datasheetindex/core/boilerplate.py`
- Test: `tests/test_boilerplate.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `count_legal_hits(text: str) -> int` in `datasheetindex.core.boilerplate`, and the module-level constant `_LEGAL_VOCABULARY: tuple[str, ...]`.

Why a second matcher rather than a shared one: `_BOILERPLATE_PATTERNS`'s legal branch is anchored `^(...)$` (`boilerplate.py:57,71`), so against page 1's footer *sentence* it scores zero, not the 4 hits the spec measured. And several of its branches deliberately require a qualifier — `product liability`, `important notices` — because bare `liability` and `information` are common substantive section titles. In running prose that judgement inverts: a bare "liability" in a footer sentence *is* the signal. `classify_title` is left untouched (see Global Constraints).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_boilerplate.py`:

```python
from datasheetindex.core.boilerplate import _LEGAL_VOCABULARY, count_legal_hits


def test_legal_footer_sentence_scores_non_zero():
    """The test the anchored title pattern fails; it pins the prose matcher."""
    footer = (
        "Infineon Technologies AG makes no warranty of any kind with respect "
        "to this document, and disclaims all liability arising from its use. "
        "Please note the disclaimer and the section headed Warnings at the "
        "end of this document. Specifications are subject to change without "
        "notice."
    )
    assert count_legal_hits(footer) >= 4
    assert classify_title(footer) == ""


def test_bare_liability_is_a_prose_hit_but_not_a_legal_title():
    assert count_legal_hits("we accept no liability") == 1
    assert classify_title("Liability") == ""
    assert classify_title("Product Liability") == "legal"


def test_features_prose_scores_zero_legal_hits():
    features = (
        "32-bit Arm Cortex-M4F CPU at 150 MHz, 2 MByte flash, 1 MByte SRAM, "
        "up to 102 programmable GPIOs, 12-bit 2-Msps SAR ADC, CAPSENSE."
    )
    assert count_legal_hits(features) == 0


def test_prose_matcher_covers_every_vocabulary_stem():
    """Every vocabulary entry is reachable, asserted against the constant.

    The entries are regex fragments, so they cannot be fed back through the
    matcher directly (`warrant(?:y|ies)` does not match its own pattern text).
    The length assertion fails if a term is dropped from the constant; the loop
    fails if one stops matching.
    """
    stems = [
        "disclaimer",
        "warranty",
        "liability",
        "liable",
        "trademark",
        "copyright",
        "patent",
        "indemnify",
        "terms and conditions",
        "limitation of liability",
        "export control",
        "subject to change without notice",
        "no license",
        "as is",
        "at your own risk",
    ]
    assert len(_LEGAL_VOCABULARY) == len(stems)
    for stem in stems:
        assert count_legal_hits(stem) >= 1, stem


def test_legal_hits_is_case_insensitive():
    assert count_legal_hits("TRADEMARKS") == count_legal_hits("trademarks") == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_boilerplate.py -v`
Expected: FAIL with `ImportError: cannot import name 'count_legal_hits'`.

- [ ] **Step 3: Add the vocabulary and the matcher**

Append to `src/datasheetindex/core/boilerplate.py`, after `_BOILERPLATE_PATTERNS`:

```python
# Vocabulary that signals legal boilerplate in *running prose*. Used by the
# front-matter `legal_hits` signal, and deliberately NOT shared with the
# `legal` branch of `_BOILERPLATE_PATTERNS` above, which serves a different
# question. That pattern is anchored to a whole title and several of its
# branches require a qualifier (`product liability`, `important notices`),
# because bare `liability` and `information` are common substantive section
# titles. In prose the judgement inverts: a bare "liability" in a footer
# sentence is exactly the signal. One list cannot serve both without either
# weakening the title matcher -- which publishes a flag in the artifact -- or
# under-counting prose. The partial duplication is the cheaper failure.
_LEGAL_VOCABULARY: tuple[str, ...] = (
    r"disclaimers?",
    r"warrant(?:y|ies)",
    r"liabilit(?:y|ies)",
    r"liable",
    r"trademarks?",
    r"copyrights?",
    r"patents?",
    r"indemnif\w*",
    r"terms\s+and\s+conditions",
    r"limitations?\s+of\s+liability",
    r"export\s+control",
    r"subject\s+to\s+change\s+without\s+notice",
    r"no\s+license",
    r"as\s+is",
    r"at\s+your\s+own\s+risk",
)

_LEGAL_PROSE_RE = re.compile(
    r"\b(?:" + "|".join(_LEGAL_VOCABULARY) + r")\b", re.IGNORECASE
)
```

and add the function at the end of the module:

```python
def count_legal_hits(text: str) -> int:
    """Count legal-boilerplate vocabulary matches in running prose.

    A count, not a verdict: a cover letter scores high and a features page
    scores zero, but what to do about that is the caller's decision.
    """
    return len(_LEGAL_PROSE_RE.findall(text))
```

Multi-word entries use `\s+` rather than a literal space so a phrase broken across a line break still matches. Alternation is ordered, so `limitations? of liability` consumes the whole phrase as one hit rather than also counting the inner `liability` — a signal, not an inventory.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_boilerplate.py -v`
Expected: all PASS, **including every pre-existing `classify_title` and `flag_boilerplate` test unmodified** — this task adds names and changes no behaviour. If any existing test changed, revert it and fix the addition instead.

- [ ] **Step 5: Lint, format, type check**

Run: `uv run ruff format src tests && uv run ruff check src tests && uv run ty check src`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/datasheetindex/core/boilerplate.py tests/test_boilerplate.py
git commit -m "feat: add a prose legal-boilerplate matcher beside the title matcher

The existing legal pattern is anchored to a whole title and scores zero on
a footer sentence, and several branches need a qualifier because bare
'liability' is a common substantive title. In prose that inverts. Two
matchers, one comment saying why, and classify_title untouched."
```

---

### Task 4: Per-page signals

**Files:**
- Modify: `src/datasheetindex/core/preamble.py`
- Test: `tests/test_preamble.py`

**Interfaces:**
- Consumes: `PreamblePage` (Task 1), `count_legal_hits` (Task 3).
- Produces: `PreamblePage.bullets`, `.legal_hits`, `.has_features_heading` now carry computed values. No signature changes.

Measured discrimination the signals must reproduce (spec section 4): a bulleted features page scores high `bullets`, zero `legal_hits`, and a `Features` heading; a legal cover page scores zero `bullets` and non-zero `legal_hits`. Unit density is deliberately **not** implemented — see the spec's section 5.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_preamble.py`:

```python
from datasheetindex.core.preamble import _page_signals


def test_signals_on_a_bulleted_features_page():
    """Glyphs go in as escapes -- the project bans literal Unicode in tests."""
    text = (
        "CY8C62x8\n"
        "General description\n"
        "The PSoC 6 MCU is a dual-core device.\n"
        "Features\n"
        "\u2022 32-bit Arm Cortex-M4F CPU at 150 MHz\n"
        "\u2022 2 MByte flash and 1 MByte SRAM\n"
        "- up to 102 programmable GPIOs\n"
        "\u25aa 12-bit 2-Msps SAR ADC\n"
    )
    signals = _page_signals(text)

    assert signals["bullets"] == 4
    assert signals["legal_hits"] == 0
    assert signals["has_features_heading"] is True


def test_signals_on_a_legal_cover_page():
    text = (
        "Product Change Notification\n"
        "TI requires acknowledgement of receipt of this notification "
        "within 30 days.\n"
        "TI makes no warranty and accepts no liability; see the trademark "
        "and copyright notices.\n"
    )
    signals = _page_signals(text)

    assert signals["bullets"] == 0
    assert signals["legal_hits"] >= 3
    assert signals["has_features_heading"] is False


def test_a_leading_hyphen_needs_whitespace_to_count_as_a_bullet():
    """Datasheets are full of temperature ranges; those are not bullets."""
    assert _page_signals("-40 to +85 degrees C\n")["bullets"] == 0
    assert _page_signals("- a real bullet\n")["bullets"] == 1


def test_features_heading_matches_a_whole_line_only():
    assert _page_signals("Features\n")["has_features_heading"] is True
    assert _page_signals("General Description:\n")["has_features_heading"] is True
    assert (
        _page_signals("Features of the analog subsystem\n")[
            "has_features_heading"
        ]
        is False
    )


def test_build_front_matter_populates_signals_per_page():
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(
        pymupdf.Rect(50, 50, 500, 300),
        "Features\n- first feature\n- second feature",
        fontsize=10,
    )
    page = doc.new_page()
    page.insert_textbox(
        pymupdf.Rect(50, 50, 500, 300),
        "We disclaim all warranty and liability.",
        fontsize=10,
    )
    fm = build_front_matter(doc)
    doc.close()

    assert fm.pages[0].bullets == 2
    assert fm.pages[0].has_features_heading is True
    assert fm.pages[0].legal_hits == 0
    assert fm.pages[1].bullets == 0
    assert fm.pages[1].legal_hits >= 2


def test_signals_reflect_the_whole_page_even_when_truncated():
    """Signals describe the page read, not the fragment shown."""
    doc = pymupdf.open()
    for _ in range(2):
        page = doc.new_page()
        writer = pymupdf.TextWriter(page.rect)
        y = 72
        for _ in range(40):
            writer.append((72, y), "- " + "A" * 60)
            y += 14
        writer.write_text(page)
    fm = build_front_matter(doc, max_chars=100)
    doc.close()

    assert fm.char_truncated is True
    assert fm.pages[1].bullets > 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_preamble.py -k signals -v`
Expected: FAIL with `ImportError: cannot import name '_page_signals'`.

- [ ] **Step 3: Implement the signals**

At the top of `src/datasheetindex/core/preamble.py`, add `import re` and the import of the matcher:

```python
import re

from datasheetindex.core.boilerplate import count_legal_hits
```

Add the constants below `DEFAULT_MAX_CHARS`:

```python
# A line opening with a bullet glyph, or with a dash *followed by whitespace*.
# The whitespace requirement is load-bearing: datasheets are full of lines like
# "-40 to +85" and a temperature range is not a feature bullet.
_BULLET_RE = re.compile(
    r"^\s*(?:[\u2022\u25aa\u25cb\u25e6\u2023\u00b7\u2219*]|[-\u2013\u2014]\s)"
)

# A heading, matched as a whole line so that "Features of the analog subsystem"
# does not count.
_FEATURES_HEADINGS = frozenset({"features", "general description"})
```

Add the helper above `build_front_matter`:

```python
def _page_signals(text: str) -> dict[str, object]:
    """Per-page evidence: bullet lines, legal vocabulary, a features heading.

    Reported, not acted on. The three signals separated a real datasheet's
    front matter from a product-change notice's cover letter cleanly and
    independently on both measured fixtures; a unit-density signal is
    deliberately left out because it is noisy in both directions (it misses
    "150-MHz" and "40 microamp", and false-positives on part numbers like
    "CY8C62x8/A").
    """
    lines = text.splitlines()
    return {
        "bullets": sum(1 for line in lines if _BULLET_RE.match(line)),
        "legal_hits": count_legal_hits(text),
        "has_features_heading": any(
            line.strip().rstrip(":").strip().lower() in _FEATURES_HEADINGS
            for line in lines
        ),
    }
```

In `build_front_matter`, build the `pages` list from the signals — note it uses `texts`, the full extracted page text, not the truncated `kept`:

```python
    pages = [
        PreamblePage(
            page=offset + 1,
            chars=len(page_text),
            **_page_signals(page_text),  # type: ignore[arg-type]
        )
        for offset, page_text in enumerate(texts)
    ]
```

If `ty` rejects the `**` splat, expand it explicitly instead:

```python
    pages = []
    for offset, page_text in enumerate(texts):
        signals = _page_signals(page_text)
        pages.append(
            PreamblePage(
                page=offset + 1,
                chars=len(page_text),
                bullets=int(signals["bullets"]),
                legal_hits=int(signals["legal_hits"]),
                has_features_heading=bool(signals["has_features_heading"]),
            )
        )
```

Prefer the explicit form if there is any type friction; it is the same length and does not need an ignore comment.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_preamble.py tests/test_boilerplate.py -v`
Expected: all PASS.

- [ ] **Step 5: Lint, format, type check**

Run: `uv run ruff format src tests && uv run ruff check src tests && uv run ty check src`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/datasheetindex/core/preamble.py tests/test_preamble.py
git commit -m "feat: report per-page front-matter signals

bullets, legal_hits and a features-heading flag, computed on the whole
extracted page rather than the fragment shown. Evidence for the agent, not
a skip heuristic in the library: wrongly dropping page 1 of a real
datasheet costs the general description and half the features, wrongly
keeping a cover page costs some tokens."
```

---

### Task 5: Wire it into the artifact

**Files:**
- Modify: `src/datasheetindex/index.py:30` (import), `:591-592` (call), `:735` (JSON)
- Test: `tests/test_index.py` (seven stub sites at lines 285, 389, 510, 592, 650, 743, 797)

**Interfaces:**
- Consumes: `build_front_matter`, `FrontMatter`, `PreamblePage.to_dict` from Tasks 1-4.
- Produces: a new additive top-level ToC JSON key `preamble_pages: list[dict]`, always present, one entry per page read, each `{"page", "chars", "bullets", "legal_hits", "has_features_heading"}`.

`index.py` calls `build_front_matter(doc)` with no keyword arguments, so the seven monkeypatch stubs stay one-argument lambdas.

**This is the one change in the plan that can quietly weaken existing tests.** The seven stubs patch `datasheetindex.index.generate_preamble`; once `index.py` calls `build_front_matter`, an unrepointed stub does not fail — it silently stops stubbing, and the test starts doing real extraction. Step 1's assertion that the stub value reaches the JSON is what makes that loud.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_index.py`:

```python
def test_preamble_pages_is_emitted(tmp_path):
    """One entry per page read, with the signal fields present."""
    pdf = tmp_path / "two.pdf"
    doc = pymupdf.open()
    for _ in range(2):
        page = doc.new_page()
        page.insert_textbox(
            pymupdf.Rect(50, 50, 500, 300),
            "Features\n- one\n- two",
            fontsize=10,
        )
    doc.save(str(pdf))
    doc.close()

    with DatasheetIndex(str(pdf)) as index:
        result = index.build(output_dir=str(tmp_path))
    data = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))

    assert [p["page"] for p in data["preamble_pages"]] == [1, 2]
    for entry in data["preamble_pages"]:
        assert set(entry) == {
            "page",
            "chars",
            "bullets",
            "legal_hits",
            "has_features_heading",
        }
    assert data["preamble"].startswith("--- PAGE 1 ---")


def test_single_page_document_yields_one_preamble_page(tmp_path):
    pdf = tmp_path / "one.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(
        pymupdf.Rect(50, 50, 500, 200), "Only page", fontsize=10
    )
    doc.save(str(pdf))
    doc.close()

    with DatasheetIndex(str(pdf)) as index:
        result = index.build(output_dir=str(tmp_path))
    data = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))

    assert len(data["preamble_pages"]) == 1
    assert "later pages were not examined" not in data["preamble"]
```

Match the existing file's fixture idiom — check how neighbouring tests in `tests/test_index.py` construct a PDF and call `build`, and follow that rather than the sketch above if it differs (some use a module-level helper, and `DatasheetIndex`/`json`/`Path`/`pymupdf` may already be imported).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_index.py -k preamble -v`
Expected: FAIL with `KeyError: 'preamble_pages'`.

- [ ] **Step 3: Switch `index.py` to `build_front_matter`**

Change the import at `src/datasheetindex/index.py:30`:

```python
from datasheetindex.core.preamble import build_front_matter
```

Change the call at `:591-592`:

```python
        # 2. Generate the page-marked front matter and its per-page signals
        front_matter = build_front_matter(doc)
        preamble = front_matter.text
```

Add the key immediately after `"preamble"` at `:735`:

```python
                "preamble": preamble,
                "preamble_pages": [p.to_dict() for p in front_matter.pages],
```

- [ ] **Step 4: Repoint the seven stubs**

In `tests/test_index.py`, add the import:

```python
from datasheetindex.core.preamble import FrontMatter
```

and replace each of the seven occurrences of

```python
    monkeypatch.setattr("datasheetindex.index.generate_preamble", lambda _doc: "pre")
```

with

```python
    monkeypatch.setattr(
        "datasheetindex.index.build_front_matter",
        lambda _doc: FrontMatter(
            text="pre",
            pages=[],
            chars_shown=3,
            chars_extracted=3,
            pages_read=1,
            total_pages=1,
        ),
    )
```

Then add, in the **first** of the seven stubbed tests, an assertion that the stub actually takes effect (place it beside that test's existing assertions on the emitted JSON, using whatever variable it already holds the parsed JSON in):

```python
    # A stub that stops taking effect must fail here rather than pass quietly.
    assert data["preamble"] == "pre"
    assert data["preamble_pages"] == []
```

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest 2>&1 | tee /tmp/preamble-suite.log`
Expected: all PASS. If a test fails on preamble content, check it is not one of the seven that lost its stub. `grep -n "generate_preamble" tests/test_index.py` must return nothing.

- [ ] **Step 6: Lint, format, type check**

Run: `uv run ruff format src tests && uv run ruff check src tests && uv run ty check src`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/datasheetindex/index.py tests/test_index.py
git commit -m "feat: emit preamble_pages in the ToC JSON

index.py switches to build_front_matter; preamble_pages is a new additive
top-level key, always present, one entry per page read. The seven stubs in
test_index.py are repointed -- an unrepointed one would not fail, it would
silently stop stubbing, so one test now asserts the stub value reaches the
emitted JSON."
```

---

### Task 6: Sanity-check the budget, then document and release

**Files:**
- Create (throwaway, not committed): `/tmp/claude-1000/-home-yeqi-projects-datasheetindex/81b3687b-ea4e-4352-88ba-c59c4ee4cc5b/scratchpad/measure_front_matter.py`
- Modify: `CHANGELOG.md`, `pyproject.toml:3`, `README.md:13`, `README.md:465`, `docs/datasheetindex_architecture.md` (around lines 65-80), `CLAUDE.md`
- Delete: `docs/superpowers/specs/2026-07-25-preamble-front-matter-design.md`

**Interfaces:**
- Consumes: `build_front_matter` from Tasks 1-4, `DatasheetIndex` (unchanged).
- Produces: no code interfaces. Version 0.26.0.

The spec asks for the 5000-character default to be sanity-checked against a wider corpus before release: "The default is chosen from two documents". Do the measurement, record the numbers in the CHANGELOG, and only change the default if a document is materially cut.

- [ ] **Step 1: Measure the default against every PDF on hand**

Write the script:

```python
"""Report front-matter size against the current default budget."""

import sys
from pathlib import Path

import pymupdf

from datasheetindex.core.preamble import DEFAULT_MAX_CHARS, build_front_matter

for arg in sys.argv[1:]:
    path = Path(arg)
    if not path.exists():
        print(f"MISSING {path}")
        continue
    doc = pymupdf.open(str(path))
    fm = build_front_matter(doc)
    third = len(doc) >= 3 and len(build_front_matter(doc, max_pages=3).text)
    doc.close()
    print(
        f"{path.name[:44]:44s} pages={fm.total_pages:4d} "
        f"extracted={fm.chars_extracted:5d} shown={fm.chars_shown:5d} "
        f"cut={fm.char_truncated} framing={len(fm.text) - fm.chars_shown:4d}"
    )
print(f"default max_chars={DEFAULT_MAX_CHARS}")
```

Run it against everything available locally:

```bash
uv run python <scratchpad>/measure_front_matter.py \
  ../data2page/*.pdf \
  /tmp/infineon-psoc-6-mcu-cy8c62x8-cy8c62xa-datasheet-datasheet-en.pdf \
  /tmp/ti_202510080021_10132025_.pdf \
  2>&1 | tee /tmp/front-matter-budget.log
```

Expected: the PSoC 6 reports `extracted=4747`-ish with `cut=False` — the whole point of the raise. Record the actual table; it goes in the CHANGELOG. If a document reports `cut=True` at 5000, note it and leave the default alone anyway (the caller can raise it, and the note now discloses the cut) unless more than half the corpus is cut, in which case raise the default and say so.

- [ ] **Step 2: Bump the version**

In `pyproject.toml:3`: `version = "0.26.0"`.

The version bump is load-bearing beyond release hygiene: `core/artifact_cache.py` gates reuse on exact `datasheetindex_version` equality (`artifact_cache.py:278`), so every artifact built before this change is invalidated automatically and no stale JSON without `preamble_pages` is ever served.

- [ ] **Step 3: Write the CHANGELOG entry**

Add at the top of `CHANGELOG.md`, below the header, matching the existing entries' density (they state measurements, not intentions):

```markdown
## [0.26.0] - 2026-07-26

### Changed
- **The `preamble` is now page-marked, larger by default, and says what it dropped.** It carries `--- PAGE N ---` markers in the same format as the page-matched text file, so every line is attributable and citable; `generate_preamble` gains `max_pages` alongside `max_chars`, both keyword-only on the new `build_front_matter`, and the character default rises from 2400 to 5000. The old default was not merely small, it kept the wrong half: measured on the PSoC 6 (CY8C62x8), pages 1-2 hold 4747 characters and 2385 were emitted, and because a character budget applied in document order retains whatever sits earliest, the emitted text ended on page 1's legal footer while page 2's 13 serial communication blocks, 32 TCPWMs, 12-bit 2-Msps SAR ADC and "up to 102 programmable GPIOs" were dropped. Both documents to hand hit the cap, so truncation was the normal case, not an edge case. Measured against the local corpus at the new default: [insert the Step 1 table].
- **Truncation is disclosed rather than silent**, per the 0.18.0 principle that silence is not a completeness claim. Each cap gets its own `=== NOTE: ... ===` line because the extractor's knowledge differs between them: `max_chars` knows the total on the pages it read (`preamble truncated at 5000 characters; 5000 of 7118 characters from pages 1-2 shown, ending mid-page on page 2`), while `max_pages` knows only that later pages exist (`preamble covers pages 1-2 of 134; later pages were not examined`) -- it deliberately does not claim front matter continues, or count characters it never extracted, since reading one page past the limit to describe what was skipped would defeat the limit. Both fire when both caps bite, character note first.
- **Compatibility: `max_chars` now bounds document text, not the returned string.** Markers and notes are tool framing; counting them against the budget would make the amount of *content* a caller receives depend on how long the framing happens to be, and it is circular besides, since a note names the truncation point. The framing is exact after the fact (`len(text) - chars_shown`, both returned) and, for markers alone, exact before it: `sum(13 + digits(n) for n in 1..P) + (2 * P - 1)`, which is 31 at `max_pages=2`. Note lines add roughly 120 characters each -- an estimate, not a bound. Two existing tests asserted the old whole-string guarantee and are rewritten against `chars_shown`. Page text is also no longer `rstrip`ped, which is what makes the marker formula hold exactly; trailing whitespace before a marker is invisible in practice. A consumer treating the preamble as opaque prose is unaffected; one parsing it line by line now sees marker lines. `datasheet-agent` was checked and does not parse it.

### Added
- **A new additive top-level ToC JSON key, `preamble_pages`**, one entry per page read: `{"page", "chars", "bullets", "legal_hits", "has_features_heading"}`. It exists because front matter is not always front matter -- a product change notification's page 1 is a cover letter, and measured on TI's PCN 20251008002.1 the entire preamble is "TI requires acknowledgement of receipt of this notification within 30 days..." with zero specifications, which nothing previously disclosed. The signals separate the two cases cleanly and independently on both fixtures: TI PCN p1 scores 0 bullets / 4 legal hits / no features heading, while PSoC 6 p1 and p2 score 34 and 43 bullets / 0 legal hits / a features heading each. Signals describe the whole page read, not the fragment shown, so truncation cannot skew them.
- **`build_front_matter(doc, *, max_pages=2, max_chars=5000) -> FrontMatter`** in `core/preamble.py`, returning `text`, `pages`, `chars_shown`, `chars_extracted`, `pages_read`, `total_pages`, plus `char_truncated` and `pages_omitted` properties. There is deliberately no single `truncated` flag: the two caps are independent, so one boolean could not tell a caller whether to raise `max_chars`, `max_pages`, or both. The NOTE lines are rendered from these same fields, so prose and structure cannot disagree. `generate_preamble` remains as a wrapper returning `.text`.
- **`count_legal_hits()` in `core/boilerplate.py`**, an unanchored prose matcher over a new `_LEGAL_VOCABULARY` constant. It is a second matcher, not a shared one: the existing `legal` title pattern is anchored `^(...)$` and scores **zero** on a footer sentence, and several of its branches deliberately require a qualifier (`product liability`, `important notices`) because bare `liability` and `information` are common substantive section titles. In running prose that judgement inverts -- a bare "liability" in a footer sentence is exactly the signal -- so one list serving both would either weaken the title matcher, which publishes a flag in the artifact, or under-count prose. `classify_title` is untouched and every one of its tests passes unmodified.

### Removed
- Nothing. No key was removed and no signature changed incompatibly.
```

Replace `[insert the Step 1 table]` with the real numbers from Step 1 before committing. Leaving the placeholder in is a failure.

- [ ] **Step 4: Reject the two things a future reader will try to re-add**

Both are already argued in the spec; carry the short form into `docs/datasheetindex_architecture.md` so it survives the spec's deletion. In the preamble section (around `docs/datasheetindex_architecture.md:65-80`), after the existing "The agent IS the LLM" paragraph, add:

```markdown
#### Decisions already settled by measurement

**Skipping a cover or legal page is rejected.** Detecting front matter that
is not front matter and dropping it was considered. The error is asymmetric:
wrongly skipping page 1 of a real datasheet costs the general description and
half the features -- the most valuable page in the document -- while wrongly
keeping a cover page costs some tokens. Two documents is also not a corpus to
calibrate against. So the library reports `preamble_pages` signals and the
agent decides; a caller given the signals can implement skipping, but a
library that skips forecloses the alternative. This is the same shape of
decision as the table-engine note in `CLAUDE.md`: stability is the point.

**Unit density is deliberately not a signal.** A count of numeric-plus-unit
tokens looks like the obvious fourth signal. A naive ASCII pattern undercounts
badly -- 5 matches on PSoC page 1 against 30 for a corrected one -- because it
misses `150-MHz` (hyphen separator), `1.1-V`, and `40 uA` (micro sign, which
needs both U+00B5 and U+03BC). The corrected pattern then false-positives on
part numbers, which datasheets are full of: `8/A` from `CY8C62x8/A`, `4F` from
`Cortex-M4F`. Noisy in both directions, and the three shipped signals already
discriminate perfectly on both fixtures. Add it later if a consumer needs it,
calibrated against part-number forms.
```

Also update the sample JSON at `docs/datasheetindex_architecture.md:80` so its `preamble` value starts with `--- PAGE 1 ---\n`, and the code sketch at `:681-682` and `:741` to call `build_front_matter` and emit `preamble_pages`, so the doc does not describe a call that no longer exists.

- [ ] **Step 5: Update README and CLAUDE.md**

`README.md:13` — the deliverable summary: change `a preamble (pages 1-2 raw text) for agent orientation` to `a page-marked preamble (pages 1-2 raw text, with per-page signals in preamble_pages) for agent orientation`.

`README.md:465` — the file tree comment: change `# Pages 1-2 raw text extraction` to `# Page-marked front matter + per-page signals`.

`CLAUDE.md` — delete the whole **"Known pending work"** section. It documents exactly this defect, and it is now false. If the Step 1 measurement found any document still cut at 5000, replace the section with a two-line note naming that document and its character count instead of deleting it outright.

- [ ] **Step 6: Delete the spec**

```bash
git rm docs/superpowers/specs/2026-07-25-preamble-front-matter-design.md
```

`CLAUDE.md` states the rule that makes this correct: the spec "is kept **because** it is unbuilt ... its presence there means 'not done', not 'how it works'." Its lasting rationale moved into the architecture doc in Step 4 and the CHANGELOG in Step 3, both of which are tracked on both remotes.

- [ ] **Step 7: Verify everything before claiming done**

Run:

```bash
uv run pytest 2>&1 | tee /tmp/preamble-final.log
uv run ruff format --check src tests && uv run ruff check src tests
uv run ty check src
uv run pre-commit run --all-files
grep -rn "generate_preamble" src/ tests/ docs/ README.md
```

Expected: full suite green; lint, format and types clean; pre-commit green; the `grep` shows `generate_preamble` only in `core/preamble.py` (definition and wrapper) and in `tests/test_preamble.py` (the wrapper test) — no doc still describing it as the entry point.

- [ ] **Step 8: Confirm the artifact by eye once**

```bash
uv run datasheetindex build ../data2page/Infineon-TLE9350BSJ-DataSheet-v01_00-EN.pdf \
  --output-dir /tmp/fm-check
uv run python -c "
import json
d = json.load(open('/tmp/fm-check/Infineon-TLE9350BSJ-DataSheet-v01_00-EN.json', encoding='utf-8'))
print(d['preamble'][:400])
print('...')
print(d['preamble'][-300:])
print(d['preamble_pages'])
"
```

Confirm the CLI flag name against `uv run datasheetindex build --help` and the JSON filename against the directory listing before running the second command. Expected: the text opens with `--- PAGE 1 ---`, ends with the page NOTE (the TLE9350 has more than 2 pages), and `preamble_pages` holds two entries with plausible signal counts.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "docs: page-marked front matter, and 0.26.0

Records the measured budget check behind max_chars=5000, moves the two
settled rejections (page skipping, unit density) into the architecture
doc, and deletes the spec -- which was on disk only to mean 'not built'."
```

---

## Notes for the implementer

**Do not add**, however tempting:
- A `truncated: bool` on `FrontMatter`. It cannot be given an honest meaning; see Task 1's docstring.
- Any skip, reorder, or classification of pages. Report evidence; do not act on it.
- A unit-density signal. Rejected on measurement; see Task 6 Step 4.
- Changes to the page-matched text file. It already carries markers.
- Anything in `datasheet-agent`. The preamble-based axis strategy is filed there as its issue #1 and is a consumer of this, not part of it.

**Release, when the work is merged on `main`:** follow `CLAUDE.md`'s publishing workflow — `git switch gitlab-main && git merge main && git push gitlab gitlab-main:main`, then `git tag -a v0.26.0` and `git push gitlab v0.26.0`. The tag is the only release trigger, and never republish a version.
