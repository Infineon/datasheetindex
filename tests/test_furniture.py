"""Unit tests for running header/footer decision logic."""

from __future__ import annotations

from datasheetindex.core.furniture import (
    MAX_FURNITURE_CHARS,
    detect_furniture,
    furniture_threshold,
    has_lexical_evidence,
    is_candidate,
    normalize_key,
)


def test_normalize_key_collapses_whitespace_and_masks_digits():
    assert normalize_key("  Datasheet   46 \n 002-23185 Rev. *S  ") == (
        "Datasheet # #-# Rev. *S"
    )


def test_normalize_key_keeps_letters_distinct():
    """Masking digits must not make two different headers compare equal."""
    assert normalize_key("Chapter 3: Timers") != normalize_key("Chapter 4: Serial")


def test_normalize_key_of_blank_text_is_empty():
    assert normalize_key("   \n  ") == ""


def test_is_candidate_accepts_a_short_multi_line_block():
    """The PSoC footer is ONE block of four short lines.

    An earlier design excluded blocks of three or more lines, copying
    PageIndex. Measured, that discards this exact footer on 132 of 134
    pages -- the majority of the furniture the feature exists to remove.
    This test pins the removed rule so it cannot be reintroduced.
    """
    footer = "Datasheet\n46\n002-23185 Rev. *S\n2025-11-06"
    assert is_candidate(footer) is True


def test_is_candidate_rejects_long_blocks():
    assert is_candidate("x" * (MAX_FURNITURE_CHARS + 1)) is False
    assert is_candidate("x" * MAX_FURNITURE_CHARS) is True


def test_is_candidate_rejects_caption_prefixes():
    for caption in (
        "Table 43 (continued) USB specifications",
        "table 8 alternate functions",
        "Figure 12. Block diagram",
        "Fig. 3 timing",
        "Chart 2",
    ):
        assert is_candidate(caption) is False, caption


def test_is_candidate_does_not_reject_words_merely_starting_with_a_prefix():
    """'Tables' is not the caption keyword 'Table'; the boundary matters."""
    assert is_candidate("Tablet computer interface") is True


def test_is_candidate_rejects_blank_text():
    assert is_candidate("   \n ") is False


def test_furniture_threshold_uses_the_page_fraction():
    assert furniture_threshold(134) == 67
    assert furniture_threshold(42) == 21
    assert furniture_threshold(25) == 13  # ceil(12.5)


def test_furniture_threshold_floor_protects_short_documents():
    """A 1- or 2-page document can never reach the floor, so never strips."""
    assert furniture_threshold(1) == 3
    assert furniture_threshold(2) == 3
    assert furniture_threshold(0) == 3


def test_detect_furniture_counts_each_key_once_per_page():
    """A key repeated within one page counts once, not twice."""
    page_keys = [["hdr", "hdr"], ["hdr"], ["hdr"], ["other"]]
    assert detect_furniture(page_keys, total_pages=4) == frozenset({"hdr"})


def test_detect_furniture_requires_the_threshold():
    page_keys = [["a"], ["a"], ["b"], ["b"], ["b"], ["b"]]
    # threshold = max(3, ceil(0.5 * 6)) = 3; "a" has 2, "b" has 4.
    assert detect_furniture(page_keys, total_pages=6) == frozenset({"b"})


def test_detect_furniture_on_a_two_page_document_finds_nothing():
    page_keys = [["hdr"], ["hdr"]]
    assert detect_furniture(page_keys, total_pages=2) == frozenset()


def test_detect_furniture_on_no_pages_is_empty():
    assert detect_furniture([], total_pages=0) == frozenset()


def test_detect_furniture_ignores_keys_with_no_letters():
    """A digit-masked key carrying no letters is content, not furniture.

    A bare page-number footer normalizes to ``#`` on every page, which clears
    the threshold trivially. Treating it as furniture then deletes every
    bare-number block in either band -- genuine numeric table rows -- because
    they normalize to the same key. The alphabetic key on the same input must
    still be detected, or this test would also pass on a detector that found
    nothing at all.
    """
    page_keys = [["#", "#.# #.#", "Datasheet # Rev. B"] for _ in range(8)]
    assert detect_furniture(page_keys, total_pages=8) == frozenset(
        {"Datasheet # Rev. B"}
    )


def test_detect_furniture_accepts_a_key_whose_letters_are_not_latin():
    """``str.isalpha`` rather than ``[A-Za-z]``: any script is evidence.

    Written as escapes to keep this file ASCII. The key is Cyrillic
    "Stranitsa #" -- a page-number footer in a Russian-language document,
    which an ``[A-Za-z]`` class would reject as evidence-free and delete.
    """
    key = "\u0421\u0442\u0440\u0430\u043d\u0438\u0446\u0430 #"
    page_keys = [[key] for _ in range(6)]
    assert detect_furniture(page_keys, total_pages=6) == frozenset({key})


def test_has_lexical_evidence_separates_masked_numbers_from_text():
    assert has_lexical_evidence("#") is False
    assert has_lexical_evidence("#.# #.#") is False
    assert has_lexical_evidence("- # -") is False
    assert has_lexical_evidence("") is False
    # A real survey key: every one of the 16 measured keys has letters.
    assert has_lexical_evidence("# V#.# #-#-#") is True
