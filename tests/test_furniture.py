"""Unit tests for running header/footer decision logic."""

from __future__ import annotations

from datasheetindex.core.furniture import (
    MAX_FURNITURE_CHARS,
    PARITY_DOMINANCE,
    detect_furniture,
    furniture_threshold,
    has_lexical_evidence,
    is_candidate,
    normalize_key,
)


def _split_pages(total_pages, odd_count, even_count, key="ACME AWC-# Controller"):
    """Page keys putting ``key`` on N odd-numbered and M even-numbered pages.

    Index 0 is page 1, so the "odd" pages are the even indices. Every page
    also carries a page-unique key, so no test can pass merely because the
    document is empty.
    """
    pages = []
    used = [0, 0]
    wanted = (odd_count, even_count)
    for index in range(total_pages):
        keys = [f"body page {index}"]
        parity = index % 2
        if used[parity] < wanted[parity]:
            used[parity] += 1
            keys.append(key)
        pages.append(keys)
    assert used == [odd_count, even_count]
    return pages


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


def test_detect_furniture_accepts_a_key_dominating_one_parity():
    """The alternating odd/even header, which no overall count can reach.

    Modelled on ``micro_atmega328.pdf``, where four genuine furniture keys sit
    at exactly 146 of 294 pages -- one page under the overall threshold -- and
    the document strips nothing without this route. The counts here are built
    so the OVERALL route genuinely fails: 9 of 20 pages against a threshold of
    10. Were that not so, the test would pass through the old route and prove
    nothing about the new one.
    """
    key = "ACME AWC-# Controller"
    page_keys = _split_pages(20, odd_count=9, even_count=0, key=key)
    assert furniture_threshold(20) == 10  # and the key is on only 9 pages
    assert detect_furniture(page_keys, total_pages=20) == frozenset({key})


def test_detect_furniture_rejects_a_key_spread_across_both_parities():
    """Dominance is the point: a bare parity threshold would admit this.

    10 odd and 9 even of 40 pages is 19 -- under the overall threshold of 20 --
    while 10 clears the parity bucket's own threshold of 10. Without the
    dominance rule this key would qualify, and with it any key on roughly a
    quarter of a document would, which is the unreviewed loosening the 0.5
    fraction was measured to reject.
    """
    key = "Register description"
    page_keys = _split_pages(40, odd_count=10, even_count=9, key=key)
    assert furniture_threshold(40) == 20
    assert furniture_threshold(20) == 10  # the per-bucket threshold it clears
    assert detect_furniture(page_keys, total_pages=40) == frozenset()


def test_detect_furniture_rejects_an_uneven_recurrence():
    """The accepted price, pinned with the real numbers that pay it.

    ``www.ti.com`` on ``ti_lm358.pdf`` is 22 pages of one parity and 8 of the
    other, out of 68. It is genuine furniture, it clears the parity bucket's
    threshold of 17, and dominance declines it anyway: 8 against 22 is an
    uneven recurrence, not an alternating layout. Admitting it would mean
    admitting every similarly uneven key.
    """
    key = "www.ti.com"
    page_keys = _split_pages(68, odd_count=22, even_count=8, key=key)
    assert furniture_threshold(68) == 34
    assert furniture_threshold(34) == 17  # cleared, and dominance still rejects
    assert detect_furniture(page_keys, total_pages=68) == frozenset()


def test_detect_furniture_dominance_boundary_is_inclusive():
    """``other <= PARITY_DOMINANCE * here``, not ``<``.

    At exactly the ratio the key qualifies. Pinning the boundary keeps a later
    refactor from silently tightening or loosening the rule by one page.
    """
    key = "ACME AWC-# Controller"
    here, other = 10, int(PARITY_DOMINANCE * 10)
    page_keys = _split_pages(40, odd_count=here, even_count=other, key=key)
    assert detect_furniture(page_keys, total_pages=40) == frozenset({key})

    over = _split_pages(40, odd_count=here, even_count=other + 1, key=key)
    assert detect_furniture(over, total_pages=40) == frozenset()


def test_detect_furniture_parity_route_still_requires_letters():
    """The letter requirement applies to BOTH routes.

    A bare page-number footer alternating by parity normalizes to ``#`` and
    dominates its bucket perfectly. Admitting it via the new route would
    reinstate exactly the content deletion ``has_lexical_evidence`` exists to
    prevent. The lettered key on the same pages must still be found, or a
    detector that returned nothing at all would also pass.
    """
    key = "ATmega#P [DATASHEET]"
    page_keys = _split_pages(20, odd_count=9, even_count=0, key=key)
    for index in range(0, 20, 2):
        page_keys[index].append("#")
    assert detect_furniture(page_keys, total_pages=20) == frozenset({key})


def test_detect_furniture_parity_route_respects_the_min_pages_floor():
    """A tiny document cannot produce furniture through the parity route.

    On a 4-page document a parity bucket holds 2 pages, so a key on every page
    of one parity still has only 2 pages of evidence against the floor of 3.
    Without the floor applying per bucket, a 4-page document would strip on
    two matching pages.
    """
    key = "ACME AWC-# Controller"
    for pages in (2, 4):
        page_keys = _split_pages(
            pages, odd_count=(pages + 1) // 2, even_count=0, key=key
        )
        assert detect_furniture(page_keys, total_pages=pages) == frozenset(), pages


def test_has_lexical_evidence_separates_masked_numbers_from_text():
    assert has_lexical_evidence("#") is False
    assert has_lexical_evidence("#.# #.#") is False
    assert has_lexical_evidence("- # -") is False
    assert has_lexical_evidence("") is False
    # A real survey key: every one of the 16 measured keys has letters.
    assert has_lexical_evidence("# V#.# #-#-#") is True
