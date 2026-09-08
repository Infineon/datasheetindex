"""Tests for the token-economy measurement script.

The script lives in ``scripts/`` rather than in the package: it measures the
library, it is not part of it, and shipping it in the wheel would add public
surface for something only the docs consume. That is why this module loads it
by path instead of importing it.
"""

import importlib.util
import statistics
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "token_economy.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("token_economy", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["token_economy"] = module
    spec.loader.exec_module(module)
    return module


te = _load_module()


def _words(text: str) -> int:
    """A deterministic stand-in for a real BPE encoder."""
    return len(text.split())


# --------------------------------------------------------------------------
# Leaf-section walk
# --------------------------------------------------------------------------


def test_leaf_sections_yields_only_childless_nodes():
    toc = [
        {
            "title": "Overview",
            "start_page": 1,
            "end_page": 4,
            "nodes": [
                {"title": "Features", "start_page": 1, "end_page": 2},
                {"title": "Block diagram", "start_page": 3, "end_page": 4},
            ],
        },
        {"title": "Electrical Characteristics", "start_page": 5, "end_page": 9},
    ]
    leaves = list(te.iter_leaf_sections(toc))
    assert [leaf["title"] for leaf in leaves] == [
        "Features",
        "Block diagram",
        "Electrical Characteristics",
    ]


def test_leaf_sections_skips_nodes_without_a_usable_page_range():
    """A node whose range is absent or inverted cannot be read back.

    ``end_page`` defaults to 0 in ``TocNode``, so an un-enriched tree
    serializes leaves the reader would reject. Dropping them here keeps the
    measurement from counting a section it never actually read.
    """
    toc = [
        {"title": "Good", "start_page": 2, "end_page": 3},
        {"title": "No end page", "start_page": 2, "end_page": 0},
        {"title": "Inverted", "start_page": 7, "end_page": 4},
    ]
    assert [leaf["title"] for leaf in te.iter_leaf_sections(toc)] == ["Good"]


@pytest.mark.parametrize(
    ("size", "expected_index"),
    [
        # Nearest-rank p90 is ceil(0.9 * n). `round` is banker's rounding, so
        # it answers 4 for n=5 (an 80th percentile) while agreeing with ceil
        # at n=15 -- a silent, size-dependent understatement in a published
        # column.
        (5, 5),
        (15, 14),
        (1, 1),
        (10, 9),
    ],
)
def test_percentile_is_nearest_rank_at_every_size(size, expected_index):
    values = list(range(1, size + 1))
    assert te._percentile(values, 0.9) == expected_index


def test_percentile_of_nothing_is_zero():
    assert te._percentile([], 0.9) == 0


def test_leaf_sections_handles_an_empty_toc():
    assert list(te.iter_leaf_sections([])) == []


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def _measurement(**overrides):
    base = dict(
        name="ds.pdf",
        pages=10,
        full_tokens=10_000,
        manifest_tokens=500,
        section_tokens=[100, 200, 300],
        cold_build_s=2.0,
        warm_build_s=0.1,
    )
    base.update(overrides)
    return te.DocumentMeasurement(**base)


def test_measurement_derives_median_p90_and_ratio():
    m = _measurement(section_tokens=[100, 200, 300])
    assert m.section_median == 200
    assert m.section_p90 == 300
    # The first answer is the manifest the agent plans from plus the one
    # section it reads -- not the document.
    assert m.answer_tokens == 700
    assert m.ratio == pytest.approx(10_000 / 700)


def test_measurement_prices_a_follow_up_question_without_the_manifest():
    """The manifest is paid once per document, not once per question.

    Pricing every question at ``manifest + section`` understates the library
    on exactly the workload it is for -- an agent asking several things about
    one part, with the map already in context and the artifact already built.
    """
    m = _measurement(section_tokens=[100, 200, 300])
    assert m.followup_tokens == 200
    assert m.followup_ratio == pytest.approx(10_000 / 200)


def test_a_document_with_no_sections_prices_no_follow_up_either():
    m = _measurement(section_tokens=[])
    assert m.followup_tokens is None
    assert m.followup_ratio is None


def test_measurement_with_no_readable_sections_has_no_ratio():
    """A document with no usable ToC has no section read to price.

    Reporting a ratio of ``full / manifest`` there would flatter us: the agent
    would fall back to ``search_text``, which this script does not measure.
    """
    m = _measurement(section_tokens=[])
    assert m.section_median == 0
    assert m.answer_tokens is None
    assert m.ratio is None


def test_summarize_reports_the_median_ratio_across_documents():
    measurements = [
        _measurement(
            name="a.pdf", full_tokens=10_000, manifest_tokens=500, section_tokens=[500]
        ),
        _measurement(
            name="b.pdf",
            full_tokens=90_000,
            manifest_tokens=4_000,
            section_tokens=[1_000],
        ),
    ]
    summary = te.summarize(measurements)
    assert summary["documents"] == 2
    assert summary["priced"] == 2
    ratios = [10_000 / 1_000, 90_000 / 5_000]
    assert summary["median_ratio"] == pytest.approx(statistics.median(ratios))
    assert summary["min_ratio"] == pytest.approx(min(ratios))
    assert summary["max_ratio"] == pytest.approx(max(ratios))
    followups = [10_000 / 500, 90_000 / 1_000]
    assert summary["median_followup_ratio"] == pytest.approx(
        statistics.median(followups)
    )


def test_summarize_ignores_unpriced_documents_but_still_counts_them():
    measurements = [
        _measurement(
            name="a.pdf", full_tokens=10_000, manifest_tokens=500, section_tokens=[500]
        ),
        _measurement(name="b.pdf", section_tokens=[]),
    ]
    summary = te.summarize(measurements)
    assert summary["documents"] == 2
    assert summary["priced"] == 1
    assert summary["median_ratio"] == pytest.approx(10.0)


def test_summarize_of_nothing_is_empty_rather_than_an_error():
    summary = te.summarize([])
    assert summary["documents"] == 0
    assert summary["priced"] == 0
    assert summary["median_ratio"] is None


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def test_render_markdown_lists_every_document_and_the_headline():
    measurements = [
        _measurement(
            name="a.pdf", full_tokens=10_000, manifest_tokens=500, section_tokens=[500]
        ),
        _measurement(name="b.pdf", section_tokens=[]),
    ]
    text = te.render_markdown(measurements, te.summarize(measurements))
    assert "a.pdf" in text
    assert "b.pdf" in text
    # The unpriced document must be visibly unpriced, not silently absent or
    # rendered as a zero.
    assert "n/a" in text
    assert "10.0" in text
    # Both prices, because they answer different questions.
    assert "first" in text.lower()
    assert "further" in text.lower()


def test_render_markdown_records_documents_that_failed_to_measure():
    """A truncated table must say it is truncated.

    The documented command writes straight to ``docs/token-economy.md``, so a
    run against a half-fetched corpus would otherwise replace the published
    results with a quietly shorter table.
    """
    m = _measurement(section_tokens=[500])
    summary = te.summarize([m])
    text = te.render_markdown([m], summary, failures=["broken.pdf: not a PDF"])
    assert "broken.pdf" in text
    assert "not a PDF" in text


def test_render_markdown_reproduces_the_flags_the_run_actually_used():
    """The emitted command must regenerate *this* table.

    Without ``--allow-editable-reuse`` a checkout reports no warm timing at
    all, so a block omitting it cannot reproduce a table showing cache hits.
    """
    m = _measurement(section_tokens=[500])
    summary = te.summarize([m])
    text = te.render_markdown([m], summary, extra_flags=["--allow-editable-reuse"])
    assert "--allow-editable-reuse" in text


def test_render_markdown_states_what_the_numbers_do_not_claim():
    """The caveats travel with the table, not only in the README.

    A generated file is the one a reader lands on from a link, and a ratio
    with no stated baseline is the easiest number in this repository to
    misread.
    """
    m = _measurement(section_tokens=[500])
    text = te.render_markdown([m], te.summarize([m]))
    lowered = text.lower()
    # It is a text-only baseline, so it understates what a PDF really costs.
    assert "conservative" in lowered
    # And it says nothing about whether the answer is right.
    assert "accuracy" in lowered or "correct" in lowered
    # The build timings exclude figure captioning, which is on by default.
    assert "caption" in lowered
    # The two token medians are medians in their own right and do not divide
    # to the ratio; a reader must not be invited to check that arithmetic.
    assert "median of the per-document ratios" in lowered


# --------------------------------------------------------------------------
# End to end, against a real build
# --------------------------------------------------------------------------


def test_measure_document_prices_a_real_build(toc_pdf, tmp_path):
    out = tmp_path / "artifacts"
    m = te.measure_document(toc_pdf, encode=_words, output_dir=str(out))

    assert m.name == toc_pdf.name
    assert m.pages == 3
    assert m.full_tokens > 0
    assert m.manifest_tokens > 0
    # The fixture's two bookmarks are both leaves.
    assert len(m.section_tokens) == 2
    assert all(count > 0 for count in m.section_tokens)
    assert m.cold_build_s >= 0
    # The warm build reuses the on-disk artifact, so it is measured through a
    # fresh instance -- an in-memory hit would report a meaningless ~0.
    assert m.warm_build_s >= 0


@pytest.fixture
def not_editable(monkeypatch):
    """Force the editability probe False.

    Reuse is disabled on an editable install by design, and the suite runs
    from an editable checkout -- so without this the warm pass measures that
    rule rather than the cache. ``test_reuse.py`` carries the same fixture for
    the same reason.
    """
    monkeypatch.setattr("datasheetindex.tools.bound.is_editable_install", lambda: False)


def test_measure_document_reports_the_warm_build_as_a_cache_hit(
    toc_pdf, tmp_path, not_editable
):
    """The warm number means nothing unless it really was a reuse.

    The cold pass forces a rebuild, so the warm pass is the only one that can
    hit the cache -- and it is run through a *fresh* instance, because
    ``build_datasheet`` short circuits in memory and would report a
    meaningless ~0 otherwise. ``warm_reused`` is how the rendered table can
    say so rather than asking the reader to trust the timing.
    """
    out = tmp_path / "artifacts"
    m = te.measure_document(toc_pdf, encode=_words, output_dir=str(out))
    assert m.warm_reused is True
    assert m.notes == []


def test_measure_document_says_so_when_the_warm_build_was_not_a_hit(
    toc_pdf, tmp_path, monkeypatch
):
    """State the rule; do not merely observe that this checkout is editable.

    ``test_reuse.py::test_an_editable_install_never_reuses`` says the same
    thing in the same words. Leaning on the ambient install would make this
    test pass for the environment's reason and fail against a wheel install,
    where reuse is enabled and the warm pass really is a hit.

    The number must degrade to a stated ``n/a``, never to a fast-looking
    timing the table would present as a cache hit.
    """
    monkeypatch.setattr("datasheetindex.tools.bound.is_editable_install", lambda: True)
    out = tmp_path / "artifacts"
    m = te.measure_document(toc_pdf, encode=_words, output_dir=str(out))
    assert m.warm_reused is False
    assert any("not a cache hit" in note for note in m.notes)


def test_render_markdown_shows_no_warm_timing_for_a_missed_cache():
    m = _measurement(section_tokens=[500], warm_build_s=0.05, warm_reused=False)
    text = te.render_markdown([m], te.summarize([m]))
    assert "0.05s" not in text
