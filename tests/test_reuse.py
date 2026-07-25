"""Tests for on-disk and in-memory artifact reuse."""

import pymupdf
import pytest

from datasheetindex.index import TOC_FALLBACK_THRESHOLD, DatasheetIndex
from datasheetindex.models import TocNode, TocQuality
from datasheetindex.tools.bound import DatasheetTools


@pytest.fixture
def toc_pdf(tmp_path):
    """A PDF whose ToC quality clears TOC_FALLBACK_THRESHOLD.

    A synthetic PDF with no bookmarks scores 0.00 against a threshold of 0.3,
    so the fallback is eligible, CI has no credentials, no client can be
    created, and the build is marked llm_enrichment_incomplete -- which makes
    the document permanently uncacheable and every reuse-hit test unpassable.
    Two set_toc entries on three pages score 0.62.
    """
    from datasheetindex.core.quality import assess_toc_quality
    from datasheetindex.core.structure import build_tree, extract_toc

    doc = pymupdf.open()
    for _ in range(3):
        page = doc.new_page()
        writer = pymupdf.TextWriter(page.rect)
        writer.append((72, 72), "Body text for this page of the datasheet")
        writer.write_text(page)
    doc.set_toc([[1, "Overview", 1], [1, "Electrical Characteristics", 2]])
    pdf_path = tmp_path / "ds.pdf"
    doc.save(str(pdf_path))

    nodes = build_tree(extract_toc(doc), len(doc))
    score = assess_toc_quality(nodes, len(doc)).score
    doc.close()
    assert score >= TOC_FALLBACK_THRESHOLD, (
        f"fixture ToC scores {score}, below the {TOC_FALLBACK_THRESHOLD} "
        "threshold; it would be marked enrichment-incomplete and never reused"
    )
    return pdf_path


@pytest.fixture
def not_editable(monkeypatch):
    """Force the editability probe False.

    The suite runs from an editable checkout, where reuse is disabled by
    design, so a hit test against the ambient environment would report the
    editable-install rule rather than the behaviour it names.
    """
    monkeypatch.setattr("datasheetindex.tools.bound.is_editable_install", lambda: False)


@pytest.fixture
def build_spy(monkeypatch):
    """Count real builds, so 'rebuilt' and 'reused' are direct assertions."""
    calls: list[str] = []
    original = DatasheetIndex.build

    def counting_build(self, *args, **kwargs):
        calls.append(self.pdf_path)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DatasheetIndex, "build", counting_build)
    return calls


def _force_weak_quality(monkeypatch):
    """Make the ToC fallback eligible on any document.

    Patched at the index's own reference, so both the original and the
    candidate assessment return the same low score -- which is what makes
    _accept_llm_toc_candidate decline, keeping this helper's effect confined to
    'the fallback runs' rather than 'the fallback wins'.
    """
    monkeypatch.setattr(
        "datasheetindex.index.assess_toc_quality",
        lambda _nodes, _pages: TocQuality(
            score=0.1,
            entry_count=1,
            max_depth=1,
            page_coverage=0.2,
            recommend_summaries=True,
            details="forced low",
        ),
    )


def test_in_memory_hit_returns_without_rebuilding(tmp_path, toc_pdf, build_spy):
    """The common path must keep its cache."""
    out = str(tmp_path / "out")
    with DatasheetTools(str(toc_pdf)) as tools:
        tools.build_datasheet(output_dir=out)
        assert len(build_spy) == 1
        tools.build_datasheet(output_dir=out)

    assert len(build_spy) == 1


def test_retry_on_the_same_instance_rebuilds(tmp_path, toc_pdf, build_spy, monkeypatch):
    """A transient failure must not be served from memory forever.

    Without the in-memory condition this test fails while every disk test still
    passes, which is exactly how the gap survived a review.
    """
    out = str(tmp_path / "out")
    attempts: list[int] = []

    def dummy_callable(_system, _user):
        return "unused"

    def flaky_fallback(_text, _pages, _callable):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("gateway timeout")
        return [
            TocNode(title="Thin", level=1, start_page=1, end_page=3, node_id="0001")
        ]

    monkeypatch.setattr(
        DatasheetIndex, "_try_create_default_llm_client", lambda _self: dummy_callable
    )
    monkeypatch.setattr(
        "datasheetindex.llm.toc_fallback.generate_toc_from_text", flaky_fallback
    )
    _force_weak_quality(monkeypatch)

    with DatasheetTools(str(toc_pdf)) as tools:
        first = tools.build_datasheet(output_dir=out)
        assert first.llm_enrichment_incomplete is True
        second = tools.build_datasheet(output_dir=out)

    assert len(build_spy) == 2, "the degraded artifact was served from memory"
    assert second.llm_enrichment_incomplete is False
    assert attempts == [1, 1]
