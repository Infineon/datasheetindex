"""Tests for on-disk and in-memory artifact reuse."""

import asyncio
import hashlib
import json
import os
from dataclasses import fields

import pymupdf
import pytest

from datasheetindex.core.artifact_cache import read_sidecar, sidecar_path, write_sidecar
from datasheetindex.index import TOC_FALLBACK_THRESHOLD, DatasheetIndex
from datasheetindex.models import TocNode, TocQuality
from datasheetindex.tools.bound import DatasheetTools, _BuildOptions
from datasheetindex.tools.defs import create_datasheet_tool_defs


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
        # y=400 sits well outside the top/bottom furniture bands (20%/80% of
        # an 842pt page): identical text on every page would otherwise be
        # detected as a running header and stripped from scan_pages' output.
        writer.append((72, 400), "Body text for this page of the datasheet")
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
def other_toc_pdf(tmp_path):
    """A second document, unmistakably distinct from ``toc_pdf``.

    Different page count, different ``set_toc`` entries, and different body
    text, so an A -> B -> A -> B switch test can tell the two apart. Scored
    against ``TOC_FALLBACK_THRESHOLD`` for the same reason ``toc_pdf`` is: a
    bookmark-free PDF scores 0.00 and would be marked
    llm_enrichment_incomplete, which makes it permanently uncacheable and the
    switch-back would rebuild for that reason instead of the one under test.
    """
    from datasheetindex.core.quality import assess_toc_quality
    from datasheetindex.core.structure import build_tree, extract_toc

    doc = pymupdf.open()
    for _ in range(5):
        page = doc.new_page()
        writer = pymupdf.TextWriter(page.rect)
        # See toc_pdf above: keep body text out of the furniture bands so it
        # is not detected as a running header and stripped.
        writer.append((72, 400), "Unique marker Zephyr for the other datasheet")
        writer.write_text(page)
    doc.set_toc(
        [
            [1, "Introduction", 1],
            [1, "Pin Configuration", 2],
            [1, "Absolute Maximum Ratings", 4],
        ]
    )
    pdf_path = tmp_path / "other.pdf"
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
def figure_pdf(tmp_path):
    """``toc_pdf`` plus one raster region, so captioning has a candidate.

    ``toc_pdf`` carries no images at all, so every caption count taken on it is
    trivially zero: a capability test written against it would pass whatever
    the rule did. Same ToC, same body text, one image above ``min_area_pct``.
    """
    from datasheetindex.core.quality import assess_toc_quality
    from datasheetindex.core.structure import build_tree, extract_toc
    from datasheetindex.core.textfile import scan_pages

    doc = pymupdf.open()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20))
    pix.set_rect(pix.irect, (10, 20, 30))
    # A flat colour renders as one distinct colour and the blank-region guard
    # (figure_captions._is_blank_region) skips it before dispatch; an accent
    # stripe keeps this fixture below that threshold, like real content.
    pix.set_rect(pymupdf.IRect(0, 0, 20, 1), (200, 200, 200))
    for number in range(3):
        page = doc.new_page(width=595, height=842)
        writer = pymupdf.TextWriter(page.rect)
        writer.append((72, 72), "Body text for this page of the datasheet")
        writer.write_text(page)
        if number == 1:
            page.insert_image(pymupdf.Rect(50, 200, 545, 600), pixmap=pix)
    doc.set_toc([[1, "Overview", 1], [1, "Electrical Characteristics", 2]])
    pdf_path = tmp_path / "figures.pdf"
    doc.save(str(pdf_path))

    nodes = build_tree(extract_toc(doc), len(doc))
    score = assess_toc_quality(nodes, len(doc)).score
    rasters = [entry for entry in scan_pages(doc).figures if entry["kind"] == "raster"]
    doc.close()
    assert score >= TOC_FALLBACK_THRESHOLD, (
        f"fixture ToC scores {score}, below the {TOC_FALLBACK_THRESHOLD} "
        "threshold; it would be marked enrichment-incomplete and never reused"
    )
    assert len(rasters) == 1, (
        f"fixture yields {len(rasters)} raster candidates, not 1; the caption "
        "tests would assert nothing"
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


def test_a_build_writes_a_sidecar_beside_the_deliverables(tmp_path, toc_pdf):
    out = tmp_path / "out"
    with DatasheetTools(str(toc_pdf)) as tools:
        artifacts = tools.build_datasheet(output_dir=str(out))

    assert artifacts.json_path is not None
    assert artifacts.text_path is not None
    sidecar = sidecar_path(out, artifacts.json_path.stem)
    assert sidecar.exists()
    assert sorted(p.name for p in out.iterdir()) == [
        "ds.build.json",
        "ds.json",
        "ds.txt",
    ]

    record = read_sidecar(sidecar)
    assert record is not None
    assert record.json_name == artifacts.json_path.name
    assert record.text_name == artifacts.text_path.name
    assert (
        record.json_sha256
        == hashlib.sha256(artifacts.json_path.read_bytes()).hexdigest()
    )
    assert (
        record.text_sha256
        == hashlib.sha256(artifacts.text_path.read_bytes()).hexdigest()
    )
    assert record.source_sha256 == hashlib.sha256(toc_pdf.read_bytes()).hexdigest()
    assert record.source_size == toc_pdf.stat().st_size
    assert record.build_options["include_summaries"] is False
    assert record.build_options["model"] is None
    assert record.llm_enrichment_incomplete is False
    # details is absent from the deliverable and must be in the sidecar.
    assert "details" in record.toc_quality
    assert "details" not in artifacts.json_data["toc_quality"]


def test_a_source_swapped_mid_build_writes_no_sidecar(
    tmp_path, toc_pdf, build_spy, monkeypatch, caplog
):
    """Hashing the source after the build would record the wrong generation.

    Simulates a fetcher replacing the PDF in place while a build is running by
    wrapping ``DatasheetIndex.build`` itself: the wrapped call still returns
    real artifacts (PyMuPDF already had the original bytes open), but by the
    time it returns, the source file on disk is a different generation. If
    the source were fingerprinted after the build, the sidecar would record
    the new bytes' hash while describing the old bytes' artifacts, and every
    later request would compare the new bytes against themselves and match
    forever.
    """
    out = tmp_path / "out"
    original = DatasheetIndex.build
    swapped_bytes = toc_pdf.read_bytes() + b"%swapped-revision\n"

    def swapping_build(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        toc_pdf.write_bytes(swapped_bytes)
        return result

    monkeypatch.setattr(DatasheetIndex, "build", swapping_build)

    with caplog.at_level("WARNING", logger="datasheetindex.tools.bound"):
        with DatasheetTools(str(toc_pdf)) as tools:
            artifacts = tools.build_datasheet(output_dir=str(out))

    assert len(build_spy) == 1, "the swap should not itself trigger a rebuild"
    assert artifacts.json_path is not None
    assert artifacts.text_path is not None
    assert artifacts.json_path.exists()
    assert artifacts.text_path.exists()
    assert not sidecar_path(out, artifacts.json_path.stem).exists()
    assert "changed while the build was running" in caplog.text


def test_a_normal_build_records_the_sources_own_hash(tmp_path, toc_pdf):
    """The ordinary path still records the pre-build source fingerprint."""
    out = tmp_path / "out"
    with DatasheetTools(str(toc_pdf)) as tools:
        artifacts = tools.build_datasheet(output_dir=str(out))

    assert artifacts.json_path is not None
    sidecar = sidecar_path(out, artifacts.json_path.stem)
    record = read_sidecar(sidecar)
    assert record is not None
    assert record.source_sha256 == hashlib.sha256(toc_pdf.read_bytes()).hexdigest()


def test_build_options_to_dict_covers_every_dataclass_field():
    """A hand-written map could silently omit a future field from the cache
    key; ``asdict`` cannot -- pin that the two stay in lockstep."""
    options = _BuildOptions(
        output_dir="out",
        output_stem=None,
        include_summaries=False,
        model=None,
        caption_figures=True,
        max_figure_captions=20,
        vision_model=None,
        text_model=None,
        strip_furniture=True,
        regenerate_toc=False,
    )

    assert set(options.to_dict()) == {f.name for f in fields(_BuildOptions)}


def test_regenerate_toc_is_part_of_the_cache_key():
    """It changes the artifact's CONTENT -- the ToC is rewritten -- so an
    artifact built one way must not be served for a request that asked for the
    other. Same reasoning as strip_furniture in 0.33.0."""
    from dataclasses import fields

    from datasheetindex.tools.bound import _BuildOptions

    assert "regenerate_toc" in {f.name for f in fields(_BuildOptions)}


def test_flipping_regenerate_toc_changes_the_recorded_key():
    from typing import Any

    from datasheetindex.tools.bound import _BuildOptions

    common: dict[str, Any] = dict(
        output_dir="out",
        output_stem=None,
        include_summaries=False,
        model=None,
        caption_figures=True,
        max_figure_captions=20,
        vision_model=None,
        text_model=None,
        strip_furniture=True,
    )
    off = _BuildOptions(regenerate_toc=False, **common)
    on = _BuildOptions(regenerate_toc=True, **common)
    assert off.to_dict() != on.to_dict()


def test_a_failing_sidecar_write_does_not_fail_the_build(
    tmp_path, toc_pdf, monkeypatch
):
    """Caching is best effort; the artifacts are correct either way.

    Injected at the sidecar writer, not by making the output directory
    unwritable -- that would fail the deliverable writes too, so the build would
    raise for a different reason and the test would prove nothing.
    """

    def boom(_path, _record):
        raise OSError("no space left on device")

    monkeypatch.setattr("datasheetindex.tools.bound.write_sidecar", boom)
    out = tmp_path / "out"

    with DatasheetTools(str(toc_pdf)) as tools:
        artifacts = tools.build_datasheet(output_dir=str(out))

    assert artifacts.json_path is not None
    assert artifacts.text_path is not None
    assert artifacts.json_path.exists()
    assert artifacts.text_path.exists()
    assert artifacts.json_data["total_pages"] == 3
    assert artifacts.text_content
    assert not sidecar_path(out, artifacts.json_path.stem).exists()


def test_the_next_build_after_a_failed_sidecar_write_rebuilds(
    tmp_path, toc_pdf, build_spy, monkeypatch
):
    """No sidecar means rebuild, not error.

    Restores ``write_sidecar`` with a second explicit ``setattr`` rather than
    ``monkeypatch.undo()``: ``build_spy`` and this test share one ``monkeypatch``
    fixture instance (pytest caches it per test), so ``undo()`` would also
    revert build_spy's own patch on ``DatasheetIndex.build`` and silently stop
    counting the second build.
    """
    out = tmp_path / "out"

    def boom(_path, _record):
        raise OSError("no space left on device")

    monkeypatch.setattr("datasheetindex.tools.bound.write_sidecar", boom)
    with DatasheetTools(str(toc_pdf)) as tools:
        tools.build_datasheet(output_dir=str(out))
    monkeypatch.setattr("datasheetindex.tools.bound.write_sidecar", write_sidecar)

    with DatasheetTools(str(toc_pdf)) as tools:
        rebuilt = tools.build_datasheet(output_dir=str(out))

    assert len(build_spy) == 2
    assert rebuilt.json_data["total_pages"] == 3


def test_the_sidecar_is_removed_before_the_deliverables_are_rewritten(
    tmp_path, toc_pdf, monkeypatch
):
    """Invalidate, write data, publish -- in that order."""
    out = tmp_path / "out"
    with DatasheetTools(str(toc_pdf)) as tools:
        artifacts = tools.build_datasheet(output_dir=str(out))
    assert artifacts.json_path is not None
    sidecar = sidecar_path(out, artifacts.json_path.stem)
    assert sidecar.exists()

    seen_during_build: list[bool] = []
    original = DatasheetIndex.build

    def observing_build(self, *args, **kwargs):
        seen_during_build.append(sidecar.exists())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DatasheetIndex, "build", observing_build)

    with DatasheetTools(str(toc_pdf)) as tools:
        tools.build_datasheet(output_dir=str(out), force_rebuild=True)

    assert seen_during_build == [False], "the stale sidecar outlived the data"
    assert sidecar.exists()


def test_a_second_fresh_instance_reuses_the_artifacts(
    tmp_path, toc_pdf, build_spy, not_editable
):
    """The check that established the problem, inverted."""
    out = tmp_path / "out"
    with DatasheetTools(str(toc_pdf)) as tools:
        first = tools.build_datasheet(output_dir=str(out))
    assert first.json_path is not None
    assert first.text_path is not None
    os.utime(first.json_path, (0, 0))
    os.utime(first.text_path, (0, 0))

    with DatasheetTools(str(toc_pdf)) as tools:
        second = tools.build_datasheet(output_dir=str(out))

    assert len(build_spy) == 1, "the second instance rebuilt"
    assert first.json_path.stat().st_mtime == 0, "the JSON was rewritten"
    assert first.text_path.stat().st_mtime == 0, "the text file was rewritten"
    assert second.json_data == first.json_data
    assert second.text_content == first.text_content
    assert second.toc_quality == first.toc_quality
    assert [n.title for n in second.nodes] == [n.title for n in first.nodes]
    assert second.nodes[0].table_count == first.nodes[0].table_count
    assert second.json_path == first.json_path
    assert second.text_path == first.text_path
    assert second.llm_enrichment_incomplete is False
    assert second.llm_enrichment_notes == ()


def test_reused_quality_carries_details_the_deliverable_drops(
    tmp_path, toc_pdf, not_editable, build_spy
):
    """The reason TocQuality is stored whole rather than recomputed.

    ``build_spy`` pins that the second build actually came from disk: without
    it, deleting ``_reuse_from_disk`` entirely still leaves this test passing,
    since a fresh rebuild of the same PDF also produces matching `details`.
    """
    out = tmp_path / "out"
    with DatasheetTools(str(toc_pdf)) as tools:
        first = tools.build_datasheet(output_dir=str(out))
    with DatasheetTools(str(toc_pdf)) as tools:
        second = tools.build_datasheet(output_dir=str(out))

    assert len(build_spy) == 1, "the second call rebuilt instead of reusing"
    assert first.toc_quality is not None
    assert second.toc_quality is not None
    assert second.toc_quality.details == first.toc_quality.details
    assert second.toc_quality.details != ""


def test_changing_the_vision_model_blocks_reuse(
    tmp_path, toc_pdf, not_editable, build_spy, monkeypatch
):
    """Captioning with a different model must not serve the old model's captions.

    ``model`` has always been in the cache key because it changes what is in
    the artifact; ``DATASHEETINDEX_VISION_MODEL`` changes the same thing and was
    not, so switching it served the previous model's captions from disk in
    silence. That is worst for the person most likely to touch the knob --
    someone switching *because* the captions were not good enough, who would
    see no change and conclude it does nothing.

    ``not_editable`` is required: on-disk reuse is off in an editable checkout,
    so without it this passes for the wrong reason (every call rebuilds).
    """
    out = str(tmp_path / "out")
    monkeypatch.delenv("DATASHEETINDEX_VISION_MODEL", raising=False)
    with DatasheetTools(str(toc_pdf)) as tools:
        tools.build_datasheet(output_dir=out)
    assert len(build_spy) == 1

    # Unchanged environment: the artifact is still good.
    with DatasheetTools(str(toc_pdf)) as tools:
        tools.build_datasheet(output_dir=out)
    assert len(build_spy) == 1, "an unchanged environment must still reuse"

    monkeypatch.setenv("DATASHEETINDEX_VISION_MODEL", "some-other-vision-model")
    with DatasheetTools(str(toc_pdf)) as tools:
        tools.build_datasheet(output_dir=out)
    assert len(build_spy) == 2, "a changed vision model must rebuild"


def test_changing_the_text_model_blocks_reuse(
    tmp_path, toc_pdf, not_editable, build_spy, monkeypatch
):
    """``DATASHEETINDEX_MODEL`` changes the artifact, so it must key the cache.

    Same argument as the vision knob above, one layer up: the text model writes
    the reconstructed ToC and the section summaries, so serving the previous
    model's output from disk after the knob moves is the identical silent
    failure. Keyed as the **env value** rather than the resolved model, so that
    an unset knob stays keyed by ``model`` alone and does not record the same
    fact twice.
    """
    out = str(tmp_path / "out")
    monkeypatch.delenv("DATASHEETINDEX_MODEL", raising=False)
    with DatasheetTools(str(toc_pdf)) as tools:
        tools.build_datasheet(output_dir=out)
    assert len(build_spy) == 1

    with DatasheetTools(str(toc_pdf)) as tools:
        tools.build_datasheet(output_dir=out)
    assert len(build_spy) == 1, "an unchanged environment must still reuse"

    monkeypatch.setenv("DATASHEETINDEX_MODEL", "some-other-text-model")
    with DatasheetTools(str(toc_pdf)) as tools:
        tools.build_datasheet(output_dir=out)
    assert len(build_spy) == 2, "a changed text model must rebuild"


def test_flipping_the_furniture_hatch_blocks_reuse(
    tmp_path, toc_pdf, not_editable, build_spy, monkeypatch
):
    """``DATASHEETINDEX_FURNITURE`` changes the text file, so it must key it.

    Third instance of the class the two model knobs above already fixed, and
    the most direct: this hatch decides whether the published text file keeps
    or drops every running header and footer. Without it in the key, building
    with stripping on and rebuilding with the hatch set matched on version,
    source and options and served the stale stripped file -- and ``text_sha256``
    agreed, because it hashes that same stale file. The reverse is equally
    stale.

    Both directions are exercised: someone reaching for the hatch is turning
    it on, and someone else later turns it back off.

    ``not_editable`` is required: on-disk reuse is off in an editable checkout,
    so without it every call rebuilds and this passes for the wrong reason.
    """
    out = str(tmp_path / "out")
    monkeypatch.delenv("DATASHEETINDEX_FURNITURE", raising=False)
    with DatasheetTools(str(toc_pdf)) as tools:
        tools.build_datasheet(output_dir=out)
    assert len(build_spy) == 1

    with DatasheetTools(str(toc_pdf)) as tools:
        tools.build_datasheet(output_dir=out)
    assert len(build_spy) == 1, "an unchanged environment must still reuse"

    monkeypatch.setenv("DATASHEETINDEX_FURNITURE", "0")
    with DatasheetTools(str(toc_pdf)) as tools:
        tools.build_datasheet(output_dir=out)
    assert len(build_spy) == 2, "turning stripping off must rebuild"

    monkeypatch.delenv("DATASHEETINDEX_FURNITURE")
    with DatasheetTools(str(toc_pdf)) as tools:
        tools.build_datasheet(output_dir=out)
    assert len(build_spy) == 3, "turning stripping back on must rebuild"


def test_the_furniture_hatch_spelling_does_not_split_the_cache(
    tmp_path, toc_pdf, not_editable, build_spy, monkeypatch
):
    """The key records the resolved boolean, not the spelling.

    ``0``, ``false``, ``no`` and ``off`` all mean one thing, so recording the
    raw string would spread one artifact across four keys and rebuild on a
    change that cannot reach the output.
    """
    out = str(tmp_path / "out")
    monkeypatch.setenv("DATASHEETINDEX_FURNITURE", "0")
    with DatasheetTools(str(toc_pdf)) as tools:
        tools.build_datasheet(output_dir=out)
    assert len(build_spy) == 1

    monkeypatch.setenv("DATASHEETINDEX_FURNITURE", "off")
    with DatasheetTools(str(toc_pdf)) as tools:
        tools.build_datasheet(output_dir=out)
    assert len(build_spy) == 1, "a different spelling of off must still reuse"


def test_a_dotenv_sourced_text_model_is_visible_to_the_cache_key(
    tmp_path, toc_pdf, not_editable, build_spy, monkeypatch
):
    """The key must not read the environment before ``.env`` has been folded in.

    ``load_dotenv`` used to run only inside ``create_llm_client``, while
    ``_BuildOptions`` -- the key -- is built before any client exists. So on the
    first build of a process a ``.env``-configured model keyed as ``None`` and
    the build ran on the ``.env`` value: the second call rebuilt with nothing
    changed, and a later session's first call matched the stale ``None`` sidecar
    and served the previous model's ToC and summaries.

    The stand-in for ``.env`` is a fake ``dotenv`` whose ``load_dotenv``
    populates ``os.environ``, which is exactly what the real one does and what
    the autouse hermetic fixture otherwise neutralises. Both builds run with it
    installed, so a rebuild here can only come from the two reads disagreeing.
    """
    import sys
    import types

    def _load_dotenv(*_args, **_kwargs):
        # Through monkeypatch, not os.environ directly: delenv on an
        # already-absent name records nothing to restore, so a raw setdefault
        # here escapes the test and reaches the integration tests, which opt out
        # of the hermetic fixture and would then ask the gateway for a model
        # named "from-dotenv". The no-override check keeps the real
        # load_dotenv's semantics.
        if "DATASHEETINDEX_MODEL" not in os.environ:
            monkeypatch.setenv("DATASHEETINDEX_MODEL", "from-dotenv")

    monkeypatch.delenv("DATASHEETINDEX_MODEL", raising=False)
    monkeypatch.setitem(
        sys.modules, "dotenv", types.SimpleNamespace(load_dotenv=_load_dotenv)
    )

    out = str(tmp_path / "out")
    with DatasheetTools(str(toc_pdf)) as tools:
        tools.build_datasheet(output_dir=out)
    assert len(build_spy) == 1
    assert os.environ.get("DATASHEETINDEX_MODEL") == "from-dotenv", (
        "the fake .env never loaded, so this test proves nothing"
    )

    with DatasheetTools(str(toc_pdf)) as tools:
        tools.build_datasheet(output_dir=out)
    assert len(build_spy) == 1, (
        "the second build keyed a model the first one did not record"
    )


def test_an_explicit_model_makes_the_text_model_env_irrelevant_to_the_key(
    tmp_path, toc_pdf, not_editable, build_spy, monkeypatch
):
    """The negative half of the key: a knob that cannot decide must not rebuild.

    With ``model`` given it outranks ``DATASHEETINDEX_MODEL``, so the env value
    cannot reach the artifact. Keying it anyway would throw away every cached
    artifact each time an unrelated deployment default moved -- a cost with no
    corresponding correctness win.

    The factory is stubbed because an explicit ``model`` makes ``build_datasheet``
    construct a client eagerly, which raises without credentials; the hermetic
    fixture guarantees there are none. What is under test is the cache key, not
    the gateway.
    """

    def dummy_callable(_system, _user):
        return "unused"

    monkeypatch.setattr(
        "datasheetindex.llm.client.create_llm_client",
        lambda *_args, **_kwargs: dummy_callable,
    )

    out = str(tmp_path / "out")
    monkeypatch.delenv("DATASHEETINDEX_MODEL", raising=False)
    with DatasheetTools(str(toc_pdf)) as tools:
        tools.build_datasheet(output_dir=out, model="gpt-5")
    assert len(build_spy) == 1

    monkeypatch.setenv("DATASHEETINDEX_MODEL", "cannot-reach-this-artifact")
    with DatasheetTools(str(toc_pdf)) as tools:
        tools.build_datasheet(output_dir=out, model="gpt-5")
    assert len(build_spy) == 1, "a knob the explicit model overrides must not rebuild"


def test_dotenv_is_folded_in_before_the_output_dir_is_resolved(
    tmp_path, toc_pdf, monkeypatch
):
    """``.env`` must land before *any* DATASHEETINDEX_* read, not just the models.

    ``resolve_default_output_dir`` reads ``DATASHEETINDEX_OUTPUT_DIR`` earlier in
    ``build_datasheet`` than the model readers run, so loading ``.env`` at the
    model readers left this one variable seeing a pre-``.env`` environment on
    the first build of a process and a post-``.env`` one on the second. That is
    the same defect as the model bug, one field over, and it is why the load
    happens once at the top rather than inside each reader -- the ordering
    question then stops existing for variables added later.

    Not a model test: it is here because the guarantee is about ``.env``
    ordering, and it is the cheapest variable to observe it with.
    """
    import sys
    import types

    env_dir = tmp_path / "from-dotenv"

    def _load_dotenv(*_args, **_kwargs):
        if "DATASHEETINDEX_OUTPUT_DIR" not in os.environ:
            monkeypatch.setenv("DATASHEETINDEX_OUTPUT_DIR", str(env_dir))

    monkeypatch.delenv("DATASHEETINDEX_OUTPUT_DIR", raising=False)
    monkeypatch.setitem(
        sys.modules, "dotenv", types.SimpleNamespace(load_dotenv=_load_dotenv)
    )

    with DatasheetTools(str(toc_pdf)) as tools:
        artifacts = tools.build_datasheet()

    assert artifacts.json_path is not None
    assert env_dir in artifacts.json_path.parents, (
        f"the first build ignored the .env output dir: {artifacts.json_path}"
    )


def test_a_whitespace_only_model_cannot_collide_in_the_cache_key(
    tmp_path, toc_pdf, not_editable, build_spy, monkeypatch
):
    """The key and the factory must agree on what counts as naming a model.

    ``create_llm_client`` strips its argument, so ``model=" "`` names nothing
    and the env var decides. The key decided the same question with plain
    truthiness, and ``" "`` is truthy -- so it recorded ``text_model=None``
    while the build resolved through ``DATASHEETINDEX_MODEL``. Two different
    models, one key: the second env value was served the first one's artifact.

    This is the collision case, so it is a correctness test, not a
    performance one -- unlike its sibling above, a failure here means a stale
    artifact reached a caller.
    """

    def dummy_callable(_system, _user):
        return "unused"

    monkeypatch.setattr(
        "datasheetindex.llm.client.create_llm_client",
        lambda *_args, **_kwargs: dummy_callable,
    )
    out = str(tmp_path / "out")

    monkeypatch.setenv("DATASHEETINDEX_MODEL", "model-a")
    with DatasheetTools(str(toc_pdf)) as tools:
        tools.build_datasheet(output_dir=out, model=" ")
    assert len(build_spy) == 1

    monkeypatch.setenv("DATASHEETINDEX_MODEL", "model-b")
    with DatasheetTools(str(toc_pdf)) as tools:
        tools.build_datasheet(output_dir=out, model=" ")
    assert len(build_spy) == 2, "model-b was served model-a's artifact"


def test_a_whitespace_only_model_does_not_satisfy_the_summaries_guard(
    tmp_path, toc_pdf, monkeypatch
):
    """``include_summaries`` requires a model the caller actually named.

    The factory is stubbed so that a *credential* ValueError cannot stand in
    for the guard's: without it this passes whether the guard fires or not,
    which is how the first attempt at this test proved nothing.
    """

    def dummy_callable(_system, _user):
        return "unused"

    monkeypatch.setattr(
        "datasheetindex.llm.client.create_llm_client",
        lambda *_args, **_kwargs: dummy_callable,
    )
    with DatasheetTools(str(toc_pdf)) as tools:
        with pytest.raises(ValueError, match="requires --model"):
            tools.build_datasheet(
                output_dir=str(tmp_path / "out"),
                model="   ",
                include_summaries=True,
            )


def test_reuse_populates_every_field_the_tools_read(
    tmp_path, toc_pdf, not_editable, build_spy
):
    """A partially populated instance would fail later and at a distance.

    ``build_spy`` pins that the second call is a disk reuse: without it, a
    fresh rebuild also populates every field the tools read, and this test
    would pass just as well with ``_reuse_from_disk`` deleted outright.
    """
    out = tmp_path / "out"
    with DatasheetTools(str(toc_pdf)) as tools:
        tools.build_datasheet(output_dir=str(out))

    with DatasheetTools(str(toc_pdf)) as tools:
        tools.build_datasheet(output_dir=str(out))
        manifest = tools.get_artifact_manifest()
        matches = tools.search_text("datasheet")
        section = tools.get_section_text(1, 2)

    assert len(build_spy) == 1, "the second call rebuilt instead of reusing"
    assert manifest["total_pages"] == 3
    assert len(matches) > 0
    assert "datasheet" in section


def test_reuse_preserves_the_toc_source(tmp_path, toc_pdf, not_editable, build_spy):
    """Provenance is a property of the artifact, not of the run that served it.

    A reuse that reported "none" would tell the agent the document has no
    section map while handing it one.
    """
    out = tmp_path / "out"
    with DatasheetTools(str(toc_pdf)) as tools:
        tools.build_datasheet(output_dir=str(out))
        fresh = tools.get_artifact_manifest()["toc_source"]

    with DatasheetTools(str(toc_pdf)) as tools:
        tools.build_datasheet(output_dir=str(out))
        reused = tools.get_artifact_manifest()["toc_source"]

    assert len(build_spy) == 1, "the second call rebuilt instead of reusing"
    assert fresh == "pdf_outline"
    assert reused == fresh


@pytest.mark.parametrize(
    "mutate,token",
    [
        ("source_bytes", "source_content_changed"),
        ("build_options", "build_options_changed"),
        ("version", "version_changed"),
        ("missing_artifact", "artifact_unreadable"),
        ("truncated_artifact", "text_hash_mismatch"),
        ("same_size_text_edit", "text_hash_mismatch"),
        ("mixed_generation", "json_hash_mismatch"),
        ("incomplete_flag", "llm_enrichment_incomplete"),
        ("corrupt_sidecar", "no_sidecar"),
    ],
)
def test_invalidation_rebuilds_for_the_right_reason(
    tmp_path, toc_pdf, build_spy, not_editable, monkeypatch, caplog, mutate, token
):
    """One case per fingerprint field, each asserting which check rejected it."""
    out = tmp_path / "out"
    with DatasheetTools(str(toc_pdf)) as tools:
        first = tools.build_datasheet(output_dir=str(out))
    assert first.json_path is not None
    assert first.text_path is not None
    sidecar = sidecar_path(out, first.json_path.stem)

    if mutate == "source_bytes":
        # A same-size interior flip, not an append: appending changes the file
        # size too, which trips reuse_blocker's cheaper source_size_changed
        # check first and never reaches the content-hash comparison this case
        # names. Flipping a byte in the middle of the embedded font stream
        # (verified against this fixture) still leaves the PDF openable, so
        # the rebuild this test also performs does not itself raise.
        data = bytearray(toc_pdf.read_bytes())
        mid = len(data) // 2
        data[mid] ^= 0xFF
        toc_pdf.write_bytes(bytes(data))
    elif mutate == "build_options":
        # Tamper the stored record directly rather than passing a different
        # output_stem through kwargs: output_stem is also what names the
        # sidecar file, so changing it via kwargs makes the second build look
        # for a sidecar that was never written (no_sidecar) instead of
        # exercising the build_options comparison this case names.
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        record["build_options"]["output_stem"] = "renamed-in-record"
        sidecar.write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    elif mutate == "version":
        monkeypatch.setattr(
            "datasheetindex.tools.bound.package_version", lambda: "0.0.1-other"
        )
    elif mutate == "missing_artifact":
        first.text_path.unlink()
    elif mutate == "truncated_artifact":
        first.text_path.write_text("truncated", encoding="utf-8")
    elif mutate == "same_size_text_edit":
        content = first.text_path.read_text(encoding="utf-8")
        flipped = ("X" if content[0] != "X" else "Y") + content[1:]
        assert len(flipped) == len(content)
        first.text_path.write_text(flipped, encoding="utf-8")
    elif mutate == "mixed_generation":
        payload = json.loads(first.json_path.read_text(encoding="utf-8"))
        payload["total_pages"] = 999
        first.json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    elif mutate == "incomplete_flag":
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        record["llm_enrichment_incomplete"] = True
        record["llm_enrichment_notes"] = ["toc_fallback_raised"]
        sidecar.write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    elif mutate == "corrupt_sidecar":
        sidecar.write_text("{not json", encoding="utf-8")

    with caplog.at_level("DEBUG", logger="datasheetindex.tools.bound"):
        with DatasheetTools(str(toc_pdf)) as tools:
            again = tools.build_datasheet(output_dir=str(out))

    assert len(build_spy) == 2, f"{mutate} did not force a rebuild"
    assert token in caplog.text, f"{mutate} rebuilt, but not for {token}"
    assert again.json_path is not None
    assert again.json_path.exists()
    if mutate == "mixed_generation":
        assert again.json_data["total_pages"] == 3, "served a mixed generation"


def test_an_editable_install_never_reuses(tmp_path, toc_pdf, build_spy, monkeypatch):
    """State the rule; do not merely observe that this checkout is editable."""
    monkeypatch.setattr("datasheetindex.tools.bound.is_editable_install", lambda: True)
    out = tmp_path / "out"
    with DatasheetTools(str(toc_pdf)) as tools:
        tools.build_datasheet(output_dir=str(out))
    with DatasheetTools(str(toc_pdf)) as tools:
        tools.build_datasheet(output_dir=str(out))

    assert len(build_spy) == 2


def test_force_rebuild_bypasses_a_valid_sidecar_and_rewrites_it(
    tmp_path, toc_pdf, build_spy, not_editable
):
    out = tmp_path / "out"
    with DatasheetTools(str(toc_pdf)) as tools:
        tools.build_datasheet(output_dir=str(out))
    with DatasheetTools(str(toc_pdf)) as tools:
        tools.build_datasheet(output_dir=str(out), force_rebuild=True)

    assert len(build_spy) == 2

    # The rewritten sidecar must still be valid, or force_rebuild would
    # permanently disable reuse for that document.
    with DatasheetTools(str(toc_pdf)) as tools:
        tools.build_datasheet(output_dir=str(out))
    assert len(build_spy) == 2


def test_a_rejected_fallback_candidate_is_reused(
    tmp_path, toc_pdf, build_spy, not_editable, monkeypatch
):
    """except versus else, at the cache boundary.

    A candidate declined on the merits leaves the flag false, so its artifact
    must cache -- otherwise every document the fallback cannot help would
    re-pay the LLM cost on every request.
    """

    def dummy_callable(_system, _user):
        return "unused"

    monkeypatch.setattr(
        DatasheetIndex, "_try_create_default_llm_client", lambda _self: dummy_callable
    )
    monkeypatch.setattr(
        "datasheetindex.llm.toc_fallback.generate_toc_from_text",
        lambda _text, _pages, _callable: [
            TocNode(title="Thin", level=1, start_page=1, end_page=3, node_id="0001")
        ],
    )
    _force_weak_quality(monkeypatch)
    out = tmp_path / "out"

    with DatasheetTools(str(toc_pdf)) as tools:
        first = tools.build_datasheet(output_dir=str(out))
    assert first.llm_enrichment_incomplete is False

    with DatasheetTools(str(toc_pdf)) as tools:
        tools.build_datasheet(output_dir=str(out))

    assert len(build_spy) == 1, "a rejected candidate made the document uncacheable"


def test_a_url_source_is_resolved_before_fingerprinting(
    tmp_path, toc_pdf, build_spy, not_editable, monkeypatch
):
    """Content identity is the only thing that can make a URL cacheable.

    Each download lands on a fresh temp filename, so path identity could never
    match. This pins the resolve-then-hash ordering, whose omission would make
    URL sources silently uncacheable.
    """
    from tests.conftest import FakeResponse

    payload = toc_pdf.read_bytes()

    # Patched where the existing URL tests patch it (tests/test_index.py:135),
    # so this follows the precedent that already works rather than reaching for
    # the private ssl-fallback wrapper.
    def fake_urlopen(_url, timeout=None, **_kwargs):
        return FakeResponse(payload)

    monkeypatch.setattr("datasheetindex.index.urllib.request.urlopen", fake_urlopen)
    out = tmp_path / "out"

    with DatasheetTools("https://example.com/test.pdf") as tools:
        first = tools.build_datasheet(output_dir=str(out))
    with DatasheetTools("https://example.com/test.pdf") as tools:
        second = tools.build_datasheet(output_dir=str(out))

    assert len(build_spy) == 1, "the URL source rebuilt despite identical bytes"
    assert second.json_data == first.json_data


def test_a_disappearing_source_during_reuse_check_rebuilds(
    tmp_path, toc_pdf, build_spy, not_editable, monkeypatch, caplog
):
    """reuse_blocker's sha256_file call is not guarded by its own try/except.

    If the source vanishes between reuse_blocker's stat() and its sha256_file()
    call, sha256_file raises FileNotFoundError. That must not escape
    _reuse_from_disk -- every failure here degrades to a rebuild.
    """
    out = tmp_path / "out"
    with DatasheetTools(str(toc_pdf)) as tools:
        tools.build_datasheet(output_dir=str(out))

    def raising_reuse_blocker(*_args, **_kwargs):
        raise FileNotFoundError("source vanished")

    monkeypatch.setattr(
        "datasheetindex.tools.bound.reuse_blocker", raising_reuse_blocker
    )

    with caplog.at_level("DEBUG", logger="datasheetindex.tools.bound"):
        with DatasheetTools(str(toc_pdf)) as tools:
            again = tools.build_datasheet(output_dir=str(out))

    assert len(build_spy) == 2
    assert "source_unreadable" in caplog.text
    assert again.json_path is not None
    assert again.json_path.exists()


def test_a_corrupted_text_deliverable_rebuilds_rather_than_raises(
    tmp_path, toc_pdf, build_spy, not_editable, caplog
):
    """UnicodeDecodeError is a ValueError, not an OSError -- read_text's
    ``except OSError`` alone would let it escape _reuse_from_disk and crash
    the whole build instead of degrading to a rebuild.

    A dangling UTF-8 lead byte (what a cut through a multi-byte character like
    micro or degree leaves behind) is exactly what a truncated write in this
    domain looks like, since datasheet text is full of such characters.
    """
    out = tmp_path / "out"
    with DatasheetTools(str(toc_pdf)) as tools:
        first = tools.build_datasheet(output_dir=str(out))
    assert first.text_path is not None
    first.text_path.write_bytes(first.text_path.read_bytes() + b"\xc2")

    with caplog.at_level("DEBUG", logger="datasheetindex.tools.bound"):
        with DatasheetTools(str(toc_pdf)) as tools:
            again = tools.build_datasheet(output_dir=str(out))

    assert len(build_spy) == 2
    assert "artifact_unreadable" in caplog.text
    assert again.json_path is not None
    assert again.json_path.exists()


def test_a_corrupted_sidecar_encoding_rebuilds_rather_than_raises(
    tmp_path, toc_pdf, build_spy, not_editable, caplog
):
    """The same UnicodeDecodeError gap one level down, in read_sidecar.

    _reuse_from_disk calls read_sidecar with no try/except of its own, so a
    malformed-encoding sidecar must be handled inside read_sidecar itself.
    """
    out = tmp_path / "out"
    with DatasheetTools(str(toc_pdf)) as tools:
        first = tools.build_datasheet(output_dir=str(out))
    assert first.json_path is not None
    sidecar = sidecar_path(out, first.json_path.stem)
    sidecar.write_bytes(b"\xc2")

    with caplog.at_level("DEBUG", logger="datasheetindex.tools.bound"):
        with DatasheetTools(str(toc_pdf)) as tools:
            again = tools.build_datasheet(output_dir=str(out))

    assert len(build_spy) == 2
    assert "no_sidecar" in caplog.text
    assert again.json_path is not None
    assert again.json_path.exists()


def test_an_unresolvable_source_rebuilds(
    tmp_path, toc_pdf, build_spy, not_editable, monkeypatch, caplog
):
    """Only the reuse check's own resolve call fails.

    The rebuild that follows a rejected reuse check also resolves the source
    (to open the document, and again to write the fresh sidecar), so the
    patch must fail once and then get out of the way -- a permanent failure
    would make the fallback rebuild raise too and prove nothing about
    degrading gracefully.
    """
    out = tmp_path / "out"
    with DatasheetTools(str(toc_pdf)) as tools:
        tools.build_datasheet(output_dir=str(out))

    original_resolve = DatasheetIndex._resolve_pdf_source
    calls: list[int] = []

    def flaky_resolve(self):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("cannot resolve source")
        return original_resolve(self)

    monkeypatch.setattr(DatasheetIndex, "_resolve_pdf_source", flaky_resolve)

    with caplog.at_level("DEBUG", logger="datasheetindex.tools.bound"):
        with DatasheetTools(str(toc_pdf)) as tools:
            again = tools.build_datasheet(output_dir=str(out))

    assert len(build_spy) == 2
    assert "source_unresolvable" in caplog.text
    assert again.json_path is not None
    assert again.json_path.exists()


def test_a_malformed_json_deliverable_triggers_deserialization_failed(
    tmp_path, toc_pdf, build_spy, not_editable, caplog
):
    """The hash check must pass before the deserialize branch is reached.

    Rewriting the JSON without also fixing up the sidecar's recorded
    json_sha256 would only re-exercise json_hash_mismatch, which is already
    covered elsewhere -- that would be a test that claims to cover
    deserialization_failed but actually proves nothing about it.
    """
    out = tmp_path / "out"
    with DatasheetTools(str(toc_pdf)) as tools:
        first = tools.build_datasheet(output_dir=str(out))
    assert first.json_path is not None
    sidecar_file = sidecar_path(out, first.json_path.stem)

    payload = json.loads(first.json_path.read_text(encoding="utf-8"))
    del payload["toc"]
    new_json_text = json.dumps(payload, indent=2, ensure_ascii=False)
    first.json_path.write_text(new_json_text, encoding="utf-8")
    new_json_sha256 = hashlib.sha256(new_json_text.encode("utf-8")).hexdigest()

    record = json.loads(sidecar_file.read_text(encoding="utf-8"))
    record["artifacts"]["json"]["sha256"] = new_json_sha256
    sidecar_file.write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with caplog.at_level("DEBUG", logger="datasheetindex.tools.bound"):
        with DatasheetTools(str(toc_pdf)) as tools:
            again = tools.build_datasheet(output_dir=str(out))

    assert len(build_spy) == 2
    assert "deserialization_failed" in caplog.text
    assert again.json_path is not None
    assert again.json_path.exists()


def test_a_carriage_return_in_the_extracted_text_still_reuses(
    tmp_path, toc_pdf, not_editable, build_spy, monkeypatch, caplog
):
    """The reuse-level regression: a CR byte in the extracted text must not
    permanently disable caching for the document that contains one.

    Recorded (json/text) hashes come from ``sha256_file`` on the artifact as
    written; a later reuse check re-reads the file and hashes the resulting
    string with ``sha256_text``. Reading with ``Path.read_text()``'s default
    universal-newline mode silently rewrites a lone ``\\r`` to ``\\n``, so the
    two hashes can never agree again for any document whose extracted text
    carries a CR byte -- the artifact fails validation and rebuilds on every
    later call, forever. This hit 2 of 14 real datasheets in a live corpus run.

    PyMuPDF's own text extraction always normalizes a literal CR to LF before
    handing text back to Python -- confirmed directly: ``page.get_text()``
    never returns a raw ``\\r`` no matter how it is inserted (``insert_text``,
    ``TextWriter``, ``insert_textbox``) -- so no fixture PDF reaches this bug
    through real extraction alone. The CR is injected one layer up, at
    ``_extract_page_blocks``, the seam ``scan_pages`` now composes into the
    text artifact ``atomic_write_text`` puts on disk (it read
    ``_extract_page_text`` before the two-pass furniture rewrite). Everything
    downstream of that seam -- composing the artifact, writing it, recording
    its hash, and later reading it back for the reuse check -- is the real
    production code path; only the origin of the CR byte is synthetic.
    """
    from datasheetindex.core.textfile import _extract_page_blocks as original_extract

    def extract_blocks_with_cr(page):
        return [*original_extract(page), ("\rV\rCC = 3.3 V", False)]

    monkeypatch.setattr(
        "datasheetindex.core.textfile._extract_page_blocks", extract_blocks_with_cr
    )

    out = tmp_path / "out"
    with DatasheetTools(str(toc_pdf)) as tools:
        first = tools.build_datasheet(output_dir=str(out))
    assert first.text_path is not None
    assert "\r" in first.text_path.read_bytes().decode("utf-8"), (
        "the fixture must actually carry a CR on disk, or this test proves "
        "nothing about the bug"
    )

    with caplog.at_level("DEBUG", logger="datasheetindex.tools.bound"):
        with DatasheetTools(str(toc_pdf)) as tools:
            second = tools.build_datasheet(output_dir=str(out))

    assert len(build_spy) == 1, "a CR in the extracted text forced a rebuild"
    assert "hash_mismatch" not in caplog.text
    assert second.text_content == first.text_content


def test_switching_back_to_a_prior_document_reuses_its_artifacts(
    tmp_path, toc_pdf, other_toc_pdf, not_editable, build_spy
):
    """A -> B -> A -> B must cost two builds, not four.

    This is the regression test behind the design document's rejection of
    multi-slot document addressing on the MCP surface: a switch is cheap
    because returning to a document reloads its sidecar instead of rebuilding
    it, which is the whole reason a single bound-document protocol was kept.
    Drives the real handler in ``tools/defs.py`` (not ``DatasheetTools``
    directly), since that is where the switch/rebind logic lives -- the
    switch-correctness tests in ``test_defs.py`` never revisit a document
    built earlier in the same session, so nothing there would catch a
    regression here.
    """
    handlers = {d.name: d for d in create_datasheet_tool_defs()}
    out = tmp_path / "out"

    def build(pdf_source):
        result = asyncio.run(
            handlers["build_datasheet"].handler(
                {"pdf_source": pdf_source, "output_dir": str(out)}
            )
        )
        assert result["is_error"] is False, result["content"][0]["text"]
        return result

    build(str(toc_pdf))  # A: cold build
    build(str(other_toc_pdf))  # B: cold build
    build(str(toc_pdf))  # A: must reuse, not rebuild
    build(str(other_toc_pdf))  # B: must reuse, not rebuild

    assert len(build_spy) == 2, "switching back stopped reusing a prior document"

    section = asyncio.run(
        handlers["get_section_text"].handler({"start_page": 1, "end_page": 1})
    )
    assert section["is_error"] is False
    text = json.loads(section["content"][0]["text"])["text"]
    assert "Zephyr" in text, "the final hop's queries did not return B's text"
    assert "Body text for this page" not in text, "stale text from A leaked in"

    assert sorted(p.name for p in out.iterdir()) == [
        "ds.build.json",
        "ds.json",
        "ds.txt",
        "other.build.json",
        "other.json",
        "other.txt",
    ]


class _VisionStub:
    """A default LLM client that is vision-capable and records its close.

    Registers itself on construction, so a test can count how many clients one
    ``build_datasheet`` call opened as well as whether each was closed. A
    second ``close()`` appends twice and breaks the opened/closed balance,
    which is what makes double-close visible rather than silent.
    """

    def __init__(self, opened: list, closed: list) -> None:
        self._closed = closed
        opened.append(self)

    def __call__(self, _system, _user):
        return "unused"

    def describe_image(self, _system, _image_base64, *, media_type="image/png"):
        return "a block diagram"

    def close(self):
        self._closed.append(self)


def _keyless(monkeypatch, opened=None):
    """Make ``create_llm_client`` fail the way a machine with no [llm] extra does.

    ``conftest``'s hermetic env already strips the credentials, so this is
    belt and braces for the credential path -- but it also gives a test a
    construction counter, which is how "the probe was never built" becomes an
    assertion rather than an assumption.
    """

    def fake_create(*_args, **_kwargs):
        if opened is not None:
            opened.append(1)
        raise ValueError("no credentials")

    monkeypatch.setattr("datasheetindex.llm.client.create_llm_client", fake_create)


def _with_vision(monkeypatch, opened, closed):
    """Make ``create_llm_client`` yield a vision-capable client."""
    monkeypatch.setattr(
        "datasheetindex.llm.client.create_llm_client",
        lambda *_args, **_kwargs: _VisionStub(opened, closed),
    )


def test_keyless_build_with_pending_captions_is_reused(
    tmp_path, figure_pdf, not_editable, build_spy, monkeypatch
):
    """The regression test for the DEFAULT installation.

    A plain ``uv sync`` has no ``[llm]`` extra, so without this every user
    rebuilds every document with a raster region, forever. Pending captions are
    an environment fact, not a defect: they must not set
    ``llm_enrichment_incomplete`` and must not block reuse.
    """
    _keyless(monkeypatch)
    out = str(tmp_path / "out")

    with DatasheetTools(str(figure_pdf)) as tools:
        first = tools.build_datasheet(output_dir=out)
    assert first.figure_captions_pending > 0
    assert first.llm_enrichment_incomplete is False
    assert first.llm_enrichment_notes == ()

    with DatasheetTools(str(figure_pdf)) as tools:
        second = tools.build_datasheet(output_dir=out)

    assert len(build_spy) == 1, "a keyless machine must reuse, not rebuild"
    assert second.figure_captions_pending == first.figure_captions_pending


def test_keyless_single_instance_reuses_from_memory(
    tmp_path, figure_pdf, not_editable, build_spy, monkeypatch
):
    """The memory gate's half of the reuse rule, isolated from the disk gate.

    One instance, two calls, no capability appearing in between: the second
    call must be served straight from ``self._artifacts`` without ever
    reaching ``_reuse_from_disk``. ``test_keyless_build_with_pending_captions_
    is_reused`` opens a second ``DatasheetTools`` for its second call, so it
    only ever exercises the disk gate -- a regression confined to the memory
    gate's own condition is invisible to it. On an editable install, or
    whenever the sidecar could not be written, the memory gate is the *only*
    gate, so it needs its own direct coverage.
    """
    _keyless(monkeypatch)
    out = str(tmp_path / "out")

    with DatasheetTools(str(figure_pdf)) as tools:
        first = tools.build_datasheet(output_dir=out)
        assert first.figure_captions_pending > 0

        second = tools.build_datasheet(output_dir=out)

    assert len(build_spy) == 1, "a keyless single instance must reuse, not rebuild"
    assert second is first, "the memory gate did not serve the cached artifact"


def test_capability_appearing_invalidates_the_artifact(
    tmp_path, figure_pdf, not_editable, build_spy, monkeypatch, caplog
):
    """The other half of the rule: the sidecar gate reacts to the environment."""
    _keyless(monkeypatch)
    out = str(tmp_path / "out")
    with DatasheetTools(str(figure_pdf)) as tools:
        tools.build_datasheet(output_dir=out)

    opened, closed = [], []
    _with_vision(monkeypatch, opened, closed)
    with caplog.at_level("DEBUG", logger="datasheetindex.tools.bound"):
        with DatasheetTools(str(figure_pdf)) as tools:
            second = tools.build_datasheet(output_dir=out)

    assert len(build_spy) == 2, "credentials appeared and nothing was rebuilt"
    assert "figure_captions_pending" in caplog.text, "rebuilt for the wrong reason"
    assert second.figure_captions_pending == 0
    assert any(f.get("caption_source") == "llm" for f in second.json_data["figures"])
    assert len(opened) == 1, "the disk gate and build() each built their own client"
    assert len(closed) == 1, "the probe handed to build() was not closed exactly once"

    with DatasheetTools(str(figure_pdf)) as tools:
        tools.build_datasheet(output_dir=out)
    assert len(build_spy) == 2, "a fully captioned artifact must reuse"


def test_in_memory_cache_obeys_the_capability_rule_on_one_instance(
    tmp_path, figure_pdf, not_editable, monkeypatch
):
    """One instance, so this is the memory gate and not the disk gate.

    Creating a second ``DatasheetTools`` here would silently retest the
    sidecar path, which is a different rule in a different place.
    """
    _keyless(monkeypatch)
    out = str(tmp_path / "out")

    with DatasheetTools(str(figure_pdf)) as tools:
        first = tools.build_datasheet(output_dir=out)
        assert first.figure_captions_pending > 0

        opened, closed = [], []
        _with_vision(monkeypatch, opened, closed)
        second = tools.build_datasheet(output_dir=out)

    assert second is not first, "memory served a caption-less artifact"
    assert second.figure_captions_pending == 0
    assert any(f.get("caption_source") == "llm" for f in second.json_data["figures"])
    assert len(opened) == 1, "memory -> disk -> rebuild built more than one client"
    assert len(closed) == 1


def test_pending_counts_eligible_candidates_only(
    tmp_path, figure_pdf, not_editable, monkeypatch
):
    """Counting all candidates would rebuild forever on any machine with a key,
    precisely because the caller asked for no captions."""
    _keyless(monkeypatch)
    out = str(tmp_path / "out")

    with DatasheetTools(str(figure_pdf)) as tools:
        off = tools.build_datasheet(output_dir=out, caption_figures=False)
    assert off.figure_captions_pending == 0

    with DatasheetTools(str(figure_pdf)) as tools:
        zero = tools.build_datasheet(output_dir=out + "2", max_figure_captions=0)
    assert zero.figure_captions_pending == 0


def test_no_path_returns_holding_an_unclosed_probe(
    tmp_path, figure_pdf, not_editable, build_spy, monkeypatch
):
    """memory -> disk -> rebuild opens exactly one client and closes it once."""
    opened, closed = [], []
    keyless = {"now": True}

    def fake_create(*_args, **_kwargs):
        if keyless["now"]:
            raise ValueError("no credentials")
        return _VisionStub(opened, closed)

    monkeypatch.setattr("datasheetindex.llm.client.create_llm_client", fake_create)
    out = str(tmp_path / "out")

    with DatasheetTools(str(figure_pdf)) as tools:
        first = tools.build_datasheet(output_dir=out)
        assert first.figure_captions_pending > 0
        keyless["now"] = False
        # Memory rejects, disk rejects, and the rebuild runs: three stages that
        # would each construct a client of their own without the resolver.
        tools.build_datasheet(output_dir=out)
        tools.build_datasheet(output_dir=out)

    assert len(build_spy) == 2, "the capability rule did not fire; this proves nothing"
    assert len(opened) == 1, f"{len(opened)} clients opened across one call"
    assert opened == closed, "a probe leaked a connection pool"


def test_probe_is_not_constructed_when_it_cannot_matter(
    tmp_path, figure_pdf, not_editable, monkeypatch
):
    """Laziness: a build that can produce no captions must not touch the gateway."""
    opened: list = []
    _keyless(monkeypatch, opened)
    out = str(tmp_path / "out")
    with DatasheetTools(str(figure_pdf)) as tools:
        tools.build_datasheet(output_dir=out, caption_figures=False)
        tools.build_datasheet(output_dir=out, caption_figures=False)

    assert opened == [], "probed despite nothing being pending"


def test_a_default_client_captions_without_an_explicit_model(
    tmp_path, figure_pdf, monkeypatch
):
    """One client, not two: ``build()`` self-creates for captions as well.

    Before this, ``build()`` self-created only on the weak-ToC branch, so a
    machine with credentials in ``.env`` and no explicit ``model`` never
    captioned -- silently, and against what the tool description and the CLI
    help both promise.
    """
    opened, closed = [], []
    _with_vision(monkeypatch, opened, closed)
    out = str(tmp_path / "out")

    with DatasheetTools(str(figure_pdf)) as tools:
        artifacts = tools.build_datasheet(output_dir=out)

    assert artifacts.figure_captions_pending == 0
    assert any(f.get("caption_source") == "llm" for f in artifacts.json_data["figures"])
    assert len(opened) == 1, "captioning built a second client"
    assert len(closed) == 1, "the self-created client was not closed exactly once"


def test_the_figure_digest_is_identical_fresh_or_reused(
    tmp_path, figure_pdf, not_editable, build_spy, monkeypatch
):
    """The manifest is derived from artifacts either cache gate may have served.

    ``get_artifact_manifest`` reads ``json_data``, which on a reuse is parsed
    from the bytes on disk rather than produced in memory. A digest that
    differed between the two would make the agent's view of a document depend
    on whether it happened to be the first caller, which no consumer could
    detect and none should have to.
    """
    _keyless(monkeypatch)
    out = str(tmp_path / "out")

    with DatasheetTools(str(figure_pdf)) as tools:
        tools.build_datasheet(output_dir=out)
        fresh = tools.get_artifact_manifest()["figures"]

    with DatasheetTools(str(figure_pdf)) as tools:
        tools.build_datasheet(output_dir=out)
        reused = tools.get_artifact_manifest()["figures"]

    assert len(build_spy) == 1, "the second call rebuilt; this proves nothing"
    assert isinstance(fresh, dict)
    assert fresh.get("raster") == 1, "the fixture's raster region is missing"
    assert reused == fresh


def test_a_rejected_cap_does_not_destroy_a_valid_sidecar(
    tmp_path, figure_pdf, not_editable, build_spy, monkeypatch
):
    """Validate before invalidating.

    ``build()`` rejects a bad ``max_figure_captions`` too -- but only after
    ``_build_or_reuse`` has already removed the sidecar to make room for the
    rebuild, so a call that could never succeed used to cost the document its
    cache on the way to raising.
    """
    _keyless(monkeypatch)
    out = str(tmp_path / "out")

    with DatasheetTools(str(figure_pdf)) as tools:
        artifacts = tools.build_datasheet(output_dir=out)
        assert artifacts.json_path is not None
        sidecar = sidecar_path(out, artifacts.json_path.stem)
        assert sidecar.exists()

        with pytest.raises(ValueError, match="max_figure_captions"):
            tools.build_datasheet(output_dir=out, max_figure_captions=-1)

        assert sidecar.exists(), "a rejected call removed a valid sidecar"

    with DatasheetTools(str(figure_pdf)) as tools:
        tools.build_datasheet(output_dir=out)

    assert len(build_spy) == 1, "the surviving sidecar was not usable"


def test_the_probe_does_not_upgrade_an_empty_model_to_the_default(monkeypatch):
    """The gate and the rebuild must resolve the same model, including ``""``.

    Truthiness here made ``model=""`` probe a *default* vision-capable model
    while the rebuild passed ``""`` straight through, whose caption calls all
    fail: the artifact came back incomplete, the gate saw capability, and the
    document rebuilt forever.
    """
    from datasheetindex.tools.bound import _VisionResolver

    seen: list[dict] = []

    def fake_create(**kwargs):
        seen.append(kwargs)
        raise ValueError("no credentials")

    monkeypatch.setattr("datasheetindex.llm.client.create_llm_client", fake_create)

    assert _VisionResolver("").get() is None
    assert seen == [{"model": ""}]


def test_a_missing_llm_client_no_longer_blocks_reuse(tmp_path):
    """No credentials is a fact about the environment, not a failed build.
    Rebuilding cannot create credentials, so refusing reuse costs a full
    rebuild on every request and buys nothing.

    Uses a synthetic source under ``tmp_path`` rather than a real PDF: this is
    a pure ``reuse_blocker`` fingerprint check, which only ever reads bytes
    and stat results, so a hermetic placeholder source is exact -- and unlike
    a real datasheet checked into the working tree but not into git, it exists
    on every clone, including the CI runner that gates a release tag.
    """
    from importlib.metadata import version

    from datasheetindex.core.artifact_cache import (
        ArtifactRecord,
        reuse_blocker,
        sha256_file,
    )

    src = tmp_path / "source.pdf"
    src.write_bytes(b"%PDF-1.4 not a real pdf")
    v = version("datasheetindex")
    record = ArtifactRecord(
        source_sha256=sha256_file(src),
        source_size=src.stat().st_size,
        build_options={},
        datasheetindex_version=v,
        json_name="a.json",
        json_sha256="x",
        text_name="a.txt",
        text_sha256="y",
        toc_quality={},
        toc_fallback_pending=True,
    )
    assert (
        reuse_blocker(record, source_path=src, build_options={}, running_version=v)
        is None
    )


def test_a_transient_llm_failure_still_blocks_reuse(tmp_path):
    """toc_fallback_raised and figure_caption_failed are real failures worth
    retrying; only the no-client case is a stable environment fact.

    See ``test_a_missing_llm_client_no_longer_blocks_reuse`` for why the
    source is a synthetic ``tmp_path`` file rather than a real PDF.
    """
    from importlib.metadata import version

    from datasheetindex.core.artifact_cache import (
        ArtifactRecord,
        reuse_blocker,
        sha256_file,
    )

    src = tmp_path / "source.pdf"
    src.write_bytes(b"%PDF-1.4 not a real pdf")
    v = version("datasheetindex")
    record = ArtifactRecord(
        source_sha256=sha256_file(src),
        source_size=src.stat().st_size,
        build_options={},
        datasheetindex_version=v,
        json_name="a.json",
        json_sha256="x",
        text_name="a.txt",
        text_sha256="y",
        toc_quality={},
        llm_enrichment_incomplete=True,
        llm_enrichment_notes=("toc_fallback_raised",),
    )
    assert (
        reuse_blocker(record, source_path=src, build_options={}, running_version=v)
        == "llm_enrichment_incomplete"
    )


def test_a_sidecar_written_before_the_field_existed_still_loads():
    """from_dict must default this like figure_captions_pending: it is not a
    fingerprint, and requiring it would warn on every pre-existing sidecar."""
    from datasheetindex.core.artifact_cache import ArtifactRecord

    payload = {
        "source_sha256": "a",
        "source_size": 1,
        "build_options": {},
        "datasheetindex_version": "0.0.0",
        "artifacts": {
            "json": {"name": "a.json", "sha256": "x"},
            "text": {"name": "a.txt", "sha256": "y"},
        },
        "toc_quality": {},
        "llm_enrichment_incomplete": False,
        "llm_enrichment_notes": [],
    }
    assert ArtifactRecord.from_dict(payload).toc_fallback_pending is False


def test_toc_fallback_pending_invalidates_on_disk_when_a_client_appears(
    tmp_path, toc_pdf, not_editable, build_spy, monkeypatch, caplog
):
    """The disk gate's half of the rule, mirroring
    ``test_capability_appearing_invalidates_the_artifact`` for
    ``figure_captions_pending``.

    Without ``_reuse_from_disk``'s ``toc_fallback_pending`` check, a document
    built while keyless would be served from disk forever even after
    credentials appeared, freezing its ToC at the original weak quality. This
    is the regression test for deleting that check.
    """
    _force_weak_quality(monkeypatch)
    _keyless(monkeypatch)
    out = str(tmp_path / "out")
    with DatasheetTools(str(toc_pdf)) as tools:
        first = tools.build_datasheet(output_dir=out)
    assert first.toc_fallback_pending is True
    assert first.llm_enrichment_incomplete is False

    opened, closed = [], []
    _with_vision(monkeypatch, opened, closed)
    with caplog.at_level("DEBUG", logger="datasheetindex.tools.bound"):
        with DatasheetTools(str(toc_pdf)) as tools:
            tools.build_datasheet(output_dir=out)

    assert len(build_spy) == 2, "credentials appeared and nothing was rebuilt"
    assert "toc_fallback_pending" in caplog.text, "rebuilt for the wrong reason"


def test_has_client_answers_a_different_question_than_get(monkeypatch):
    """The entire reason ``has_client`` exists instead of reusing ``get``.

    ``get`` returns the vision-filtered client, which is ``None`` for a real
    but non-vision-capable client; the ToC fallback needs any text client, so
    ``has_client`` must see it where ``get`` cannot. Also pins that asking
    both questions in one call constructs at most one client -- ``has_client``
    must reuse ``get``'s own memoization rather than probing a second time.
    """
    from datasheetindex.tools.bound import _VisionResolver

    seen: list[dict] = []

    class _TextOnlyClient:
        def __call__(self, _system, _user):
            return "unused"

    def fake_create(**kwargs):
        seen.append(kwargs)
        return _TextOnlyClient()

    monkeypatch.setattr("datasheetindex.llm.client.create_llm_client", fake_create)

    resolver = _VisionResolver(None)
    assert resolver.get() is None
    assert resolver.has_client() is True
    assert len(seen) == 1, "get() + has_client() constructed more than one client"


def test_toc_fallback_pending_is_false_when_a_client_is_obtained(tmp_path, monkeypatch):
    """The other side of the guard: dropping ``active_llm_callable is None``
    would set this even when a client WAS constructed, poisoning reuse with a
    pending flag the environment already satisfies."""
    pdf_path = tmp_path / "weak.pdf"
    doc = pymupdf.open()
    for _ in range(3):
        page = doc.new_page()
        writer = pymupdf.TextWriter(page.rect)
        writer.append((72, 400), "Body text for this page of the datasheet")
        writer.write_text(page)
    doc.save(str(pdf_path))
    doc.close()

    def dummy_callable(system, user):
        return "unused"

    monkeypatch.setattr(
        DatasheetIndex, "_try_create_default_llm_client", lambda _self: dummy_callable
    )

    idx = DatasheetIndex(str(pdf_path))
    try:
        artifacts = idx.build(output_dir=str(tmp_path / "out"))
    finally:
        idx.close()

    assert artifacts.toc_quality is not None
    assert artifacts.toc_quality.score < TOC_FALLBACK_THRESHOLD
    assert artifacts.toc_fallback_pending is False


def test_toc_fallback_pending_is_false_with_a_caller_supplied_client(
    tmp_path, toc_pdf, monkeypatch
):
    """A caller-supplied callable suppresses construction entirely, so the
    no-client branch never runs and must never mark this pending."""
    _force_weak_quality(monkeypatch)

    def dummy_callable(system, user):
        return "unused"

    idx = DatasheetIndex(str(toc_pdf))
    try:
        artifacts = idx.build(
            output_dir=str(tmp_path / "out"), llm_callable=dummy_callable
        )
    finally:
        idx.close()

    assert artifacts.toc_fallback_pending is False
