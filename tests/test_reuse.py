"""Tests for on-disk and in-memory artifact reuse."""

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
        output_dir="out", output_stem=None, include_summaries=False, model=None
    )

    assert set(options.to_dict()) == {f.name for f in fields(_BuildOptions)}


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
