"""Tests for the main DatasheetIndex orchestrator."""

import json
import logging
import re
import subprocess
import urllib.error
from pathlib import Path
from typing import Any

import pymupdf
import pytest

from datasheetindex import index as index_module
from datasheetindex.core.preamble import FrontMatter
from datasheetindex.core.textfile import PageScan
from datasheetindex.index import (
    TOC_FALLBACK_THRESHOLD,
    DatasheetIndex,
    _accept_llm_toc_candidate,
)
from datasheetindex.models import TocNode, TocQuality
from tests.conftest import spy_on_toc_fallback

DATA2PAGE_DIR = Path(__file__).resolve().parent.parent.parent / "data2page"
TLE9350_PATH = DATA2PAGE_DIR / "Infineon-TLE9350BSJ-DataSheet-v01_00-EN.pdf"


@pytest.mark.real_pdf
def test_build_produces_artifacts(tmp_path):
    """Full pipeline should produce valid artifacts."""
    if not TLE9350_PATH.exists():
        pytest.skip("Test PDF not found")

    idx = DatasheetIndex(str(TLE9350_PATH))
    artifacts = idx.build(output_dir=str(tmp_path))
    idx.close()

    # Files exist on disk
    assert artifacts.json_path is not None
    assert artifacts.text_path is not None
    assert artifacts.json_path.exists()
    assert artifacts.text_path.exists()


@pytest.mark.real_pdf
def test_json_structure(tmp_path):
    """JSON output should have the expected top-level keys."""
    if not TLE9350_PATH.exists():
        pytest.skip("Test PDF not found")

    idx = DatasheetIndex(str(TLE9350_PATH))
    artifacts = idx.build(output_dir=str(tmp_path))
    idx.close()

    data = artifacts.json_data
    assert "source" in data
    assert "total_pages" in data
    assert "preamble" in data
    assert "toc_quality" in data
    assert "toc" in data
    assert isinstance(data["toc"], list)
    assert data["total_pages"] > 0


@pytest.mark.real_pdf
def test_json_file_valid(tmp_path):
    """The JSON file on disk should be valid JSON."""
    if not TLE9350_PATH.exists():
        pytest.skip("Test PDF not found")

    idx = DatasheetIndex(str(TLE9350_PATH))
    artifacts = idx.build(output_dir=str(tmp_path))
    idx.close()

    assert artifacts.json_path is not None
    content = artifacts.json_path.read_text(encoding="utf-8")
    parsed = json.loads(content)
    assert parsed["source"] == "Infineon-TLE9350BSJ-DataSheet-v01_00-EN.pdf"


@pytest.mark.real_pdf
def test_text_file_page_alignment(tmp_path):
    """Page count in text file should match total_pages in JSON."""
    if not TLE9350_PATH.exists():
        pytest.skip("Test PDF not found")

    idx = DatasheetIndex(str(TLE9350_PATH))
    artifacts = idx.build(output_dir=str(tmp_path))
    idx.close()

    total_pages = artifacts.json_data["total_pages"]
    markers = re.findall(r"--- PAGE (\d+) ---", artifacts.text_content)
    assert len(markers) == total_pages

    # Verify sequential numbering 1..N
    numbers = [int(m) for m in markers]
    assert numbers == list(range(1, total_pages + 1))


@pytest.mark.real_pdf
def test_toc_source_reports_the_pdf_outline(tmp_path):
    """A document whose own bookmarks were used says so. The quality score
    cannot answer this: a good score and a reconstructed tree look identical."""
    if not TLE9350_PATH.exists():
        pytest.skip("Test PDF not found")

    idx = DatasheetIndex(str(TLE9350_PATH))
    artifacts = idx.build(output_dir=str(tmp_path))
    idx.close()

    assert artifacts.json_data["toc_source"] == "pdf_outline"
    assert artifacts.toc_source == "pdf_outline"


@pytest.mark.real_pdf
def test_toc_quality_populated(tmp_path):
    """Quality assessment should be populated."""
    if not TLE9350_PATH.exists():
        pytest.skip("Test PDF not found")

    idx = DatasheetIndex(str(TLE9350_PATH))
    artifacts = idx.build(output_dir=str(tmp_path))
    idx.close()

    assert artifacts.toc_quality is not None
    assert artifacts.toc_quality.score > 0
    assert artifacts.toc_quality.entry_count > 0


@pytest.mark.real_pdf
def test_lazy_doc_and_close():
    """Doc property should lazy-open; close should release."""
    if not TLE9350_PATH.exists():
        pytest.skip("Test PDF not found")

    idx = DatasheetIndex(str(TLE9350_PATH))
    assert idx._doc is None
    _ = idx.doc
    assert idx._doc is not None
    idx.close()
    assert idx._doc is None


def test_url_source_downloads_and_cleans_up(monkeypatch):
    from tests.conftest import DummyDoc, FakeResponse

    opened_paths: list[str] = []

    def fake_urlopen(url: str, timeout: int):
        assert url == "https://example.com/test.pdf"
        assert timeout > 0
        return FakeResponse(b"%PDF-1.7\nmock")

    def fake_open(path: str):
        opened_paths.append(path)
        return DummyDoc()

    monkeypatch.setattr("datasheetindex.index.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("datasheetindex.index.pymupdf.open", fake_open)

    idx = DatasheetIndex("https://example.com/test.pdf")
    _ = idx.doc
    assert len(opened_paths) == 1
    assert idx._temp_pdf_path is not None
    assert idx._temp_pdf_path.exists()
    assert idx._source_file_name() == "test.pdf"
    idx.close()
    assert idx._temp_pdf_path is None


def test_url_source_rejects_non_pdf_content_type(monkeypatch):
    from tests.conftest import FakeResponse

    def fake_urlopen(url: str, timeout: int):
        return FakeResponse(b"<!doctype html>", content_type="text/html")

    monkeypatch.setattr("datasheetindex.index.urllib.request.urlopen", fake_urlopen)

    idx = DatasheetIndex("https://example.com/test.pdf")
    with pytest.raises(ValueError, match="did not return a PDF content type"):
        _ = idx.doc


def test_url_source_rejects_non_pdf_body(monkeypatch):
    from tests.conftest import FakeResponse

    def fake_urlopen(url: str, timeout: int):
        return FakeResponse(b"<!doctype html>", content_type="application/pdf")

    monkeypatch.setattr("datasheetindex.index.urllib.request.urlopen", fake_urlopen)

    idx = DatasheetIndex("https://example.com/test.pdf")
    with pytest.raises(ValueError, match="not a valid PDF"):
        _ = idx.doc


def test_url_source_retries_on_ssl_error(monkeypatch):
    """SSL certificate errors should trigger a retry without verification."""
    import ssl

    from tests.conftest import DummyDoc, FakeResponse

    call_count = 0

    def fake_urlopen(url, timeout=None, context=None):
        nonlocal call_count
        call_count += 1
        if context is None:
            # First attempt — simulate SSL failure
            raise urllib.error.URLError(
                ssl.SSLCertVerificationError(
                    "certificate verify failed: self-signed certificate"
                )
            )
        # Retry with unverified context
        assert context.check_hostname is False
        return FakeResponse(b"%PDF-1.7\nmock")

    opened_paths: list[str] = []

    def fake_open(path: str):
        opened_paths.append(path)
        return DummyDoc()

    monkeypatch.setattr("datasheetindex.index.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("datasheetindex.index.pymupdf.open", fake_open)

    idx = DatasheetIndex("https://vendor.example.com/datasheet.pdf")
    _ = idx.doc
    assert call_count == 2
    assert len(opened_paths) == 1
    idx.close()


class _FakeBuildDoc:
    """Stand-in for a pymupdf.Document in build tests.

    ``metadata`` and ``__getitem__`` are here because the build reads the
    title block on page 1 to decide whether the datasheet covers a product
    family. A fake without them would take the advisory guard in
    ``core/variants.py`` instead of the real path, and pass for the wrong
    reason.
    """

    def __init__(self, pages: int = 3, title: str = ""):
        self._pages = pages
        self.closed = False
        self.metadata = {"title": title}

    def __len__(self):
        return self._pages

    def __getitem__(self, _index):
        return _FakeBuildPage()

    def close(self):
        self.closed = True


class _FakeBuildPage:
    def get_text(self, _kind):
        return {"blocks": []}


def test_build_auto_llm_fallback_when_quality_low(monkeypatch, tmp_path):
    quality_calls = [0]
    fallback_calls: list[object] = []
    llm_models: list[str | None] = []
    fake_llm: object | None = None

    def fake_open(_path: str):
        return _FakeBuildDoc()

    def fake_quality(_nodes, _total_pages):
        quality_calls[0] += 1
        if quality_calls[0] == 1:
            return TocQuality(score=0.0, entry_count=0, max_depth=0, page_coverage=0.0)
        return TocQuality(score=0.8, entry_count=1, max_depth=1, page_coverage=1.0)

    class _FakeAutoLlm:
        def __init__(self) -> None:
            self.closed = False

        def __call__(self, _system: str, _user: str) -> str:
            return "ok"

        def close(self) -> None:
            self.closed = True

    fake_llm = _FakeAutoLlm()

    def fake_client(model: str | None = None):
        llm_models.append(model)
        return fake_llm

    def fake_toc_from_text(_text: str, _total_pages: int, llm_callable, **_kw):
        fallback_calls.append(llm_callable)
        return [
            TocNode(
                title="Auto",
                level=1,
                start_page=1,
                end_page=1,
                node_id="0001",
            )
        ]

    monkeypatch.setattr("datasheetindex.index.pymupdf.open", fake_open)
    # index.py calls scan_pages, not generate_text; the fake docs (only
    # __len__) never reach the real per-page traversal because this patch
    # replaces it outright.
    monkeypatch.setattr(
        "datasheetindex.index.scan_pages",
        lambda _doc, **_kw: PageScan(
            text="--- PAGE 1 ---\n",
            figures=[],
            excluded_below_min_area=0,
        ),
    )
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
    monkeypatch.setattr("datasheetindex.index.extract_toc", lambda _doc: [])
    monkeypatch.setattr(
        "datasheetindex.index.build_tree", lambda _raw, _pages, **_kw: []
    )
    monkeypatch.setattr(
        "datasheetindex.index.enrich_with_table_counts",
        lambda _nodes, _doc, **_kw: _nodes,
    )
    monkeypatch.setattr(
        "datasheetindex.index.enrich_with_continued_tables",
        lambda _nodes, _text: _nodes,
    )
    monkeypatch.setattr(
        "datasheetindex.index.enrich_with_footnote_markers",
        lambda _nodes, _text: _nodes,
    )
    monkeypatch.setattr(
        "datasheetindex.index.enrich_with_cross_references",
        lambda _nodes, _text: _nodes,
    )
    monkeypatch.setattr("datasheetindex.index.assess_toc_quality", fake_quality)
    monkeypatch.setattr("datasheetindex.llm.client.create_llm_client", fake_client)
    monkeypatch.setattr(
        "datasheetindex.llm.toc_fallback.generate_toc_from_text",
        fake_toc_from_text,
    )

    idx = DatasheetIndex("dummy.pdf")
    artifacts = idx.build(output_dir=str(tmp_path))
    idx.close()

    assert artifacts.toc_quality is not None
    assert artifacts.toc_quality.score == 0.8
    # None, not a model name: the automatic fallback names no model, so the
    # factory resolves DATASHEETINDEX_MODEL and its own default. Asserting
    # "gpt-4.1" here would re-pin a default the deployment is meant to move.
    assert llm_models == [None]
    assert len(fallback_calls) == 1
    assert artifacts.json_data["toc"][0]["title"] == "Auto"
    # The tree the caller receives was written by the model, not read from the
    # PDF. Nothing else in the artifacts says so.
    assert artifacts.json_data["toc_source"] == "llm_reconstructed"
    assert artifacts.toc_source == "llm_reconstructed"
    assert fake_llm.closed is True
    # A stub that stops taking effect must fail here rather than pass quietly.
    assert artifacts.json_data["preamble"] == "pre"
    assert artifacts.json_data["preamble_pages"] == []


def test_build_auto_llm_fallback_keeps_original_when_candidate_too_thin(
    monkeypatch, tmp_path
):
    quality_calls = [0]
    fallback_calls: list[object] = []
    llm_models: list[str | None] = []

    def fake_open(_path: str):
        return _FakeBuildDoc(pages=20)

    def fake_quality(_nodes, _total_pages):
        quality_calls[0] += 1
        if quality_calls[0] == 1:
            return TocQuality(
                score=0.2,
                entry_count=6,
                max_depth=1,
                page_coverage=1.0,
            )
        return TocQuality(
            score=0.8,
            entry_count=1,
            max_depth=1,
            page_coverage=1.0,
        )

    class _FakeAutoLlm:
        def __init__(self) -> None:
            self.closed = False

        def __call__(self, _system: str, _user: str) -> str:
            return "ok"

        def close(self) -> None:
            self.closed = True

    fake_llm = _FakeAutoLlm()

    def fake_client(model: str | None = None):
        llm_models.append(model)
        return fake_llm

    def fake_toc_from_text(_text: str, _total_pages: int, llm_callable, **_kw):
        fallback_calls.append(llm_callable)
        return [
            TocNode(
                title="Auto",
                level=1,
                start_page=1,
                end_page=20,
                node_id="0001",
            )
        ]

    monkeypatch.setattr("datasheetindex.index.pymupdf.open", fake_open)
    # index.py calls scan_pages, not generate_text; the fake docs (only
    # __len__) never reach the real per-page traversal because this patch
    # replaces it outright.
    monkeypatch.setattr(
        "datasheetindex.index.scan_pages",
        lambda _doc, **_kw: PageScan(
            text="--- PAGE 1 ---\n",
            figures=[],
            excluded_below_min_area=0,
        ),
    )
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
    monkeypatch.setattr(
        "datasheetindex.index.extract_toc", lambda _doc: [[1, "Original", 1]]
    )
    monkeypatch.setattr(
        "datasheetindex.index.build_tree",
        lambda _raw, _pages, **_kw: [
            TocNode(
                title="Original",
                level=1,
                start_page=1,
                end_page=20,
                node_id="0001",
            )
        ],
    )
    monkeypatch.setattr(
        "datasheetindex.index.enrich_with_table_counts",
        lambda _nodes, _doc, **_kw: _nodes,
    )
    monkeypatch.setattr(
        "datasheetindex.index.enrich_with_continued_tables",
        lambda _nodes, _text: _nodes,
    )
    monkeypatch.setattr(
        "datasheetindex.index.enrich_with_footnote_markers",
        lambda _nodes, _text: _nodes,
    )
    monkeypatch.setattr(
        "datasheetindex.index.enrich_with_cross_references",
        lambda _nodes, _text: _nodes,
    )
    monkeypatch.setattr("datasheetindex.index.assess_toc_quality", fake_quality)
    monkeypatch.setattr("datasheetindex.llm.client.create_llm_client", fake_client)
    monkeypatch.setattr(
        "datasheetindex.llm.toc_fallback.generate_toc_from_text",
        fake_toc_from_text,
    )

    idx = DatasheetIndex("dummy.pdf")
    artifacts = idx.build(output_dir=str(tmp_path))
    idx.close()

    assert artifacts.toc_quality is not None
    assert artifacts.toc_quality.score == 0.2
    # None, not a model name: the automatic fallback names no model, so the
    # factory resolves DATASHEETINDEX_MODEL and its own default. Asserting
    # "gpt-4.1" here would re-pin a default the deployment is meant to move.
    assert llm_models == [None]
    assert len(fallback_calls) == 1
    assert artifacts.json_data["toc"][0]["title"] == "Original"
    # A rejected candidate leaves the PDF's own outline in place, so that is
    # what the field must report -- the fallback having run is not the question.
    assert artifacts.json_data["toc_source"] == "pdf_outline"
    assert fake_llm.closed is True


def _quality(score: float, entry_count: int, page_coverage: float) -> TocQuality:
    return TocQuality(
        score=score,
        entry_count=entry_count,
        max_depth=1,
        page_coverage=page_coverage,
    )


def test_accept_llm_toc_candidate_accepts_thin_candidate_without_baseline():
    """With no bookmarks to protect, a thin-but-real ToC beats no ToC at all."""
    accepted, reason = _accept_llm_toc_candidate(
        _quality(score=0.0, entry_count=0, page_coverage=0.0),
        _quality(score=0.4, entry_count=2, page_coverage=1.0),
        total_pages=12,
    )

    assert accepted is True, reason


def test_accept_llm_toc_candidate_rejects_thin_candidate_against_real_baseline():
    """The entry-count floor still guards an existing ToC."""
    accepted, reason = _accept_llm_toc_candidate(
        _quality(score=0.2, entry_count=6, page_coverage=1.0),
        _quality(score=0.8, entry_count=1, page_coverage=1.0),
        total_pages=20,
    )

    assert accepted is False
    assert "too few entries" in reason


def test_accept_llm_toc_candidate_accepts_fewer_entries_when_score_improves():
    """A cleaner, higher-scoring ToC is not vetoed for having fewer entries."""
    accepted, reason = _accept_llm_toc_candidate(
        _quality(score=0.25, entry_count=60, page_coverage=0.5),
        _quality(score=0.75, entry_count=20, page_coverage=0.9),
        total_pages=100,
    )

    assert accepted is True, reason


def test_accept_llm_toc_candidate_rejects_coverage_regression():
    accepted, reason = _accept_llm_toc_candidate(
        _quality(score=0.2, entry_count=10, page_coverage=0.9),
        _quality(score=0.6, entry_count=10, page_coverage=0.4),
        total_pages=100,
    )

    assert accepted is False
    assert "page coverage" in reason


def test_explicit_request_accepts_a_candidate_that_does_not_beat_the_baseline():
    """An enumerated outline scores well, so the score comparison would refuse
    the very replacement escalation exists to obtain."""
    ok, reason = _accept_llm_toc_candidate(
        _quality(score=0.68, entry_count=10, page_coverage=1.0),
        _quality(score=0.50, entry_count=10, page_coverage=1.0),
        total_pages=20,
        explicit_request=True,
    )
    assert ok is True, reason


def test_without_an_explicit_request_the_score_comparison_still_applies():
    ok, reason = _accept_llm_toc_candidate(
        _quality(score=0.68, entry_count=10, page_coverage=1.0),
        _quality(score=0.50, entry_count=10, page_coverage=1.0),
        total_pages=20,
    )
    assert ok is False
    assert "did not improve" in reason


def test_explicit_request_still_rejects_an_empty_candidate():
    ok, reason = _accept_llm_toc_candidate(
        _quality(score=0.68, entry_count=10, page_coverage=1.0),
        _quality(score=0.90, entry_count=0, page_coverage=1.0),
        total_pages=20,
        explicit_request=True,
    )
    assert ok is False
    assert "no entries" in reason


def test_explicit_request_still_rejects_a_coverage_regression():
    ok, reason = _accept_llm_toc_candidate(
        _quality(score=0.68, entry_count=10, page_coverage=0.9),
        _quality(score=0.90, entry_count=10, page_coverage=0.5),
        total_pages=20,
        explicit_request=True,
    )
    assert ok is False
    assert "coverage" in reason


@pytest.mark.real_pdf
def test_regenerate_toc_without_a_client_raises(
    tmp_path, pdf_tle9350_path, monkeypatch
):
    """An explicit request that cannot be honoured must not look like success.

    **Skipped on a clean clone.** The TLE9350 PDF is gitignored and absent from
    the CI checkout that gates releases, so the protection this describes does
    not exist there -- ``pdf_tle9350_path`` skips. The hermetic test below is
    the one that actually runs in CI; this one only adds a real document.
    """
    monkeypatch.setattr(
        DatasheetIndex, "_try_create_default_llm_client", lambda self: None
    )
    with DatasheetIndex(str(pdf_tle9350_path)) as index:
        with pytest.raises(RuntimeError, match="regenerate_toc"):
            index.build(output_dir=str(tmp_path), regenerate_toc=True)


def test_regenerate_toc_without_a_client_raises_on_a_good_toc(tmp_path, toc_pdf):
    """The hermetic half, and the one that isolates the guard.

    ``toc_pdf`` scores above the threshold and carries no raster figure, so
    nothing else in ``build`` wants an LLM client: the raise can only come from
    the explicit request. The real-PDF test above reaches the same guard, but
    that document has caption candidates, so it would pass even if the request
    itself were ignored -- and it skips in CI besides.
    """
    with DatasheetIndex(str(toc_pdf)) as index:
        with pytest.raises(RuntimeError, match="regenerate_toc"):
            index.build(output_dir=str(tmp_path), regenerate_toc=True)


def _stub_llm_callable():
    """A client-shaped object that is never actually called.

    The ToC fallback is stubbed out in the tests below, so this only has to be
    non-``None``: what is under test is whether ``build`` decides to run the
    fallback, not what a model would answer.
    """

    def call(system: str, user: str) -> str:  # pragma: no cover - never invoked
        raise AssertionError("the stubbed fallback should have intercepted this")

    return call


def test_regenerate_toc_runs_the_fallback_on_an_above_threshold_toc(
    tmp_path, toc_pdf, monkeypatch
):
    """The behavioural pin: an explicit request must actually reach the fallback.

    ``toc_pdf`` scores 0.62 against a 0.3 threshold and has no figures, so the
    LLM fallback is ineligible on every other route. If ``regenerate_toc`` stops
    feeding ``needs_toc_fallback``, the spy is never called and the ToC stays
    the PDF's own -- which is what the whole feature is supposed to prevent, and
    what no structural test could see.
    """
    candidate = [
        TocNode(title="Introduction", level=1, start_page=1, end_page=2),
        TocNode(title="Absolute Maximum Ratings", level=1, start_page=3, end_page=3),
    ]
    calls = spy_on_toc_fallback(monkeypatch, candidate)

    with DatasheetIndex(str(toc_pdf)) as index:
        artifacts = index.build(
            output_dir=str(tmp_path),
            llm_callable=_stub_llm_callable(),
            regenerate_toc=True,
        )

    assert calls == [3], "the LLM ToC fallback was never run"
    # Observed twice over, so a spy that fires without the result being used
    # cannot pass this: the regenerated tree is what the build published.
    assert artifacts.toc_source == "llm_reconstructed"
    assert [node.title for node in artifacts.nodes] == [
        "Introduction",
        "Absolute Maximum Ratings",
    ]


def test_a_good_toc_alone_does_not_run_the_fallback(tmp_path, toc_pdf, monkeypatch):
    """The control for the test above: without the request, nothing escalates.

    Same document, same available client, ``regenerate_toc=False``. This is what
    makes the assertion above about the parameter rather than about the client.
    """
    calls = spy_on_toc_fallback(monkeypatch, [])

    with DatasheetIndex(str(toc_pdf)) as index:
        artifacts = index.build(
            output_dir=str(tmp_path),
            llm_callable=_stub_llm_callable(),
            regenerate_toc=False,
        )

    assert calls == []
    assert artifacts.toc_source == "pdf_outline"


def test_an_explicit_request_does_not_log_the_threshold_sentence(
    tmp_path, toc_pdf, monkeypatch, caplog
):
    """The operator-facing log must not claim a good score was below threshold."""
    spy_on_toc_fallback(monkeypatch, [])

    with caplog.at_level(logging.INFO, logger="datasheetindex.index"):
        with DatasheetIndex(str(toc_pdf)) as index:
            index.build(
                output_dir=str(tmp_path),
                llm_callable=_stub_llm_callable(),
                regenerate_toc=True,
            )

    assert "below threshold (0.62 < 0.30)" not in caplog.text
    assert "requested explicitly" in caplog.text


def test_build_auto_llm_fallback_graceful_without_credentials(monkeypatch, tmp_path):
    def fake_open(_path: str):
        return _FakeBuildDoc()

    monkeypatch.setattr("datasheetindex.index.pymupdf.open", fake_open)
    # index.py calls scan_pages, not generate_text; the fake docs (only
    # __len__) never reach the real per-page traversal because this patch
    # replaces it outright.
    monkeypatch.setattr(
        "datasheetindex.index.scan_pages",
        lambda _doc, **_kw: PageScan(
            text="--- PAGE 1 ---\n",
            figures=[],
            excluded_below_min_area=0,
        ),
    )
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
    monkeypatch.setattr("datasheetindex.index.extract_toc", lambda _doc: [])
    monkeypatch.setattr(
        "datasheetindex.index.build_tree", lambda _raw, _pages, **_kw: []
    )
    monkeypatch.setattr(
        "datasheetindex.index.enrich_with_table_counts",
        lambda _nodes, _doc, **_kw: _nodes,
    )
    monkeypatch.setattr(
        "datasheetindex.index.enrich_with_continued_tables",
        lambda _nodes, _text: _nodes,
    )
    monkeypatch.setattr(
        "datasheetindex.index.enrich_with_footnote_markers",
        lambda _nodes, _text: _nodes,
    )
    monkeypatch.setattr(
        "datasheetindex.index.enrich_with_cross_references",
        lambda _nodes, _text: _nodes,
    )
    monkeypatch.setattr(
        "datasheetindex.index.assess_toc_quality",
        lambda _nodes, _total_pages: TocQuality(
            score=0.0,
            entry_count=0,
            max_depth=0,
            page_coverage=0.0,
        ),
    )

    def _raise_missing_env(model=None):
        raise ValueError("missing env")

    monkeypatch.setattr(
        "datasheetindex.llm.client.create_llm_client",
        _raise_missing_env,
    )

    idx = DatasheetIndex("dummy.pdf")
    artifacts = idx.build(output_dir=str(tmp_path))
    idx.close()

    assert artifacts.toc_quality is not None
    assert artifacts.toc_quality.score == 0.0
    assert artifacts.json_data["toc"] == []
    # No bookmarks and no reconstruction. "pdf_outline" over an empty tree
    # would claim the document was read and found to have no sections.
    assert artifacts.json_data["toc_source"] == "none"
    assert artifacts.toc_source == "none"


def test_build_llm_fallback_graceful_on_api_error(monkeypatch, tmp_path):
    """LLM API errors during ToC fallback should degrade gracefully."""

    def fake_open(_path: str):
        return _FakeBuildDoc()

    class _FakeAutoLlm:
        def __init__(self) -> None:
            self.closed = False

        def __call__(self, _system: str, _user: str) -> str:
            return "ok"

        def close(self) -> None:
            self.closed = True

    fake_llm = _FakeAutoLlm()

    def fake_client(model: str | None = None):
        return fake_llm

    def fake_toc_from_text(_text, _total_pages, _llm_callable, **_kw):
        raise RuntimeError("429 Too Many Requests")

    monkeypatch.setattr("datasheetindex.index.pymupdf.open", fake_open)
    # index.py calls scan_pages, not generate_text; the fake docs (only
    # __len__) never reach the real per-page traversal because this patch
    # replaces it outright.
    monkeypatch.setattr(
        "datasheetindex.index.scan_pages",
        lambda _doc, **_kw: PageScan(
            text="--- PAGE 1 ---\n",
            figures=[],
            excluded_below_min_area=0,
        ),
    )
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
    monkeypatch.setattr("datasheetindex.index.extract_toc", lambda _doc: [])
    monkeypatch.setattr(
        "datasheetindex.index.build_tree", lambda _raw, _pages, **_kw: []
    )
    monkeypatch.setattr(
        "datasheetindex.index.enrich_with_table_counts",
        lambda _nodes, _doc, **_kw: _nodes,
    )
    monkeypatch.setattr(
        "datasheetindex.index.enrich_with_continued_tables",
        lambda _nodes, _text: _nodes,
    )
    monkeypatch.setattr(
        "datasheetindex.index.enrich_with_footnote_markers",
        lambda _nodes, _text: _nodes,
    )
    monkeypatch.setattr(
        "datasheetindex.index.enrich_with_cross_references",
        lambda _nodes, _text: _nodes,
    )
    monkeypatch.setattr(
        "datasheetindex.index.assess_toc_quality",
        lambda _nodes, _total_pages: TocQuality(
            score=0.0, entry_count=0, max_depth=0, page_coverage=0.0
        ),
    )
    monkeypatch.setattr("datasheetindex.llm.client.create_llm_client", fake_client)
    monkeypatch.setattr(
        "datasheetindex.llm.toc_fallback.generate_toc_from_text",
        fake_toc_from_text,
    )

    idx = DatasheetIndex("dummy.pdf")
    artifacts = idx.build(output_dir=str(tmp_path))
    idx.close()

    # Should still produce artifacts with the original (empty) ToC
    assert artifacts.toc_quality is not None
    assert artifacts.toc_quality.score == 0.0
    assert artifacts.json_data["toc"] == []
    assert fake_llm.closed is True


def test_build_output_stem_override(monkeypatch, tmp_path):
    def fake_open(_path: str):
        return _FakeBuildDoc()

    monkeypatch.setattr("datasheetindex.index.pymupdf.open", fake_open)
    # index.py calls scan_pages, not generate_text; the fake docs (only
    # __len__) never reach the real per-page traversal because this patch
    # replaces it outright.
    monkeypatch.setattr(
        "datasheetindex.index.scan_pages",
        lambda _doc, **_kw: PageScan(
            text="--- PAGE 1 ---\n",
            figures=[],
            excluded_below_min_area=0,
        ),
    )
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
    monkeypatch.setattr("datasheetindex.index.extract_toc", lambda _doc: [])
    monkeypatch.setattr(
        "datasheetindex.index.build_tree", lambda _raw, _pages, **_kw: []
    )
    monkeypatch.setattr(
        "datasheetindex.index.enrich_with_table_counts",
        lambda _nodes, _doc, **_kw: _nodes,
    )
    monkeypatch.setattr(
        "datasheetindex.index.enrich_with_continued_tables",
        lambda _nodes, _text: _nodes,
    )
    monkeypatch.setattr(
        "datasheetindex.index.enrich_with_footnote_markers",
        lambda _nodes, _text: _nodes,
    )
    monkeypatch.setattr(
        "datasheetindex.index.enrich_with_cross_references",
        lambda _nodes, _text: _nodes,
    )
    monkeypatch.setattr(
        "datasheetindex.index.assess_toc_quality",
        lambda _nodes, _total_pages: TocQuality(
            score=1.0,
            entry_count=0,
            max_depth=0,
            page_coverage=0.0,
        ),
    )

    idx = DatasheetIndex("dummy.pdf")
    artifacts = idx.build(output_dir=str(tmp_path), output_stem="custom:name")
    idx.close()

    assert artifacts.json_path is not None
    assert artifacts.text_path is not None
    assert artifacts.json_path.name == "custom_name.json"
    assert artifacts.text_path.name == "custom_name.txt"


def test_resolve_default_output_dir_uses_uid_namespaced_tempdir(monkeypatch):
    """Without env override, default lands in <tempdir>/datasheetindex-<uid>."""
    import os
    import tempfile

    from datasheetindex.index import resolve_default_output_dir

    monkeypatch.delenv("DATASHEETINDEX_OUTPUT_DIR", raising=False)
    resolved = Path(resolve_default_output_dir())
    assert resolved.parent == Path(tempfile.gettempdir())
    expected_leaf = (
        f"datasheetindex-{os.getuid()}" if hasattr(os, "getuid") else "datasheetindex"
    )
    assert resolved.name == expected_leaf


def test_resolve_default_output_dir_honours_env_var(monkeypatch, tmp_path):
    from datasheetindex.index import resolve_default_output_dir

    monkeypatch.setenv("DATASHEETINDEX_OUTPUT_DIR", str(tmp_path / "deploy-pinned"))
    assert resolve_default_output_dir() == str(tmp_path / "deploy-pinned")


def test_resolve_default_output_dir_blank_env_var_falls_through(monkeypatch):
    """Empty / whitespace env var must not be treated as a valid path."""
    import tempfile

    from datasheetindex.index import resolve_default_output_dir

    for blank in ("", "   ", "\t\n"):
        monkeypatch.setenv("DATASHEETINDEX_OUTPUT_DIR", blank)
        resolved = Path(resolve_default_output_dir())
        assert resolved.parent == Path(tempfile.gettempdir())


def test_build_with_none_output_dir_writes_to_resolver_default(monkeypatch, tmp_path):
    """idx.build(output_dir=None) writes to the env-resolved default."""
    pinned = tmp_path / "env-pinned"
    monkeypatch.setenv("DATASHEETINDEX_OUTPUT_DIR", str(pinned))

    monkeypatch.setattr(
        "datasheetindex.index.pymupdf.open", lambda _path: _FakeBuildDoc()
    )
    # index.py calls scan_pages, not generate_text; the fake docs (only
    # __len__) never reach the real per-page traversal because this patch
    # replaces it outright.
    monkeypatch.setattr(
        "datasheetindex.index.scan_pages",
        lambda _doc, **_kw: PageScan(
            text="--- PAGE 1 ---\n",
            figures=[],
            excluded_below_min_area=0,
        ),
    )
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
    monkeypatch.setattr("datasheetindex.index.extract_toc", lambda _doc: [])
    monkeypatch.setattr(
        "datasheetindex.index.build_tree", lambda _raw, _pages, **_kw: []
    )
    monkeypatch.setattr(
        "datasheetindex.index.enrich_with_table_counts",
        lambda _nodes, _doc, **_kw: _nodes,
    )
    monkeypatch.setattr(
        "datasheetindex.index.enrich_with_continued_tables",
        lambda _nodes, _text: _nodes,
    )
    monkeypatch.setattr(
        "datasheetindex.index.enrich_with_footnote_markers",
        lambda _nodes, _text: _nodes,
    )
    monkeypatch.setattr(
        "datasheetindex.index.enrich_with_cross_references",
        lambda _nodes, _text: _nodes,
    )
    monkeypatch.setattr(
        "datasheetindex.index.assess_toc_quality",
        lambda _nodes, _total_pages: TocQuality(
            score=1.0, entry_count=0, max_depth=0, page_coverage=0.0
        ),
    )

    idx = DatasheetIndex("dummy.pdf")
    artifacts = idx.build()
    idx.close()

    assert artifacts.json_path is not None
    assert pinned.exists()
    assert artifacts.json_path.parent == pinned


def test_build_with_blank_output_dir_falls_through_to_resolver(monkeypatch, tmp_path):
    """Empty / whitespace output_dir must not be treated as explicit-CWD."""
    pinned = tmp_path / "env-pinned"
    monkeypatch.setenv("DATASHEETINDEX_OUTPUT_DIR", str(pinned))

    monkeypatch.setattr(
        "datasheetindex.index.pymupdf.open", lambda _path: _FakeBuildDoc()
    )
    # index.py calls scan_pages, not generate_text; the fake docs (only
    # __len__) never reach the real per-page traversal because this patch
    # replaces it outright.
    monkeypatch.setattr(
        "datasheetindex.index.scan_pages",
        lambda _doc, **_kw: PageScan(
            text="--- PAGE 1 ---\n",
            figures=[],
            excluded_below_min_area=0,
        ),
    )
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
    monkeypatch.setattr("datasheetindex.index.extract_toc", lambda _doc: [])
    monkeypatch.setattr(
        "datasheetindex.index.build_tree", lambda _raw, _pages, **_kw: []
    )
    monkeypatch.setattr(
        "datasheetindex.index.enrich_with_table_counts",
        lambda _nodes, _doc, **_kw: _nodes,
    )
    monkeypatch.setattr(
        "datasheetindex.index.enrich_with_continued_tables",
        lambda _nodes, _text: _nodes,
    )
    monkeypatch.setattr(
        "datasheetindex.index.enrich_with_footnote_markers",
        lambda _nodes, _text: _nodes,
    )
    monkeypatch.setattr(
        "datasheetindex.index.enrich_with_cross_references",
        lambda _nodes, _text: _nodes,
    )
    monkeypatch.setattr(
        "datasheetindex.index.assess_toc_quality",
        lambda _nodes, _total_pages: TocQuality(
            score=1.0, entry_count=0, max_depth=0, page_coverage=0.0
        ),
    )

    for blank in ("", "   ", "\t"):
        idx = DatasheetIndex("dummy.pdf")
        artifacts = idx.build(output_dir=blank)
        idx.close()
        assert artifacts.json_path is not None
        assert artifacts.json_path.parent == pinned


# --- POSIX path translation for a Windows-hosted server (see _resolve_local_path) ---


def test_resolve_local_path_is_a_noop_off_windows(monkeypatch):
    """A POSIX host must never rewrite a POSIX path."""
    monkeypatch.setattr(index_module, "_is_windows", lambda: False)

    def _explode():
        raise AssertionError("distros queried on a non-Windows host")

    monkeypatch.setattr(index_module, "_wsl_distros", _explode)
    assert index_module._resolve_local_path("/home/u/ds.pdf") == "/home/u/ds.pdf"


def test_resolve_local_path_leaves_existing_paths_alone(tmp_path, monkeypatch):
    """Translation is a fallback. A path that resolves is never second-guessed,
    so a genuine Windows path can't be mangled into a UNC one."""
    pdf = tmp_path / "ds.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(index_module, "_is_windows", lambda: True)

    def _explode():
        raise AssertionError("distros queried for a path that already exists")

    monkeypatch.setattr(index_module, "_wsl_distros", _explode)
    assert index_module._resolve_local_path(str(pdf)) == str(pdf)


def test_windows_paths_maps_mnt_to_a_drive(monkeypatch):
    """/mnt/c is WSL's own mount of C:, so it maps back exactly -- no guessing,
    and it must be preferred over the UNC form which would round-trip the file
    back through the distro."""
    monkeypatch.setattr(index_module, "_wsl_distros", lambda: ["Ubuntu"])
    candidates = list(index_module._windows_paths_for_posix("/mnt/c/Users/y/ds.pdf"))
    assert candidates[0] == "C:\\Users\\y\\ds.pdf"


def test_windows_paths_uses_unc_for_distro_paths(monkeypatch):
    monkeypatch.setattr(index_module, "_wsl_distros", lambda: ["Ubuntu", "Debian"])
    assert list(index_module._windows_paths_for_posix("/home/y/ds.pdf")) == [
        "\\\\wsl.localhost\\Ubuntu\\home\\y\\ds.pdf",
        "\\\\wsl.localhost\\Debian\\home\\y\\ds.pdf",
    ]


def test_resolve_local_path_picks_the_candidate_that_exists(monkeypatch):
    """With several distros installed, the one actually holding the file wins."""
    monkeypatch.setattr(index_module, "_is_windows", lambda: True)
    monkeypatch.setattr(index_module, "_wsl_distros", lambda: ["Ubuntu", "Debian"])
    found = "\\\\wsl.localhost\\Debian\\home\\y\\ds.pdf"
    monkeypatch.setattr(index_module.os.path, "exists", lambda p: p == found)

    assert index_module._resolve_local_path("/home/y/ds.pdf") == found


def test_resolve_local_path_returns_original_when_nothing_matches(monkeypatch):
    """The error the user sees must name the path they passed, not a rewrite."""
    monkeypatch.setattr(index_module, "_is_windows", lambda: True)
    monkeypatch.setattr(index_module, "_wsl_distros", lambda: ["Ubuntu"])
    monkeypatch.setattr(index_module.os.path, "exists", lambda p: False)

    assert index_module._resolve_local_path("/home/y/ds.pdf") == "/home/y/ds.pdf"


def test_windows_paths_handles_degenerate_inputs(monkeypatch):
    """A bare /mnt/c names the mount itself, with nothing after the drive, so
    it has no drive-letter spelling -- only the UNC one."""
    monkeypatch.setattr(index_module, "_wsl_distros", lambda: ["Ubuntu"])

    assert list(index_module._windows_paths_for_posix("")) == []
    assert list(index_module._windows_paths_for_posix("/")) == []
    assert list(index_module._windows_paths_for_posix("/mnt/c")) == [
        "\\\\wsl.localhost\\Ubuntu\\mnt\\c"
    ]


def test_mnt_candidate_does_not_query_wsl(monkeypatch):
    """Probing a UNC path against a stopped distro STARTS it -- tens of seconds,
    and the one unbounded step on this path. A /mnt path maps back exactly, so
    resolving it must never reach for the distro list at all."""
    monkeypatch.setattr(index_module, "_is_windows", lambda: True)

    def _explode():
        raise AssertionError("queried WSL despite an exact /mnt mapping")

    monkeypatch.setattr(index_module, "_wsl_distros", _explode)
    monkeypatch.setattr(
        index_module.os.path, "exists", lambda p: p == "C:\\Users\\y\\ds.pdf"
    )

    assert (
        index_module._resolve_local_path("/mnt/c/Users/y/ds.pdf")
        == "C:\\Users\\y\\ds.pdf"
    )


def test_wsl_query_is_bounded_and_windowless(monkeypatch):
    """The query must not use subprocess.run(capture_output=True): on timeout
    that is kill() then communicate(), which waits for an EOF a lingering WSL
    helper may never deliver -- so the timeout would bound nothing."""
    seen = {}

    class _Proc:
        def __init__(self, cmd, **kwargs):
            seen["cmd"] = cmd
            seen["kwargs"] = kwargs

        def wait(self, timeout=None):
            seen["timeout"] = timeout
            return 0

        def kill(self):  # pragma: no cover - not reached in this test
            seen["killed"] = True

    monkeypatch.setattr(index_module.subprocess, "Popen", _Proc)

    assert index_module._wsl_distros() == []
    assert seen["cmd"] == ["wsl.exe", "--list", "--quiet"]
    assert seen["timeout"] == index_module.WSL_QUERY_TIMEOUT_SECONDS
    assert seen["kwargs"]["stdout"] is not index_module.subprocess.PIPE
    assert seen["kwargs"]["creationflags"] == getattr(
        index_module.subprocess, "CREATE_NO_WINDOW", 0
    )


def _install_fake_wsl(
    monkeypatch, *, stdout=b"", returncode=0, popen_raises=None, wait_raises=None
):
    """Stub wsl.exe at the Popen seam the real query uses."""
    # Annotated: heterogeneous values, and an inferred union makes every use
    # of them a type error.
    state: dict[str, Any] = {"kills": 0, "waits": []}

    class _Proc:
        def __init__(self, cmd, **kwargs):
            if popen_raises is not None:
                raise popen_raises
            kwargs["stdout"].write(stdout)
            kwargs["stdout"].flush()

        def wait(self, timeout=None):
            state["waits"].append(timeout)
            if wait_raises is not None and len(state["waits"]) == 1:
                raise wait_raises
            return returncode

        def kill(self):
            state["kills"] += 1

    monkeypatch.setattr(index_module.subprocess, "Popen", _Proc)
    return state


def test_wsl_distros_decodes_utf16(monkeypatch):
    """wsl.exe emits UTF-16LE by default; decoding it as UTF-8 yields NUL-laced
    names that match no path while still looking plausible in a log."""
    _install_fake_wsl(monkeypatch, stdout="Ubuntu\nDebian\n".encode("utf-16-le"))
    assert index_module._wsl_distros() == ["Ubuntu", "Debian"]


def test_wsl_distros_decodes_utf8_when_wsl_utf8_is_set(monkeypatch):
    """WSL 0.64+ emits UTF-8 when WSL_UTF8=1, and that output is even-length, so
    decoding it as UTF-16 does NOT raise -- it silently yields one mojibake name
    that matches nothing, disabling the feature with no error anywhere."""
    _install_fake_wsl(monkeypatch, stdout=b"Ubuntu\r\nDebian\r\n")
    assert index_module._wsl_distros() == ["Ubuntu", "Debian"]


def test_wsl_distros_ignores_names_that_would_build_odd_paths(monkeypatch):
    _install_fake_wsl(monkeypatch, stdout="Ubuntu\n..\nbad\\name\n".encode("utf-16-le"))
    assert index_module._wsl_distros() == ["Ubuntu"]


def test_wsl_distros_returns_empty_on_nonzero_exit(monkeypatch):
    _install_fake_wsl(monkeypatch, stdout=b"", returncode=1)
    assert index_module._wsl_distros() == []


def test_wsl_distros_survives_missing_wsl(monkeypatch):
    """A Windows host with no WSL must degrade quietly, not raise."""
    _install_fake_wsl(monkeypatch, popen_raises=FileNotFoundError("wsl.exe"))
    assert index_module._wsl_distros() == []


# --- The mirror direction: a Windows path handed to a WSL-hosted server ---


def test_posix_paths_maps_a_drive_letter_to_the_mount():
    """Runs natively on the Linux CI lane -- this is the POSIX branch, so unlike
    the Windows direction it needs no platform monkeypatching to be meaningful."""
    assert list(index_module._posix_paths_for_windows("C:\\Users\\y\\ds.pdf")) == [
        "/mnt/c/Users/y/ds.pdf"
    ]


def test_posix_paths_accepts_forward_slashes_and_lowercase_drives():
    assert list(index_module._posix_paths_for_windows("d:/Data/ds.pdf")) == [
        "/mnt/d/Data/ds.pdf"
    ]


def test_posix_paths_ignores_paths_that_are_not_windows_paths():
    for value in ("/home/y/ds.pdf", "relative/ds.pdf", "", "ds.pdf"):
        assert list(index_module._posix_paths_for_windows(value)) == []


def test_resolve_local_path_translates_a_windows_path_on_a_posix_host(monkeypatch):
    monkeypatch.setattr(index_module, "_is_windows", lambda: False)
    monkeypatch.setattr(
        index_module.os.path, "exists", lambda p: p == "/mnt/c/Users/y/ds.pdf"
    )

    assert (
        index_module._resolve_local_path("C:\\Users\\y\\ds.pdf")
        == "/mnt/c/Users/y/ds.pdf"
    )


def test_resolve_local_path_keeps_a_windows_path_when_nothing_matches(monkeypatch):
    """The error must still name what the caller passed."""
    monkeypatch.setattr(index_module, "_is_windows", lambda: False)
    monkeypatch.setattr(index_module.os.path, "exists", lambda p: False)

    original = "C:\\Users\\y\\ds.pdf"
    assert index_module._resolve_local_path(original) == original


def test_resolve_local_path_never_rewrites_a_posix_path_that_exists(
    tmp_path, monkeypatch
):
    """The POSIX host must not start second-guessing paths that already work."""
    monkeypatch.setattr(index_module, "_is_windows", lambda: False)
    pdf = tmp_path / "ds.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    assert index_module._resolve_local_path(str(pdf)) == str(pdf)


def test_wsl_query_kills_a_wedged_child(monkeypatch):
    """A timeout that leaves wsl.exe running is the orphan this rewrite exists
    to prevent, and every wait after the kill must still be bounded."""
    state = _install_fake_wsl(
        monkeypatch, wait_raises=subprocess.TimeoutExpired("wsl.exe", 5)
    )

    assert index_module._wsl_distros() == []
    assert state["kills"] == 1
    assert len(state["waits"]) == 2
    assert all(timeout is not None for timeout in state["waits"])


def test_wsl_query_kills_the_child_on_a_non_timeout_interruption(monkeypatch):
    """KeyboardInterrupt is not an OSError/SubprocessError, so it propagates --
    but the child must still be killed on the way out. This is the pair that
    pins `except BaseException` rather than `except TimeoutExpired`."""
    state = _install_fake_wsl(monkeypatch, wait_raises=KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        index_module._wsl_distros()

    assert state["kills"] == 1


def test_posix_paths_refuses_another_distros_unc_path(monkeypatch):
    """The share is distro-scoped. Stripping the prefix blindly turns Debian's
    file into our /home/y/ds.pdf, which on a machine with the same user in two
    distros silently resolves to a different document that exists."""
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")

    assert (
        list(
            index_module._posix_paths_for_windows(
                "\\\\wsl.localhost\\Debian\\home\\y\\ds.pdf"
            )
        )
        == []
    )


def test_posix_paths_unwraps_our_own_distro_case_insensitively(monkeypatch):
    """Windows paths are case-insensitive, so \\\\WSL.localhost and a differently
    cased distro name are the same share."""
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")

    for value in (
        "\\\\wsl.localhost\\Ubuntu\\home\\y\\ds.pdf",
        "\\\\WSL.localhost\\ubuntu\\home\\y\\ds.pdf",
        "\\\\wsl$\\Ubuntu\\home\\y\\ds.pdf",
    ):
        assert list(index_module._posix_paths_for_windows(value)) == ["/home/y/ds.pdf"]


def test_posix_paths_yields_nothing_without_a_known_distro(monkeypatch):
    """Off WSL there is no way to tell whose filesystem the path names."""
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)

    assert (
        list(
            index_module._posix_paths_for_windows(
                "\\\\wsl.localhost\\Ubuntu\\home\\y\\ds.pdf"
            )
        )
        == []
    )


def test_posix_paths_rejects_bare_roots(monkeypatch):
    """A bare root maps to a directory that exists, so _resolve_local_path would
    "resolve" to it and the not-found error would name a path nobody passed."""
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")

    assert list(index_module._posix_paths_for_windows("C:\\")) == []
    assert (
        list(index_module._posix_paths_for_windows("\\\\wsl.localhost\\Ubuntu\\")) == []
    )


def _simple_pdf(tmp_path, name="ds.pdf", pages=2, with_toc=True):
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page()
        writer = pymupdf.TextWriter(page.rect)
        writer.append((72, 72), "Body text for this page of the datasheet")
        writer.write_text(page)
    if with_toc:
        doc.set_toc([[1, "Overview", 1], [1, "Electrical Characteristics", 2]])
    pdf_path = tmp_path / name
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_artifact_stem_matches_the_written_filenames(tmp_path):
    """build_datasheet uses this to find the sidecar; it must not drift."""
    pdf_path = _simple_pdf(tmp_path, name="My Datasheet.pdf")

    idx = DatasheetIndex(str(pdf_path))
    try:
        stem = idx.artifact_stem(None)
        artifacts = idx.build(output_dir=str(tmp_path / "out"))
        override = idx.artifact_stem("custom stem")
        overridden = idx.build(
            output_dir=str(tmp_path / "out2"), output_stem="custom stem"
        )
    finally:
        idx.close()

    assert artifacts.json_path is not None
    assert artifacts.text_path is not None
    assert overridden.json_path is not None
    assert overridden.text_path is not None
    assert artifacts.json_path.name == f"{stem}.json"
    assert artifacts.text_path.name == f"{stem}.txt"
    assert overridden.json_path.name == f"{override}.json"
    assert overridden.text_path.name == f"{override}.txt"


def test_deliverables_match_the_returned_values_byte_for_byte(tmp_path):
    """Atomic writes must not change a single byte of either deliverable."""
    pdf_path = _simple_pdf(tmp_path)

    idx = DatasheetIndex(str(pdf_path))
    try:
        artifacts = idx.build(output_dir=str(tmp_path / "out"))
    finally:
        idx.close()

    assert artifacts.json_path is not None
    assert artifacts.text_path is not None
    expected_json = json.dumps(artifacts.json_data, indent=2, ensure_ascii=False)
    assert artifacts.json_path.read_text(encoding="utf-8") == expected_json
    assert artifacts.text_path.read_text(encoding="utf-8") == artifacts.text_content
    # The sidecar's TocQuality carries `details`; the deliverable must not.
    assert "details" not in artifacts.json_data["toc_quality"]
    assert sorted(artifacts.json_data["toc_quality"]) == [
        "entry_count",
        "max_depth",
        "page_coverage",
        "recommend_summaries",
        "score",
    ]


def test_no_temp_files_are_left_in_the_output_directory(tmp_path):
    pdf_path = _simple_pdf(tmp_path)
    out = tmp_path / "out"

    idx = DatasheetIndex(str(pdf_path))
    try:
        idx.build(output_dir=str(out))
    finally:
        idx.close()

    assert sorted(p.name for p in out.iterdir()) == ["ds.json", "ds.txt"]


def test_a_failed_text_write_leaves_the_previous_generation_readable(
    tmp_path, monkeypatch
):
    """The reason for atomic writes: no truncated deliverable on disk."""
    pdf_path = _simple_pdf(tmp_path)
    out = tmp_path / "out"

    idx = DatasheetIndex(str(pdf_path))
    try:
        first = idx.build(output_dir=str(out))
        assert first.text_path is not None
        assert first.json_path is not None
        first_text = first.text_path.read_text(encoding="utf-8")
        real_write = index_module.atomic_write_text

        def failing_write(path, content):
            if path.suffix == ".txt":
                raise OSError("disk full")
            real_write(path, content)

        monkeypatch.setattr(index_module, "atomic_write_text", failing_write)

        with pytest.raises(OSError):
            idx.build(output_dir=str(out))
    finally:
        idx.close()

    assert first.text_path.read_text(encoding="utf-8") == first_text
    json.loads(first.json_path.read_text(encoding="utf-8"))
    assert not any(p.name.endswith(".tmp") for p in out.iterdir())


def test_no_obtainable_client_marks_toc_fallback_pending(tmp_path, monkeypatch):
    """No credentials is an environment fact, not a failed build.

    A build eligible for the ToC fallback that finds no LLM client must not be
    cached as incomplete -- rebuilding cannot create credentials, so that would
    cost a full rebuild on every request forever. It is recorded separately as
    ``toc_fallback_pending`` instead, mirroring ``figure_captions_pending``.
    """
    pdf_path = _simple_pdf(tmp_path, name="weak.pdf", pages=3, with_toc=False)
    monkeypatch.setattr(
        DatasheetIndex, "_try_create_default_llm_client", lambda _self: None
    )

    idx = DatasheetIndex(str(pdf_path))
    try:
        artifacts = idx.build(output_dir=str(tmp_path / "out"))
    finally:
        idx.close()

    assert artifacts.toc_quality is not None
    assert artifacts.toc_quality.score < TOC_FALLBACK_THRESHOLD
    assert artifacts.toc_fallback_pending is True
    assert artifacts.llm_enrichment_incomplete is False


def test_a_raising_fallback_marks_enrichment_incomplete(tmp_path, monkeypatch):
    """One bad network moment must not produce a permanently cacheable artifact."""
    pdf_path = _simple_pdf(tmp_path, name="weak.pdf", pages=3, with_toc=False)

    def dummy_callable(_system, _user):
        return "unused"

    def raising_fallback(_text, _pages, _callable, **_kw):
        raise RuntimeError("gateway timeout")

    monkeypatch.setattr(
        DatasheetIndex, "_try_create_default_llm_client", lambda _self: dummy_callable
    )
    monkeypatch.setattr(
        "datasheetindex.llm.toc_fallback.generate_toc_from_text", raising_fallback
    )

    idx = DatasheetIndex(str(pdf_path))
    try:
        artifacts = idx.build(output_dir=str(tmp_path / "out"))
    finally:
        idx.close()

    assert artifacts.llm_enrichment_incomplete is True
    assert "toc_fallback_raised" in artifacts.llm_enrichment_notes
    # The build still succeeded on the native ToC.
    assert artifacts.json_path is not None
    assert artifacts.json_path.exists()


def test_a_tls_failure_fails_the_build_rather_than_becoming_a_note(
    tmp_path, monkeypatch
):
    """The one fallback failure worth stopping the build for.

    Every other failure degrades to ``toc_fallback_raised`` and a usable
    artifact, which is right: they are transient, and a native ToC is better
    than no index. A certificate the trust store rejects is neither transient
    nor invisible-by-necessity -- it recurs on every build, is fixed in one
    variable, and its note would sit unread inside an artifact that looks fine.
    Raising is what turns a missed ``LITELLM_TLS_VERIFY`` migration into
    something the operator finds out about on the first build.
    """
    from datasheetindex.llm.client import LlmTlsVerificationError

    pdf_path = _simple_pdf(tmp_path, name="weak.pdf", pages=3, with_toc=False)

    def dummy_callable(_system, _user):
        return "unused"

    def tls_failure(_text, _pages, _callable, **_kw):
        raise LlmTlsVerificationError("certificate verify failed")

    monkeypatch.setattr(
        DatasheetIndex, "_try_create_default_llm_client", lambda _self: dummy_callable
    )
    monkeypatch.setattr(
        "datasheetindex.llm.toc_fallback.generate_toc_from_text", tls_failure
    )

    idx = DatasheetIndex(str(pdf_path))
    try:
        # The message is pinned in tests/test_llm_client.py; what matters here
        # is that build() does not convert it into an enrichment note.
        with pytest.raises(LlmTlsVerificationError):
            idx.build(output_dir=str(tmp_path / "out"))
    finally:
        idx.close()


def test_a_rejected_fallback_candidate_is_complete(tmp_path, monkeypatch):
    """except versus else: a candidate declined on the merits is a decision.

    Marking it incomplete would re-pay the LLM cost on every request for
    exactly the documents the fallback cannot help.

    The rejection is driven by stubbing _accept_llm_toc_candidate rather than
    by constructing a candidate the real heuristic happens to decline. With no
    baseline ToC to protect, that heuristic accepts even a single thin node
    (see test_accept_llm_toc_candidate_accepts_thin_candidate_without_baseline),
    so a fixture-based rejection would silently test acceptance instead.
    """
    pdf_path = _simple_pdf(tmp_path, name="weak.pdf", pages=3, with_toc=False)

    def dummy_callable(_system, _user):
        return "unused"

    monkeypatch.setattr(
        DatasheetIndex, "_try_create_default_llm_client", lambda _self: dummy_callable
    )
    monkeypatch.setattr(
        "datasheetindex.llm.toc_fallback.generate_toc_from_text",
        lambda _text, _pages, _callable, **_kw: [
            TocNode(
                title="Candidate", level=1, start_page=1, end_page=3, node_id="0001"
            )
        ],
    )
    monkeypatch.setattr(
        "datasheetindex.index._accept_llm_toc_candidate",
        lambda _baseline, _candidate, *, total_pages, explicit_request=False: (
            False,
            "declined for this test",
        ),
    )

    idx = DatasheetIndex(str(pdf_path))
    try:
        artifacts = idx.build(output_dir=str(tmp_path / "out"))
    finally:
        idx.close()

    # The else branch ran: the candidate was discarded, so its node is absent.
    assert all(node.title != "Candidate" for node in artifacts.nodes)
    assert artifacts.llm_enrichment_incomplete is False
    assert artifacts.llm_enrichment_notes == ()


def test_a_good_toc_is_complete_without_any_llm(tmp_path):
    """The common path must not be permanently uncacheable."""
    pdf_path = _simple_pdf(tmp_path, pages=3)

    idx = DatasheetIndex(str(pdf_path))
    try:
        artifacts = idx.build(output_dir=str(tmp_path / "out"))
    finally:
        idx.close()

    assert artifacts.toc_quality is not None
    assert artifacts.toc_quality.score >= TOC_FALLBACK_THRESHOLD
    assert artifacts.llm_enrichment_incomplete is False
    assert artifacts.llm_enrichment_notes == ()


def test_build_emits_figures_and_exclusions(tmp_path):
    pdf = tmp_path / "fig.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20))
    pix.set_rect(pix.irect, (0, 0, 255))
    page.insert_image(pymupdf.Rect(100, 100, 400, 400), pixmap=pix)
    page.insert_image(pymupdf.Rect(10, 10, 25, 25), pixmap=pix)  # below 1.0%
    doc.save(pdf)
    doc.close()

    with DatasheetIndex(str(pdf)) as index:
        artifacts = index.build(output_dir=str(tmp_path / "out"))

    data = artifacts.json_data
    assert [entry["kind"] for entry in data["figures"]] == ["raster"]
    assert data["figures_excluded"] == {
        "below_min_area_pct": 1,
        "min_area_pct": 1.0,
    }
    assert artifacts.json_path is not None
    on_disk = json.loads(artifacts.json_path.read_text(encoding="utf-8"))
    assert on_disk["figures"] == data["figures"]


def test_document_with_no_figures_emits_an_empty_array(tmp_path):
    # Key presence distinguishes "none found" from "predates the feature".
    pdf = tmp_path / "plain.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    writer = pymupdf.TextWriter(page.rect)
    writer.append((72, 72), "Supply voltage 5V")
    writer.write_text(page)
    doc.save(pdf)
    doc.close()

    with DatasheetIndex(str(pdf)) as index:
        artifacts = index.build(output_dir=str(tmp_path / "out"))

    assert artifacts.json_data["figures"] == []
    assert artifacts.json_data["figures_excluded"]["below_min_area_pct"] == 0


def test_a_page_with_both_an_image_and_a_caption_yields_two_entries(tmp_path):
    # Pins the no-merge decision: a future change must not start associating
    # a caption with a region by proximity.
    pdf = tmp_path / "both.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20))
    pix.set_rect(pix.irect, (0, 0, 255))
    page.insert_image(pymupdf.Rect(100, 300, 400, 600), pixmap=pix)
    writer = pymupdf.TextWriter(page.rect)
    writer.append((72, 72), "Figure 7. Block diagram")
    writer.write_text(page)
    doc.save(pdf)
    doc.close()

    with DatasheetIndex(str(pdf)) as index:
        artifacts = index.build(output_dir=str(tmp_path / "out"))

    kinds = sorted(entry["kind"] for entry in artifacts.json_data["figures"])
    assert kinds == ["caption", "raster"]


def test_negative_max_figure_captions_raises(tmp_path):
    pdf = tmp_path / "x.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(pdf)
    doc.close()

    with DatasheetIndex(str(pdf)) as index:
        with pytest.raises(ValueError, match="max_figure_captions"):
            index.build(output_dir=str(tmp_path / "out"), max_figure_captions=-1)
        with pytest.raises(ValueError, match="max_figure_captions"):
            # bool is a subclass of int -- True would silently become a cap
            # of 1 if this were not rejected explicitly.
            index.build(output_dir=str(tmp_path / "out"), max_figure_captions=True)
        with pytest.raises(ValueError, match="max_figure_captions"):
            # A non-int (e.g. float) must be rejected here rather than
            # reaching candidates[:2.5] and raising TypeError deep inside
            # figure_captions.py.
            index.build(
                output_dir=str(tmp_path / "out"),
                max_figure_captions=2.5,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            )


def test_caption_figures_true_with_no_model_does_not_raise(tmp_path):
    # Unlike include_summaries. With the default True, that guard would make
    # every keyless build raise -- destroying the fallback this design needs.
    pdf = tmp_path / "y.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(pdf)
    doc.close()

    with DatasheetIndex(str(pdf)) as index:
        artifacts = index.build(output_dir=str(tmp_path / "out"))

    assert artifacts.json_data["figures"] == []


def test_figure_captions_excluded_is_always_emitted(tmp_path):
    pdf = tmp_path / "z.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(pdf)
    doc.close()

    with DatasheetIndex(str(pdf)) as index:
        artifacts = index.build(output_dir=str(tmp_path / "out"))

    assert artifacts.json_data["figure_captions_excluded"] == {
        "above_max": 0,
        "max_figure_captions": 20,
    }


class _RecordingLlmClient:
    """A vision-capable client that records which LLM path called it.

    One object serving both interfaces is the point: the summaries branch and
    the captioning branch share a client by design, so only a recorder that
    can tell their calls apart can show which branches ran.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, system: str, user: str) -> str:
        self.calls.append("summary")
        return "a section summary"

    def describe_image(
        self, system: str, image_base64: str, *, media_type: str = "image/png"
    ) -> str:
        self.calls.append("caption")
        return "a block diagram of the device"

    def close(self) -> None:
        pass


def _figure_pdf(tmp_path, name="fig-toc.pdf"):
    """A good-ToC document carrying one raster region above ``min_area_pct``.

    The raster is what makes ``build()`` self-create a client at all, so a
    keyless summaries test written against ``_simple_pdf`` would pass whatever
    the gate did. Each page carries well over ``summarizer.MIN_TEXT_LENGTH``
    characters for the same reason on the other side: a thin page is skipped by
    the summarizer, so "no summary call" would prove nothing.
    """
    doc = pymupdf.open()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20))
    pix.set_rect(pix.irect, (10, 20, 30))
    # A single flat colour renders as one distinct colour end to end, which
    # the blank-region guard (figure_captions._is_blank_region) treats as
    # nothing to caption and skips before ever dispatching -- an accent
    # stripe keeps this fixture's raster region below that threshold, same
    # as a real figure's non-uniform content would.
    pix.set_rect(pymupdf.IRect(0, 0, 20, 1), (200, 200, 200))
    for number in range(2):
        page = doc.new_page(width=595, height=842)
        writer = pymupdf.TextWriter(page.rect)
        for line in range(6):
            writer.append(
                (72, 72 + line * 14),
                f"Body text line {line} for this page of the datasheet",
            )
        writer.write_text(page)
        if number == 1:
            page.insert_image(pymupdf.Rect(50, 200, 545, 600), pixmap=pix)
    doc.set_toc([[1, "Overview", 1], [1, "Electrical Characteristics", 2]])
    pdf_path = tmp_path / name
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_keyless_build_runs_no_summaries_even_with_a_figure(tmp_path, monkeypatch):
    """Figure presence must not switch summaries on.

    ``include_summaries=True`` with no ``llm_callable`` produced nothing
    through 0.24.0, because the only self-created client came from the
    weak-ToC branch and this document does not need it. Captioning added a
    second construction site, and gating summaries on the client's mere
    existence let a single inserted image start one LLM call per ToC section --
    unbounded by ``max_figure_captions`` and disclosed nowhere.
    """
    pdf_path = _figure_pdf(tmp_path)
    client = _RecordingLlmClient()
    monkeypatch.setattr(
        DatasheetIndex, "_try_create_default_llm_client", lambda _self: client
    )

    with DatasheetIndex(str(pdf_path)) as index:
        artifacts = index.build(
            output_dir=str(tmp_path / "out"), include_summaries=True
        )

    assert "summary" not in client.calls, (
        "a keyless build ran summaries; the self-created caption client must "
        "not sanction them"
    )
    assert all(node.summary == "" for node in artifacts.nodes)


def test_keyless_build_still_captions_its_figures(tmp_path, monkeypatch):
    """The other side of the gate: captioning is the whole point of self-creation.

    Fails if someone "fixes" the summaries leak by refusing to self-create a
    client for captions at all.
    """
    pdf_path = _figure_pdf(tmp_path)
    client = _RecordingLlmClient()
    monkeypatch.setattr(
        DatasheetIndex, "_try_create_default_llm_client", lambda _self: client
    )

    with DatasheetIndex(str(pdf_path)) as index:
        artifacts = index.build(
            output_dir=str(tmp_path / "out"), include_summaries=True
        )

    assert client.calls == ["caption"]
    rasters = [f for f in artifacts.json_data["figures"] if f["kind"] == "raster"]
    assert [f["caption_source"] for f in rasters] == ["llm"]
    assert artifacts.figure_captions_pending == 0


def test_a_supplied_client_does_run_summaries(tmp_path, monkeypatch):
    """A caller who hands in a client asked for the LLM work explicitly."""
    pdf_path = _figure_pdf(tmp_path)
    client = _RecordingLlmClient()
    monkeypatch.setattr(
        DatasheetIndex,
        "_try_create_default_llm_client",
        lambda _self: pytest.fail("a supplied client must suppress construction"),
    )

    with DatasheetIndex(str(pdf_path)) as index:
        artifacts = index.build(
            output_dir=str(tmp_path / "out"),
            include_summaries=True,
            llm_callable=client,
        )

    assert "summary" in client.calls
    assert any(node.summary == "a section summary" for node in artifacts.nodes)
    assert "caption" in client.calls, "a supplied vision client must still caption"


def test_preamble_pages_is_emitted(tmp_path):
    """One entry per page read, with the signal fields present."""
    pdf_path = tmp_path / "two.pdf"
    doc = pymupdf.open()
    for _ in range(2):
        page = doc.new_page()
        page.insert_textbox(
            pymupdf.Rect(50, 50, 500, 300),
            "Features\n- one\n- two",
            fontsize=10,
        )
    doc.save(str(pdf_path))
    doc.close()

    with DatasheetIndex(str(pdf_path)) as index:
        artifacts = index.build(output_dir=str(tmp_path / "out"))

    data = artifacts.json_data
    assert [p["page"] for p in data["preamble_pages"]] == [1, 2]
    for entry in data["preamble_pages"]:
        assert set(entry) == {
            "page",
            "chars",
            "bullets",
            "has_features_heading",
        }
    assert data["preamble"].startswith("--- PAGE 1 ---")


def test_single_page_document_yields_one_preamble_page(tmp_path):
    """A one-page document must still produce a one-entry preamble_pages."""
    pdf_path = tmp_path / "one.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(50, 50, 500, 200), "Only page", fontsize=10)
    doc.save(str(pdf_path))
    doc.close()

    with DatasheetIndex(str(pdf_path)) as index:
        artifacts = index.build(output_dir=str(tmp_path / "out"))

    data = artifacts.json_data
    assert len(data["preamble_pages"]) == 1
    assert "later pages were not examined" not in data["preamble"]
