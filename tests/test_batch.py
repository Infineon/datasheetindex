"""Tests for batch processing."""

from pathlib import Path

import pymupdf
import pytest

from datasheetindex.batch import BatchResult, build_batch
from datasheetindex.models import DatasheetArtifacts

DATA2PAGE_DIR = Path(__file__).resolve().parent.parent.parent / "data2page"
TLE9350_PATH = DATA2PAGE_DIR / "Infineon-TLE9350BSJ-DataSheet-v01_00-EN.pdf"


def _make_pdf(path: Path, text: str) -> Path:
    """Write a one-page PDF carrying a line of text."""
    doc = pymupdf.open()
    page = doc.new_page()
    writer = pymupdf.TextWriter(page.rect)
    writer.append((72, 72), text)
    writer.write_text(page)
    doc.save(str(path))
    doc.close()
    return path


def test_empty_list(tmp_path):
    result = build_batch([], output_dir=str(tmp_path))
    assert isinstance(result, BatchResult)
    assert result.total == 0
    assert result.success_count == 0
    assert result.failure_count == 0


@pytest.mark.real_pdf
def test_single_pdf(tmp_path):
    if not TLE9350_PATH.exists():
        pytest.skip("Test PDF not found")
    result = build_batch([str(TLE9350_PATH)], output_dir=str(tmp_path))
    assert result.success_count == 1
    assert result.failure_count == 0
    assert result.total == 1
    assert result.succeeded[0].json_path is not None


def test_nonexistent_pdf(tmp_path):
    result = build_batch(["nonexistent.pdf"], output_dir=str(tmp_path))
    assert result.success_count == 0
    assert result.failure_count == 1
    assert result.failed[0].pdf_path == "nonexistent.pdf"
    assert result.failed[0].error != ""


@pytest.mark.real_pdf
def test_mixed_success_failure(tmp_path):
    if not TLE9350_PATH.exists():
        pytest.skip("Test PDF not found")
    result = build_batch(
        [str(TLE9350_PATH), "does_not_exist.pdf"],
        output_dir=str(tmp_path),
    )
    assert result.success_count == 1
    assert result.failure_count == 1
    assert result.total == 2


def test_all_failures(tmp_path):
    result = build_batch(
        ["bad1.pdf", "bad2.pdf", "bad3.pdf"],
        output_dir=str(tmp_path),
    )
    assert result.success_count == 0
    assert result.failure_count == 3


def test_multiple_pdfs(tmp_path):
    """Two documents build in one batch, against real DatasheetIndex instances.

    Synthetic rather than real datasheets on purpose. Batch is content-agnostic
    -- it loops, allocates output stems, and closes each document -- so nothing
    here needs a vendor PDF's quirks, and requiring two of them is what made
    this test skip on every machine and in CI alike: `data2page` carries one
    datasheet, so the second `.exists()` never held and the whole multi-document
    path went permanently uncovered while reading as coverage.

    This is also the only test that drives two *real* builds through
    `build_batch`. `test_duplicate_stems_get_unique_output_names` asserts the
    same counts but monkeypatches `DatasheetIndex` away, so it never exercises
    real construction, build, and the per-document `close()` in the loop's
    `finally`.
    """
    pdf_a = _make_pdf(tmp_path / "alpha.pdf", "Supply voltage 4.5V to 5.5V")
    pdf_b = _make_pdf(tmp_path / "bravo.pdf", "Operating temperature -40C to 125C")
    output_dir = tmp_path / "out"

    result = build_batch([str(pdf_a), str(pdf_b)], output_dir=str(output_dir))

    assert result.success_count == 2
    assert result.failure_count == 0
    assert result.total == 2
    # Distinct stems must produce distinct artifacts -- the counts alone would
    # pass even if both documents wrote over one output name.
    stems = sorted(a.json_path.stem for a in result.succeeded if a.json_path)
    assert stems == ["alpha", "bravo"]
    for artifact in result.succeeded:
        assert artifact.json_path is not None and artifact.json_path.exists()
        assert artifact.text_path is not None and artifact.text_path.exists()


def test_build_batch_forwards_caption_options(monkeypatch, tmp_path):
    """caption_figures/max_figure_captions must reach DatasheetIndex.build,
    the same way include_summaries and llm_callable already do."""
    captured_kwargs: list[dict[str, object]] = []

    class _FakeIndex:
        def __init__(self, pdf_path: str) -> None:
            self.pdf_path = pdf_path

        def _output_stem(self) -> str:
            return Path(self.pdf_path).stem

        def build(
            self,
            output_dir: str = "output",
            include_summaries: bool = False,
            llm_callable=None,
            output_stem: str | None = None,
            caption_figures: bool = True,
            max_figure_captions: int = 20,
        ) -> DatasheetArtifacts:
            captured_kwargs.append(
                {
                    "caption_figures": caption_figures,
                    "max_figure_captions": max_figure_captions,
                }
            )
            stem = output_stem or self._output_stem()
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            json_path = out / f"{stem}.json"
            text_path = out / f"{stem}.txt"
            json_path.write_text(self.pdf_path, encoding="utf-8")
            text_path.write_text(self.pdf_path, encoding="utf-8")
            return DatasheetArtifacts(json_path=json_path, text_path=text_path)

        def close(self) -> None:
            pass

    monkeypatch.setattr("datasheetindex.batch.DatasheetIndex", _FakeIndex)

    result = build_batch(
        [str(tmp_path / "a.pdf")],
        output_dir=str(tmp_path / "out"),
        caption_figures=False,
        max_figure_captions=3,
    )

    assert result.success_count == 1
    assert captured_kwargs == [{"caption_figures": False, "max_figure_captions": 3}]


def test_batch_result_properties():
    result = BatchResult()
    assert result.total == 0
    assert result.success_count == 0
    assert result.failure_count == 0


def test_duplicate_stems_get_unique_output_names(monkeypatch, tmp_path):
    output_dir = tmp_path / "out"

    class _FakeIndex:
        def __init__(self, pdf_path: str) -> None:
            self.pdf_path = pdf_path

        def _output_stem(self) -> str:
            return Path(self.pdf_path).stem

        def build(
            self,
            output_dir: str = "output",
            include_summaries: bool = False,
            llm_callable=None,
            output_stem: str | None = None,
            caption_figures: bool = True,
            max_figure_captions: int = 20,
        ) -> DatasheetArtifacts:
            _ = include_summaries, llm_callable, caption_figures, max_figure_captions
            stem = output_stem or self._output_stem()
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            json_path = out / f"{stem}.json"
            text_path = out / f"{stem}.txt"
            json_path.write_text(self.pdf_path, encoding="utf-8")
            text_path.write_text(self.pdf_path, encoding="utf-8")
            return DatasheetArtifacts(json_path=json_path, text_path=text_path)

        def close(self) -> None:
            pass

    monkeypatch.setattr("datasheetindex.batch.DatasheetIndex", _FakeIndex)

    result = build_batch(
        [
            str(tmp_path / "a" / "shared.pdf"),
            str(tmp_path / "b" / "shared.pdf"),
        ],
        output_dir=str(output_dir),
    )

    assert result.success_count == 2
    assert result.failure_count == 0
    json_names: list[str] = []
    for artifact in result.succeeded:
        assert artifact.json_path is not None
        json_names.append(artifact.json_path.name)

    assert json_names == [
        "shared.json",
        "shared-2.json",
    ]
