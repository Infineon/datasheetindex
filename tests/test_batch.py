"""Tests for batch processing."""

from pathlib import Path

import pytest

from datasheetindex.batch import BatchResult, build_batch
from datasheetindex.models import DatasheetArtifacts

DATA2PAGE_DIR = Path(__file__).resolve().parent.parent.parent / "data2page"
TLE9350_PATH = DATA2PAGE_DIR / "Infineon-TLE9350BSJ-DataSheet-v01_00-EN.pdf"
TLE9371_PATH = DATA2PAGE_DIR / "infineon-tle9371vle-datasheet-en.pdf"


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


@pytest.mark.real_pdf
def test_multiple_pdfs(tmp_path):
    if not TLE9350_PATH.exists() or not TLE9371_PATH.exists():
        pytest.skip("Test PDFs not found")
    result = build_batch(
        [str(TLE9350_PATH), str(TLE9371_PATH)],
        output_dir=str(tmp_path),
    )
    assert result.success_count == 2
    assert result.failure_count == 0


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
        ) -> DatasheetArtifacts:
            _ = include_summaries, llm_callable
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
