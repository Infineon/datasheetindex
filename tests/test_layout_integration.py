"""Tests that exercise the real ML layout engine.

Skipped unless the optional [layout] extra is installed. Everything else in the
suite verifies the guard against a fake hook; these are the only tests that
would notice pymupdf4llm changing _use_layout, use_layout(), or the shape of
page.layout_information.
"""

import asyncio
import subprocess
import sys
from pathlib import Path

import pymupdf
import pytest

from datasheetindex.core.engine import classic_tables
from datasheetindex.tools.defs import create_datasheet_tool_session
from tests.test_structure import _expected_classic_counts, _write_mixed_table_pdf

pytest.importorskip("pymupdf.layout")

pytestmark = pytest.mark.layout

HELPER = Path(__file__).parent / "_fresh_layout_process.py"
PAGES = 6


@pytest.fixture
def mixed_pdf(tmp_path: Path) -> Path:
    pdf = tmp_path / "mixed.pdf"
    _write_mixed_table_pdf(pdf, pages=PAGES)
    return pdf


def _classic_counts(pdf: Path) -> dict[int, int]:
    doc = pymupdf.open(str(pdf))
    try:
        with classic_tables():
            return {i: len(doc[i].find_tables().tables) for i in range(len(doc))}
    finally:
        doc.close()


def test_classic_tables_pins_the_real_engine(mixed_pdf):
    """The pin, verified against the ONNX engine rather than a stub.

    The layout engine finds [2,3,2,3,2,3] on this fixture; the classic detector
    finds [1,2,1,2,1,2]. Inside classic_tables() we must see the latter.
    """
    # Activates the hook for this process. Not resolvable by ty in the default
    # lane, where the [layout] extra is deliberately excluded from dev deps.
    import pymupdf4llm  # noqa: F401  # ty: ignore[unresolved-import]

    assert pymupdf._get_layout is not None, "layout hook should be active"
    assert _classic_counts(mixed_pdf) == _expected_classic_counts(PAGES)


def test_markdown_survives_a_classic_tables_round_trip(mixed_pdf, tmp_path):
    """Drives the real handler, which is where a TypeError would surface.

    build_datasheet on a 6-page document takes the sequential path (the
    parallel threshold is 12 pages), so it enters and exits classic_tables().
    """
    session = create_datasheet_tool_session()
    defs = {d.name: d for d in session.defs}
    try:
        build = asyncio.run(
            defs["build_datasheet"].handler(
                {"pdf_source": str(mixed_pdf), "output_dir": str(tmp_path / "out")}
            )
        )
        assert build["is_error"] is False, build

        # A second round-trip through the guard, exactly as a rebuild would do.
        _classic_counts(mixed_pdf)

        result = asyncio.run(defs["extract_table_markdown"].handler({"page": 1}))
    finally:
        session.close()

    # defs.py catches Exception and returns an error envelope, so asserting
    # is_error is False is what makes a TypeError here fail the test. The
    # existing handler tests only assert isinstance(is_error, bool), which
    # passes on a TypeError.
    assert result["is_error"] is False, result
    assert "|" in result["content"][0]["text"]


def test_fresh_process_build_then_extract_table_markdown(mixed_pdf, tmp_path):
    """Acceptance criterion 4: the real public path, in a pristine interpreter.

    Drives DatasheetTools.build_datasheet and .extract_table_markdown, not the
    engine primitives, so a regression in that wiring fails here.

    Cannot be asserted in-process: tests/test_defs.py imports pymupdf4llm and
    sorts before this file, so the precondition is already gone by the time
    pytest reaches us. A subprocess is the only durable guarantee.

    See the helper's docstring for what this does and does not prove -- notably,
    the permanent-TypeError corruption is a thread interleaving and is guarded
    by test_layout_engine_installs_the_hook_under_the_lock, not here.
    """
    proc = subprocess.run(
        [sys.executable, str(HELPER), str(mixed_pdf), str(tmp_path / "fresh-out")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "OK" in proc.stdout
