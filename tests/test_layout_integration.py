"""Tests that exercise the real ML layout engine.

Skipped unless the optional [layout] extra is installed. Everything else in the
suite verifies the guard against a fake hook; these are the only tests that
would notice pymupdf4llm changing _use_layout, use_layout(), or the shape of
page.layout_information.
"""

import asyncio
import inspect
import re
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

# Running header/footer strings for _write_running_header_pdf. Kept distinct
# from every body string so a match in the extracted markdown is unambiguous.
RUNNING_HEADER = "ACME AWC-3200 Motor Controller"
RUNNING_FOOTER = "Datasheet | www.example.invalid"
RUNNING_REVISION = "AWC-3200 Rev. B | 2026-01-15"
BODY_SENTENCE = "The device operates over the full industrial temperature range."


@pytest.fixture
def mixed_pdf(tmp_path: Path) -> Path:
    pdf = tmp_path / "mixed.pdf"
    _write_mixed_table_pdf(pdf, pages=PAGES)
    return pdf


def _write_running_header_pdf(path: Path, pages: int) -> None:
    """A document whose every page carries a running header and footer.

    `_write_mixed_table_pdf` has neither, so it cannot show whether the
    layout engine's page-header/page-footer classes are being honoured.
    Verified against the real ONNX model before this test was written: it
    labels the header and both footer lines on every page of this fixture,
    so the assertions below rest on a measured behaviour rather than an
    assumption about how the model generalizes to synthetic input.
    """
    doc = pymupdf.open()
    for p in range(pages):
        page = doc.new_page()
        page.insert_text((50, 40), RUNNING_HEADER, fontsize=9)
        for row in range(5):
            for col in range(3):
                rect = pymupdf.Rect(
                    50 + col * 90,
                    120 + row * 25,
                    50 + (col + 1) * 90,
                    120 + (row + 1) * 25,
                )
                page.draw_rect(rect, color=(0, 0, 0), width=0.7)
                page.insert_text(
                    (rect.x0 + 3, rect.y0 + 15), f"c{p}{row}{col}", fontsize=7
                )
        page.insert_text((50, 300), BODY_SENTENCE, fontsize=9)
        page.insert_text((50, 760), RUNNING_FOOTER, fontsize=8)
        page.insert_text((500, 760), str(p + 1), fontsize=8)
        page.insert_text((50, 772), RUNNING_REVISION, fontsize=8)
    doc.set_toc([[1, f"Section {i + 1}", i + 1] for i in range(pages)])
    doc.save(str(path))
    doc.close()


@pytest.fixture
def running_header_pdf(tmp_path: Path) -> Path:
    pdf = tmp_path / "running-header.pdf"
    _write_running_header_pdf(pdf, pages=PAGES)
    return pdf


def _strip_markdown_emphasis(text: str) -> str:
    """Drop the formatting the markdown generator injects mid-word, and
    collapse whitespace.

    The generator emphasises the header, so on this fixture it comes back as
    ``**ACME AWC-3200 Motor Controller**``; on the real PSoC 6 datasheet the
    same line arrives as ``**PSOC**<sup>**™**</sup> **62 MCU**``, split
    mid-string. Either way a literal substring test for the header passes on
    the *unfixed* code and proves nothing -- that false pass is exactly what
    the first version of this test did. Collapsing whitespace closes the same
    hole for a header the generator happens to wrap.

    Only the negative assertions gain sensitivity from this: none of the
    positive targets contain ``*``, ``_``, a backtick or angle brackets, so
    stripping cannot mask a genuine content loss.
    """
    return re.sub(r"\s+", " ", re.sub(r"[*_`]|<[^>]+>", "", text))


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


def test_extract_table_markdown_drops_the_running_header_and_footer(
    running_header_pdf, tmp_path
):
    """The page furniture must not reach the agent.

    `extract_table_markdown` already pays for a full layout pass, and that
    pass classifies page-header/page-footer blocks. Passing the two flags
    spends nothing extra and keeps the running header, the footer and the
    page number out of every extracted table. Measured on the PSoC 6
    datasheet: ~86 characters of pure furniture per page.

    The body assertions are what stop the fix from being "return less".
    Captions are not asserted here because they are safe by construction
    rather than by luck: `caption` is its own layout class, and
    `pymupdf4llm.helpers.document_layout` skips exactly the `page-header`
    and `page-footer` classes. A `Table N (continued)` caption -- which
    `TocNode.continued_tables` depends on -- is therefore never a candidate.
    Confirmed live on PSoC page 101, where `Table 43 (continued)` survives.

    The signature assertion is the tripwire: both `_layout_to_markdown` and
    the classic renderer swallow unknown keywords into `**kwargs`, so an
    upstream rename would silently restore the furniture. Asserting the
    parameters exist names that cause instead of leaving a maintainer to
    debug headers that came back.
    """
    import pymupdf4llm  # ty: ignore[unresolved-import]

    params = inspect.signature(pymupdf4llm._layout_to_markdown).parameters
    assert {"header", "footer"} <= set(params), (
        "pymupdf4llm._layout_to_markdown no longer takes header=/footer=; "
        f"the furniture suppression is silently inert. Parameters: {sorted(params)}"
    )

    session = create_datasheet_tool_session()
    defs = {d.name: d for d in session.defs}
    try:
        build = asyncio.run(
            defs["build_datasheet"].handler(
                {
                    "pdf_source": str(running_header_pdf),
                    "output_dir": str(tmp_path / "out"),
                }
            )
        )
        assert build["is_error"] is False, build
        result = asyncio.run(defs["extract_table_markdown"].handler({"page": 2}))
    finally:
        session.close()

    assert result["is_error"] is False, result
    flat = _strip_markdown_emphasis(result["content"][0]["text"])

    assert RUNNING_HEADER not in flat, f"running header survived:\n{flat}"
    assert RUNNING_FOOTER not in flat, f"running footer survived:\n{flat}"
    assert RUNNING_REVISION not in flat, f"revision footer survived:\n{flat}"

    # The table and the body text must still be there.
    assert "|" in flat, f"table markup lost:\n{flat}"
    assert "c110" in flat, f"table cell text lost:\n{flat}"
    assert BODY_SENTENCE in flat, f"body sentence lost:\n{flat}"
