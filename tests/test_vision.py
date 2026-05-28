"""Tests for inspect_page tool."""

import base64
from pathlib import Path
from typing import Any, cast

import pymupdf
import pytest

from datasheetindex.tools.vision import inspect_page

DATA2PAGE_DIR = Path(__file__).resolve().parent.parent.parent / "data2page"
TLE9350_PATH = DATA2PAGE_DIR / "Infineon-TLE9350BSJ-DataSheet-v01_00-EN.pdf"


def _make_test_doc() -> pymupdf.Document:
    """Create a simple test PDF with text on one page."""
    doc = pymupdf.open()
    page = doc.new_page()
    writer = pymupdf.TextWriter(page.rect)
    writer.append((72, 72), "Test content for vision")
    writer.write_text(page)
    return doc


def test_full_page_render():
    doc = _make_test_doc()
    result = inspect_page(doc, page=1)
    doc.close()

    assert len(result) == 1
    assert result[0]["type"] == "image"
    assert result[0]["mime_type"] == "image/png"
    assert len(result[0]["data"]) > 0


def test_base64_decodes_to_png():
    doc = _make_test_doc()
    result = inspect_page(doc, page=1)
    doc.close()

    png_bytes = base64.b64decode(result[0]["data"])
    # PNG magic bytes
    assert png_bytes[:4] == b"\x89PNG"


def test_region_crop_smaller():
    """Cropped region should produce a smaller image than full page."""
    doc = _make_test_doc()
    full = inspect_page(doc, page=1)
    cropped = inspect_page(
        doc, page=1, region={"top": 0.0, "bottom": 0.5, "left": 0.0, "right": 0.5}
    )
    doc.close()

    full_size = len(base64.b64decode(full[0]["data"]))
    crop_size = len(base64.b64decode(cropped[0]["data"]))
    assert crop_size < full_size


def test_page_zero_raises():
    doc = _make_test_doc()
    with pytest.raises(ValueError, match="out of range"):
        inspect_page(doc, page=0)
    doc.close()


def test_page_too_high_raises():
    doc = _make_test_doc()
    with pytest.raises(ValueError, match="out of range"):
        inspect_page(doc, page=999)
    doc.close()


def test_region_unknown_key_raises():
    doc = _make_test_doc()
    with pytest.raises(ValueError, match="Unknown region keys"):
        inspect_page(doc, page=1, region={"Top": 0.1})
    doc.close()


def test_region_out_of_bounds_raises():
    doc = _make_test_doc()
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        inspect_page(doc, page=1, region={"top": -0.1})
    doc.close()


def test_region_inverted_bounds_raises():
    doc = _make_test_doc()
    with pytest.raises(ValueError, match="top < bottom"):
        inspect_page(doc, page=1, region={"top": 0.8, "bottom": 0.2})
    doc.close()


def test_detail_low_smaller_than_high():
    """Detail tiers translate to dpi; lower detail must produce a smaller image."""
    doc = _make_test_doc()
    low = inspect_page(doc, page=1, detail="low")
    medium = inspect_page(doc, page=1, detail="medium")
    high = inspect_page(doc, page=1, detail="high")
    doc.close()

    low_bytes = len(base64.b64decode(low[0]["data"]))
    med_bytes = len(base64.b64decode(medium[0]["data"]))
    high_bytes = len(base64.b64decode(high[0]["data"]))
    assert low_bytes < med_bytes < high_bytes


def test_detail_high_matches_legacy_dpi_150():
    """detail='high' must produce identical pixel output to the pre-detail
    library default (dpi=150) so existing callers see no regression."""
    doc = _make_test_doc()
    via_detail = inspect_page(doc, page=1, detail="high")
    via_dpi = inspect_page(doc, page=1, dpi=150)
    doc.close()

    # Same render parameters → identical bytes (PyMuPDF is deterministic
    # given the same dpi and clip).
    assert via_detail[0]["data"] == via_dpi[0]["data"]


def test_explicit_dpi_overrides_detail():
    """When both are passed, dpi wins (power-user escape hatch)."""
    doc = _make_test_doc()
    via_detail_low = inspect_page(doc, page=1, detail="low")
    via_detail_low_with_dpi_150 = inspect_page(doc, page=1, detail="low", dpi=150)
    via_dpi_150 = inspect_page(doc, page=1, dpi=150)
    doc.close()

    assert via_detail_low_with_dpi_150[0]["data"] == via_dpi_150[0]["data"]
    assert via_detail_low[0]["data"] != via_detail_low_with_dpi_150[0]["data"]


def test_unknown_detail_raises():
    doc = _make_test_doc()
    with pytest.raises(ValueError, match="detail must be one of"):
        # ``cast`` so the type checker doesn't reject the deliberately
        # invalid literal we're using to exercise runtime validation.
        inspect_page(doc, page=1, detail=cast(Any, "ultra"))
    doc.close()


def test_unknown_detail_raises_even_with_explicit_dpi():
    """Validation must fire even when ``dpi`` is supplied -- otherwise a
    typo'd ``detail`` is silently ignored until the dpi override is
    removed, at which point the call starts raising in a confusing place."""
    doc = _make_test_doc()
    with pytest.raises(ValueError, match="detail must be one of"):
        inspect_page(doc, page=1, detail=cast(Any, "ultra"), dpi=150)
    doc.close()


@pytest.mark.real_pdf
def test_real_pdf_render():
    if not TLE9350_PATH.exists():
        pytest.skip("Test PDF not found")
    doc = pymupdf.open(str(TLE9350_PATH))
    result = inspect_page(doc, page=1)
    doc.close()

    png_bytes = base64.b64decode(result[0]["data"])
    assert png_bytes[:4] == b"\x89PNG"
    assert len(png_bytes) > 1000  # Non-trivial image
