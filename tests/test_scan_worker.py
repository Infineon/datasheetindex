"""Tests for datasheetindex._scan_worker: scan processes skip the ONNX engine.

Every assertion about import state runs in a fresh interpreter, since this one
imported PyMuPDF -- and with it pymupdf4llm and the ONNX engine -- long ago.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pymupdf

from datasheetindex._scan_worker import skip_layout_engine

PROBE = Path(__file__).parent / "_scan_worker_probe.py"
_ENGINE_MODULES = ("pymupdf.layout", "pymupdf4llm", "onnxruntime")


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )


def test_package_import_does_not_import_pymupdf():
    """Every worker imports this package first, to resolve its initializer.

    An eager re-export in datasheetindex/__init__.py would import PyMuPDF --
    and the ONNX engine -- before the initializer could block it.
    """
    proc = _run(
        "import sys, datasheetindex, datasheetindex._scan_worker; "
        "assert 'pymupdf' not in sys.modules, sorted(sys.modules)"
    )
    assert proc.returncode == 0, proc.stderr


def test_skip_layout_engine_keeps_the_engine_out():
    """PyMuPDF then comes up as pymupdf-lite does: no engine, hook or flag.

    And says nothing on stdout. With layout looking uninstalled, find_tables()
    would otherwise print PyMuPDF's layout advertisement there.
    """
    code = f"""
import sys
from datasheetindex._scan_worker import skip_layout_engine
skip_layout_engine()
import pymupdf
loaded = [m for m in {_ENGINE_MODULES!r} if sys.modules.get(m)]
assert not loaded, loaded
assert pymupdf._get_layout is None
assert not pymupdf.TOOLS.unset_quad_corrections()
doc = pymupdf.open()
doc.new_page().find_tables()
"""
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""


def test_skip_layout_engine_is_a_noop_once_pymupdf_is_imported(monkeypatch):
    """A worker whose host __main__ imported PyMuPDF already has the engine.

    Blocking then would only make a later import of it fail; the scan's
    classic_tables() guard covers that worker instead.
    """
    # setenv first so the delenv records an undo: a regression that sets the
    # variable here must not leak it into later tests, where it would silence
    # the layout notice test_mcp_server's stdio-wire test depends on.
    monkeypatch.setenv("PYMUPDF_SUGGEST_LAYOUT_ANALYZER", "unset-by-test")
    monkeypatch.delenv("PYMUPDF_SUGGEST_LAYOUT_ANALYZER")
    before = {name: sys.modules.get(name) for name in _ENGINE_MODULES}

    skip_layout_engine()

    assert {name: sys.modules.get(name) for name in _ENGINE_MODULES} == before
    assert "PYMUPDF_SUGGEST_LAYOUT_ANALYZER" not in os.environ


def test_pool_workers_skip_the_engine():
    """The production pool, real start method, real initializer.

    The expected value names ``pymupdf`` and is compared exactly, so a worker
    that skipped the engine by never importing PyMuPDF cannot pass this
    vacuously. Keep it an equality check.
    """
    proc = subprocess.run(
        [sys.executable, str(PROBE)], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == {"0": ["pymupdf"], "1": ["pymupdf"]}


def test_windows_helper_skips_the_engine_itself(tmp_path):
    """The helper only scans, so its own process skips the engine too.

    Its pool workers are covered by test_pool_workers_skip_the_engine; this is
    the process that builds that pool, launched as `-m`.
    """
    pdf = tmp_path / "one.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(str(pdf))
    doc.close()
    out = tmp_path / "counts.json"

    code = f"""
import sys
from datasheetindex._scan_worker import main
assert main([{str(pdf)!r}, "1", {str(out)!r}]) == 0
loaded = [m for m in {_ENGINE_MODULES!r} if sys.modules.get(m)]
assert not loaded, loaded
"""
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(out.read_text(encoding="utf-8")) == {"0": 0}
