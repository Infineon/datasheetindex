"""Tests for the table-scan processes: none of them loads the ONNX engine.

Since PyMuPDF 2.1 that needs no blocking: ``import pymupdf`` defers the engine
to first use, and the scan reaches ``find_tables()`` only through
``engine.find_classic_tables``, which never asks for it. These tests are what
keep both halves true. Every assertion about import state runs in a fresh
interpreter, since this one went through ``layout_engine()`` long ago.
"""

import json
import subprocess
import sys
from pathlib import Path

import pymupdf

PROBE = Path(__file__).parent / "_scan_worker_probe.py"
_ENGINE_MODULES = ("pymupdf.layout", "pymupdf4llm", "onnxruntime")


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )


def test_package_import_does_not_import_pymupdf():
    """Every worker imports this package first, to resolve its initializer.

    An eager re-export in datasheetindex/__init__.py would import PyMuPDF and
    datasheetindex.core in every worker and in the Windows helper just to find
    init_pool_worker, which needs neither.
    """
    proc = _run(
        "import sys, datasheetindex, datasheetindex._scan_worker; "
        "assert 'pymupdf' not in sys.modules, sorted(sys.modules)"
    )
    assert proc.returncode == 0, proc.stderr


def test_classic_scan_keeps_the_engine_out():
    """A classic scan neither imports the engine nor prints anything.

    ``find_tables()``'s default ``use_layout=True`` would import
    ``pymupdf.layout`` on the first call; stdout must stay clean because a
    sequential scan runs inside the MCP stdio server.
    """
    code = f"""
import sys
import pymupdf
from datasheetindex.core.engine import find_classic_tables
doc = pymupdf.open()
find_classic_tables(doc.new_page())
loaded = [m for m in {_ENGINE_MODULES!r} if sys.modules.get(m)]
assert not loaded, loaded
assert pymupdf._get_layout is None
assert not pymupdf.TOOLS.unset_quad_corrections()
"""
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""


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
    """The helper only scans, so its own process never loads the engine.

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
