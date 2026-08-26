"""Packaging: the library resolves from the clone, and the CLI is registered."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

BENCHMARK = Path(__file__).resolve().parents[1]
PYPROJECT = BENCHMARK / "pyproject.toml"

#: Both of these are declared by the `harness` extra, so on a Tier-1 install
#: (`.[test]` alone) there is nothing to check -- the path source and the
#: console-script declaration are still asserted, from the TOML, above.
_needs_harness = pytest.mark.skipif(
    importlib.util.find_spec("datasheetindex") is None
    or importlib.util.find_spec("anthropic") is None,
    reason="needs the harness extra: uv pip install -e '.[harness]'",
)


def _cfg() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def test_datasheetindex_resolves_from_the_parent_repo():
    """Not a PyPI requirement -- a path source pointing at the clone."""
    src = _cfg()["tool"]["uv"]["sources"]["datasheetindex"]
    assert "path" in src, src
    assert Path(BENCHMARK / src["path"]).resolve() == BENCHMARK.parent


def test_datasheetindex_carries_no_version_floor():
    """The clone supplies the version; a floor here is meaningless and can
    only break a reader whose checkout predates it."""
    harness = " ".join(_cfg()["project"]["optional-dependencies"]["harness"])
    assert "datasheetindex>" not in harness
    assert "datasheetindex=" not in harness


@_needs_harness
def test_the_library_actually_imports():
    """The point of the path source. Fails if resolution is misconfigured."""
    import datasheetindex  # noqa: F401


def test_chamber_run_console_script_is_registered():
    assert (
        _cfg()["project"]["scripts"]["chamber-run"] == "chamberbench.harness.run:main"
    )


@_needs_harness
def test_chamber_run_is_invocable():
    result = subprocess.run(
        [sys.executable, "-m", "chamberbench.harness.run", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=BENCHMARK,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--model" in result.stdout


def test_archive_and_figures_stay_out_of_the_sdist():
    excl = _cfg()["tool"]["hatch"]["build"]["targets"]["sdist"]["exclude"]
    assert "archive/" in excl
    assert "figures/" in excl
