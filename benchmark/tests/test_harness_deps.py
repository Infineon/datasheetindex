"""The harness extra declares what the harness modules actually import."""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
TESTS = Path(__file__).resolve().parent

#: Modules a test may `importorskip`, and where each comes from. A skip is only
#: legitimate for something the harness extra installs; skipping on a name
#: nothing ever provides would silently gut the run while it stayed green.
_SKIPPABLE = {
    "anthropic": "harness extra",
    "openai": "harness extra",
    "requests": "harness extra",
    "pymupdf": "harness extra, transitively via datasheetindex",
}


def _harness_extra() -> str:
    cfg = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return " ".join(cfg["project"]["optional-dependencies"]["harness"])


def test_declares_the_third_party_imports_the_harness_uses():
    names = _harness_extra()
    for pkg in (
        "anthropic",
        "openai",
        "httpx",
        "httpx2",
        "requests",
        "tenacity",
        "python-dotenv",
    ):
        assert pkg in names, pkg


def test_does_not_declare_datasheetindex_as_a_pypi_requirement():
    """datasheetindex is not on PyPI -- it IS this repository. Declaring it
    WITH A VERSION SPECIFIER would send the resolver to PyPI and break
    `uv sync --extra harness` for every reader. A bare, unversioned entry is
    fine and expected: it is what gives `[tool.uv.sources]`'s path override
    (`benchmark/pyproject.toml`) something to apply to, so the name itself is
    allowed to appear here -- only a version pin is forbidden.
    """
    extra = _harness_extra()
    assert "datasheetindex>" not in extra
    assert "datasheetindex=" not in extra


def test_base_install_stays_offline():
    """Tier 1 must remain installable with no API client at all."""
    cfg = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    deps = " ".join(cfg["project"]["dependencies"])
    for pkg in ("anthropic", "openai", "tenacity"):
        assert pkg not in deps, pkg


def _importorskip_names() -> dict[str, set[str]]:
    """Every `pytest.importorskip("X")` in tests/, by test module."""
    found: dict[str, set[str]] = {}
    for path in sorted(TESTS.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", None)
            )
            if name != "importorskip" or not node.args:
                continue
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                found.setdefault(path.name, set()).add(arg.value)
    return found


def test_every_importorskip_names_something_the_harness_extra_installs():
    """A skip guard must be able to stop skipping.

    The five harness test modules are `importorskip`-guarded so that a Tier-1
    install degrades to skips instead of five collection errors. That is only
    safe while each guarded name is something an install can actually supply:
    a typo, or a name no extra provides, turns the guard into a permanent skip
    that keeps the suite green while testing nothing.
    """
    for module, names in _importorskip_names().items():
        for name in names:
            assert name in _SKIPPABLE, (
                f"{module}: importorskip({name!r}) names nothing the harness "
                "extra installs -- it would skip forever. Add it to _SKIPPABLE "
                "with its source, or guard on a name that is really installed."
            )


def test_guarded_modules_import_once_the_extra_is_installed():
    """With the harness extra present, every guarded module must really import.

    The guard above proves each skipped name is installable; this proves the
    guards actually open. Without it, a guarded module could keep skipping for
    a reason unrelated to the extra -- a broken import further down the file --
    and the suite would stay green while five modules tested nothing. Modules
    whose guard names are genuinely absent (a Tier-1 install) are passed over,
    which is the case this whole mechanism exists to support.
    """
    checked = 0
    for module, names in _importorskip_names().items():
        if any(importlib.util.find_spec(n) is None for n in names):
            continue  # Tier-1 install: legitimately skipped.
        spec = importlib.util.spec_from_file_location(
            f"_guardcheck_{module[:-3]}", TESTS / module
        )
        assert spec is not None and spec.loader is not None, module
        loaded = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(loaded)
        except Exception as exc:  # pragma: no cover -- the failure this catches
            raise AssertionError(
                f"{module} is importorskip-guarded on {sorted(names)}, all of "
                f"which are installed, yet importing it failed: {exc!r}. The "
                "guard is hiding a real breakage."
            ) from exc
        checked += 1
    if not checked:
        pytest.skip("Tier-1 install: no guarded module has all its names present")


# ---------------------------------------------------------------------------
# The install hint must be reserved for an actual missing install
# ---------------------------------------------------------------------------

SRC = Path(__file__).resolve().parents[1] / "src"

#: Imports ``chamberbench.harness.run`` with the first import inside its
#: harness-extra ``try`` block forced to fail. ``sys.meta_path`` is used rather
#: than uninstalling anything, so the two cases differ in exactly one thing:
#: the ``name`` carried by the ImportError -- which is the only signal that
#: separates "the extra is not installed" from "the extra is installed and
#: something inside it is broken".
_SHIM = """
import sys
import importlib.abc

TARGET = "chamberbench.harness.anthropic_path"


class Raiser(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        if sys.argv[1] == "missing":
            # What a Tier-1 install really produces: the nested `import
            # anthropic` fails and the outer import statement sees it.
            raise ModuleNotFoundError("No module named 'anthropic'", name="anthropic")
        # A renamed symbol, a circular import, a half-broken install: the
        # module that failed is one of OURS.
        raise ImportError(
            "cannot import name 'extract_chamber_agentic' from " + TARGET,
            name=TARGET,
        )


sys.meta_path.insert(0, Raiser())
import chamberbench.harness.run  # noqa: F401

print("IMPORTED WITHOUT ERROR")
"""


def _harness_extra_modules() -> set[str]:
    """``run._HARNESS_EXTRA_MODULES``, read from the source, not imported."""
    tree = ast.parse((SRC / "chamberbench" / "harness" / "run.py").read_text("utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_HARNESS_EXTRA_MODULES"
            for t in node.targets
        ):
            call = node.value
            assert isinstance(call, ast.Call), "expected frozenset({...})"
            return set(ast.literal_eval(call.args[0]))
    raise AssertionError("run.py no longer defines _HARNESS_EXTRA_MODULES")


def _import_run_with(case: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    shim = tmp_path / "shim.py"
    shim.write_text(_SHIM, encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=str(SRC))
    return subprocess.run(
        [sys.executable, str(shim), case],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=env,
    )


def test_a_missing_harness_extra_gets_the_install_hint(tmp_path):
    """The Tier-1 case, unchanged: exit 2, the hint, and no traceback."""
    result = _import_run_with("missing", tmp_path)
    assert result.returncode == 2, result
    assert "needs the harness extra, which is not installed" in result.stderr
    assert "uv pip install -e '.[harness]'" in result.stderr
    assert "Traceback" not in result.stderr, result.stderr


def test_a_broken_harness_module_is_not_reported_as_a_missing_install(tmp_path):
    """An ImportError from inside our own modules must NOT get the hint.

    The guard caught every ImportError and answered all of them with "the
    harness extra is not installed", traceback suppressed. So a renamed symbol
    or a circular import in ``anthropic_path`` / ``datasheet_tools`` reached
    the reader as an install instruction, on a machine where the extra *was*
    installed -- reinstalling changes nothing, and the one piece of
    information that would have identified the fault had been thrown away.

    The two cases here differ only in ``ImportError.name``, which is exactly
    the distinction ``_is_missing_harness_extra`` draws.
    """
    result = _import_run_with("broken", tmp_path)
    assert result.returncode != 0, result
    assert "needs the harness extra" not in result.stderr, result.stderr
    assert "Traceback" in result.stderr, result.stderr
    assert "cannot import name 'extract_chamber_agentic'" in result.stderr
    assert "IMPORTED WITHOUT ERROR" not in result.stdout


def test_the_harness_extra_module_names_cover_what_the_extra_installs():
    """Every distribution in the ``harness`` extra maps to a listed module.

    ``_HARNESS_EXTRA_MODULES`` is the allow-list that decides whether an
    ImportError earns the install hint. A dependency added to the extra but
    not added there is the opposite mistake -- a genuinely missing install
    re-raised as a traceback -- and just as confusing.

    Read out of the source rather than imported: importing ``run`` on a Tier-1
    install is exactly what raises ``SystemExit(2)``, so an ``import`` here
    would make this test's own subject un-runnable in the tier it describes.
    """
    modules = _harness_extra_modules()

    #: Distribution name -> the module it imports as, where the two differ.
    aliases = {"python-dotenv": "dotenv"}
    for requirement in _harness_extra().split():
        dist = re.split(r"[><=!~\[,]", requirement)[0].strip()
        if not dist:
            continue
        module = aliases.get(dist, dist)
        assert module in modules, (
            f"{dist!r} is in the harness extra but {module!r} is not in "
            "_HARNESS_EXTRA_MODULES; a machine missing it would get a "
            "traceback instead of the install hint"
        )
