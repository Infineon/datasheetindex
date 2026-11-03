"""Tests for the layout-hook guard in datasheetindex.core.engine."""

import contextlib
import threading
import types
from typing import Any

import pymupdf
import pytest

from datasheetindex.core import engine
from datasheetindex.core.engine import _LAYOUT_LOCK, classic_tables, layout_engine


def _hook(*args: Any, **kwargs: Any) -> list[Any]:
    """Stand-in for the ONNX layout analyzer. Only its identity matters here.

    A callable, not an ``object()``: ``pymupdf._get_layout`` is declared
    ``Callable[..., Any] | None``, so assigning a bare sentinel to it fails
    type checking.
    """
    return []


def _absent_hook(*args: Any, **kwargs: Any) -> Any:
    """Sentinel meaning "the attribute did not exist". Never invoked."""
    raise AssertionError("_ABSENT sentinel must never be called")


_HOOK = _hook
_ABSENT = _absent_hook
# PyMuPDF leaves _layout_enabled unannotated, so ty types it Literal[True].
_SWITCH: Any = pymupdf


@pytest.fixture(autouse=True)
def _restore_hook():
    """Every test here mutates a process-wide global. Put it back exactly.

    Uses a sentinel rather than a None default for the same reason
    classic_tables() does: a PyMuPDF with no _get_layout attribute must not
    acquire one on teardown, or this fixture would quietly defeat
    test_classic_tables_is_a_true_noop_without_the_attribute.
    """
    saved = getattr(pymupdf, "_get_layout", _ABSENT)
    yield
    if saved is _ABSENT:
        if hasattr(pymupdf, "_get_layout"):
            delattr(pymupdf, "_get_layout")
    else:
        pymupdf._get_layout = saved


@pytest.fixture(autouse=True)
def _restore_quad_corrections():
    """The other process-wide global engine.py owns. Put it back too."""
    saved = bool(pymupdf.TOOLS.unset_quad_corrections())
    yield
    pymupdf.TOOLS.unset_quad_corrections(saved)


@pytest.fixture(autouse=True)
def _restore_layout_enabled():
    """PyMuPDF 2.1's layout switch, which classic_tables() holds on."""
    saved = _SWITCH._layout_enabled
    yield
    _SWITCH._layout_enabled = saved


def _quad_corrections_unset() -> bool:
    return bool(pymupdf.TOOLS.unset_quad_corrections())


def _lock_is_held_by_another_thread() -> bool:
    """True when _LAYOUT_LOCK cannot be acquired from a fresh thread."""
    acquired: list[bool] = []

    def probe() -> None:
        got = _LAYOUT_LOCK.acquire(blocking=False)
        acquired.append(got)
        if got:
            _LAYOUT_LOCK.release()

    thread = threading.Thread(target=probe)
    thread.start()
    thread.join()
    return not acquired[0]


def test_classic_tables_suppresses_and_restores():
    pymupdf._get_layout = _HOOK
    with classic_tables():
        assert pymupdf._get_layout is None
    assert pymupdf._get_layout is _HOOK


def test_classic_tables_restores_on_exception():
    pymupdf._get_layout = _HOOK
    with contextlib.suppress(RuntimeError), classic_tables():
        raise RuntimeError("boom")
    assert pymupdf._get_layout is _HOOK


def test_classic_tables_nests():
    pymupdf._get_layout = _HOOK
    with classic_tables():
        with classic_tables():
            assert pymupdf._get_layout is None
        assert pymupdf._get_layout is None
    assert pymupdf._get_layout is _HOOK


def test_classic_tables_is_a_true_noop_without_the_attribute(monkeypatch):
    """A PyMuPDF with no hook must not gain one. Assigning None would."""
    monkeypatch.delattr(pymupdf, "_get_layout", raising=False)
    with classic_tables():
        ran = True
    assert ran
    assert not hasattr(pymupdf, "_get_layout")


def test_classic_tables_holds_the_lock():
    pymupdf._get_layout = _HOOK
    with classic_tables():
        assert _lock_is_held_by_another_thread()


def _fake_module(*, use_layout: bool, on_use_layout=None) -> Any:
    # Any, not ModuleType: these attributes are not in ModuleType's stub,
    # so ty rejects setting them on a types.ModuleType-typed name.
    module: Any = types.ModuleType("pymupdf4llm")
    module._use_layout = use_layout

    def _use_layout(yes: bool) -> None:
        module._use_layout = yes
        if on_use_layout is not None:
            on_use_layout(yes)

    module.use_layout = _use_layout
    return module


def _fake_engine(on_activate=None) -> Any:
    """Stand-in for ``pymupdf.layout``; ``activate()`` installs the hook."""
    layout: Any = types.ModuleType("pymupdf.layout")

    def activate() -> None:
        pymupdf._get_layout = _HOOK
        if on_activate is not None:
            on_activate()

    layout.activate = activate
    return layout


def _importer(monkeypatch, modules: dict[str, Any], on_import=None) -> list[str]:
    """Route engine.importlib to ``modules``; an exception value is raised.

    Returns the names imported, in order.
    """
    imported: list[str] = []

    def import_module(name: str) -> Any:
        imported.append(name)
        if on_import is not None:
            on_import(name)
        found = modules[name]
        if isinstance(found, BaseException):
            raise found
        return found

    monkeypatch.setattr(
        engine, "importlib", types.SimpleNamespace(import_module=import_module)
    )
    return imported


def test_layout_engine_yields_the_module_and_holds_the_lock(monkeypatch):
    module = _fake_module(use_layout=True)
    pymupdf._get_layout = _HOOK
    _importer(monkeypatch, {"pymupdf4llm": module, "pymupdf.layout": _fake_engine()})
    with layout_engine() as yielded:
        assert yielded is module
        assert _lock_is_held_by_another_thread()


def test_layout_engine_installs_the_hook_under_the_lock(monkeypatch):
    """The engine import is what installs the hook, so it happens in the lock.

    Since PyMuPDF 2.1 pymupdf4llm no longer imports the engine itself; left to
    the first get_layout() inside to_markdown, the import -- and the hook --
    would land outside the lock. One interleaving with classic_tables() then
    restores a stale None over the fresh hook.
    """
    held: list[bool] = []

    def on_import(name: str) -> None:
        held.append(_lock_is_held_by_another_thread())
        if name == "pymupdf.layout":
            pymupdf._get_layout = _HOOK  # activate() at import does this

    pymupdf._get_layout = None
    imported = _importer(
        monkeypatch,
        {
            "pymupdf4llm": _fake_module(use_layout=True),
            "pymupdf.layout": _fake_engine(),
        },
        on_import=on_import,
    )
    with layout_engine():
        assert pymupdf._get_layout is _HOOK
    assert imported == ["pymupdf4llm", "pymupdf.layout"]
    assert held == [True, True]


def test_layout_engine_repairs_a_missing_hook(monkeypatch):
    """An imported engine with no hook was clobbered by a stale restore."""
    activated: list[bool] = []
    pymupdf._get_layout = None
    _importer(
        monkeypatch,
        {
            "pymupdf4llm": _fake_module(use_layout=True),
            "pymupdf.layout": _fake_engine(on_activate=lambda: activated.append(True)),
        },
    )
    with layout_engine():
        assert pymupdf._get_layout is _HOOK
    assert activated == [True]


def test_layout_engine_leaves_the_engine_alone_when_layout_is_off(monkeypatch):
    """pymupdf4llm decided against layout: importing the engine would be waste,
    and a broken onnxruntime would fail where it never needed to load."""
    pymupdf._get_layout = None
    imported = _importer(monkeypatch, {"pymupdf4llm": _fake_module(use_layout=False)})
    with layout_engine():
        assert pymupdf._get_layout is None
    assert imported == ["pymupdf4llm"]


def test_layout_engine_degrades_when_the_engine_cannot_be_imported(monkeypatch, caplog):
    """A broken onnxruntime routes to_markdown to the classic renderer.

    That is what PyMuPDF 2.0 did on its own. Since 2.1 pymupdf4llm decides from
    find_spec alone, so without this the error would surface from inside
    to_markdown -- and layout_active() would let layout-only keywords through
    to the classic renderer, which prints them to stdout.
    """
    calls: list[bool] = []
    module = _fake_module(use_layout=True, on_use_layout=calls.append)
    _importer(
        monkeypatch,
        {
            "pymupdf4llm": module,
            "pymupdf.layout": ImportError("libonnxruntime.so: cannot open"),
        },
    )
    with caplog.at_level("WARNING"), layout_engine() as yielded:
        assert not engine.layout_active(yielded)
    assert calls == [False]
    assert "onnxruntime" in caplog.text


def test_layout_engine_raises_a_helpful_import_error(monkeypatch):
    _importer(monkeypatch, {"pymupdf4llm": ImportError("pymupdf4llm")})
    with pytest.raises(ImportError, match="Install pymupdf instead of pymupdf-lite"):
        with layout_engine():
            pass


def test_classic_tables_holds_layout_enabled_on_and_restores():
    """use_layout(False) anywhere in the process must not change the counts.

    PyMuPDF 2.1 adds TEXT_ACCURATE_BBOXES to the scan's text page only while
    _layout_enabled is on, so a sequential scan in a process that switched it
    off would read different character boxes than a fresh pool worker.
    """
    _SWITCH._layout_enabled = False
    with classic_tables():
        assert _SWITCH._layout_enabled is True
    assert _SWITCH._layout_enabled is False


def test_find_classic_tables_enters_the_guard_itself():
    """No caller can forget the guard: the read sees the guard's state."""
    seen: list[tuple[bool, bool]] = []

    class _Page:
        def find_tables(self, **kwargs: Any) -> Any:
            assert kwargs == {"use_layout": False}
            seen.append((_quad_corrections_unset(), _lock_is_held_by_another_thread()))
            return types.SimpleNamespace(tables=[])

    pymupdf.TOOLS.unset_quad_corrections(True)
    engine.find_classic_tables(_Page())
    assert seen == [(False, True)]
    assert _quad_corrections_unset()


def test_classic_tables_holds_quad_corrections_on_and_restores():
    """pymupdf4llm unsets quad corrections at import; the scan must not see it.

    With them unset, classic cell text comes back mangled ("1.71" -> "171\\n.")
    and the table counts may move -- tests/test_layout_integration.py pins that
    on a real datasheet.
    """
    pymupdf.TOOLS.unset_quad_corrections(True)
    with classic_tables():
        assert not _quad_corrections_unset()
    assert _quad_corrections_unset()


def test_classic_tables_restores_quad_corrections_on_error():
    pymupdf.TOOLS.unset_quad_corrections(True)
    with contextlib.suppress(RuntimeError), classic_tables():
        raise RuntimeError("boom")
    assert _quad_corrections_unset()


def test_layout_engine_keeps_quad_corrections_on_for_every_thread(monkeypatch):
    """The flag stays off even while pymupdf4llm runs, because it is global.

    Turning it on for the duration would let a build on another thread -- whose
    page reads take no lock -- read mangled text into its cached artifact.
    """
    _importer(
        monkeypatch,
        {
            "pymupdf4llm": _fake_module(use_layout=True),
            "pymupdf.layout": _fake_engine(),
        },
    )
    pymupdf.TOOLS.unset_quad_corrections(False)
    with layout_engine():
        assert not _quad_corrections_unset()
    assert not _quad_corrections_unset()


@pytest.mark.parametrize(
    "entry_point",
    [
        # A reader whose own imports never name engine.py: the reset reaches it
        # through datasheetindex.core/__init__.py, and only through that.
        "import datasheetindex.core.textfile",
        "from datasheetindex import DatasheetIndex",
        "from datasheetindex import DatasheetTools",
    ],
)
def test_every_page_reading_entry_point_resets_quad_corrections(entry_point):
    """By the time any page is read, the import-time reset has happened.

    A host that imported pymupdf4llm before us has turned the flag on, so a
    process that never touched layout_engine() would otherwise read every page
    with it on. Asserted in a fresh interpreter, since this one has been through every
    other test's toggling. The package root itself no longer implies the reset
    (it resolves its exports lazily, see datasheetindex/__init__.py), so this
    pins the entry points a host actually reads pages through.
    """
    import subprocess
    import sys

    code = (
        # The host imported pymupdf4llm first, which turns the flag on.
        f"import pymupdf, pymupdf4llm; {entry_point}; "
        "assert not pymupdf.TOOLS.unset_quad_corrections()"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr


def test_layout_engine_resets_a_flag_its_first_import_set(monkeypatch):
    """The reset follows the imports, because an import is what sets the flag.

    Since PyMuPDF 2.1 `import pymupdf` no longer imports pymupdf4llm, so the
    import inside layout_engine() is normally the first, and its helpers turn
    the flag on as it runs.
    """

    def on_import(name: str) -> None:
        pymupdf.TOOLS.unset_quad_corrections(True)

    _importer(
        monkeypatch,
        {
            "pymupdf4llm": _fake_module(use_layout=True),
            "pymupdf.layout": _fake_engine(),
        },
        on_import=on_import,
    )
    pymupdf.TOOLS.unset_quad_corrections(False)
    with layout_engine():
        assert not _quad_corrections_unset()
    assert not _quad_corrections_unset()


def test_layout_engine_resets_the_flag_when_the_import_fails(monkeypatch):
    def half_import(name: str) -> Any:
        pymupdf.TOOLS.unset_quad_corrections(True)
        raise ImportError(name)

    monkeypatch.setattr(
        engine, "importlib", types.SimpleNamespace(import_module=half_import)
    )
    pymupdf.TOOLS.unset_quad_corrections(False)
    with pytest.raises(ImportError), layout_engine():
        pass
    assert not _quad_corrections_unset()
