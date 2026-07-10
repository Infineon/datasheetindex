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
    # Annotated Any: a bare ModuleType has no _use_layout/use_layout attributes,
    # so ty rejects setting them on a types.ModuleType-typed name.
    module: Any = types.ModuleType("pymupdf4llm")
    module._use_layout = use_layout
    module.use_layout = on_use_layout or (lambda yes: None)
    return module


def test_layout_engine_yields_the_module_and_holds_the_lock(monkeypatch):
    module = _fake_module(use_layout=True)
    pymupdf._get_layout = _HOOK
    monkeypatch.setattr(
        engine, "importlib", types.SimpleNamespace(import_module=lambda name: module)
    )
    with layout_engine() as yielded:
        assert yielded is module
        assert _lock_is_held_by_another_thread()


def test_layout_engine_installs_the_hook_under_the_lock(monkeypatch):
    """The import is what installs the hook, so it must happen inside the lock.

    An import outside the lock can interleave with classic_tables(), which then
    restores a stale None while pymupdf4llm._use_layout stays True -- and every
    later to_markdown() raises TypeError for the life of the process.
    """
    held: list[bool] = []

    def import_module(name: str) -> types.ModuleType:
        assert name == "pymupdf4llm"
        held.append(_lock_is_held_by_another_thread())
        pymupdf._get_layout = _HOOK  # the real import does exactly this
        return _fake_module(use_layout=True)

    pymupdf._get_layout = None
    monkeypatch.setattr(
        engine, "importlib", types.SimpleNamespace(import_module=import_module)
    )
    with layout_engine():
        assert pymupdf._get_layout is _HOOK
    assert held == [True]


def test_layout_engine_repairs_a_missing_hook(monkeypatch):
    calls: list[bool] = []

    def on_use_layout(yes: bool) -> None:
        calls.append(yes)
        pymupdf._get_layout = _HOOK

    module = _fake_module(use_layout=True, on_use_layout=on_use_layout)
    pymupdf._get_layout = None
    monkeypatch.setattr(
        engine, "importlib", types.SimpleNamespace(import_module=lambda name: module)
    )
    with layout_engine():
        assert pymupdf._get_layout is _HOOK
    assert calls == [True]


def test_layout_engine_does_not_repair_when_layout_is_off(monkeypatch):
    """A pymupdf4llm whose `import pymupdf.layout` failed has no engine to
    reinstall. Repairing it would call use_layout(True) into an ImportError."""
    calls: list[bool] = []
    module = _fake_module(use_layout=False, on_use_layout=calls.append)
    pymupdf._get_layout = None
    monkeypatch.setattr(
        engine, "importlib", types.SimpleNamespace(import_module=lambda name: module)
    )
    with layout_engine():
        assert pymupdf._get_layout is None
    assert calls == []


def test_layout_engine_raises_a_helpful_import_error(monkeypatch):
    def boom(name: str):
        raise ImportError(name)

    monkeypatch.setattr(engine, "importlib", types.SimpleNamespace(import_module=boom))
    with pytest.raises(ImportError, match="uv sync --extra layout"):
        with layout_engine():
            pass
