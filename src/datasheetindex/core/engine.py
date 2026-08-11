"""Ownership of PyMuPDF's process-wide table-engine hook.

Importing ``pymupdf4llm`` installs an ONNX-backed callable at
``pymupdf._get_layout``. ``Page.find_tables()`` consults that global on every
call, so which table engine runs is a property of the *process*, not of the
document. This module is the only place in the package that imports
``pymupdf4llm``, and the only place that reads or writes the hook.

Both context managers below serialize on one re-entrant lock:

* :func:`classic_tables` suppresses the hook, so ``find_tables()`` uses
  PyMuPDF's classic geometric detector.
* :func:`layout_engine` imports ``pymupdf4llm`` and yields it with its hook
  installed.

The import must happen inside the lock because the import *is* what installs
the hook. An import racing :func:`classic_tables` lets the guard restore a
stale ``None`` while ``pymupdf4llm._use_layout`` stays ``True``; ``to_markdown``
then routes into ``_layout_to_markdown``, iterates a ``None``
``page.layout_information``, and raises ``TypeError``. The module is cached, so
that breakage is permanent for the process.

Nesting :func:`layout_engine` inside :func:`classic_tables` on one thread is
silently wrong rather than loud: the lock is re-entrant so it will not
deadlock, and the invariant check below reinstalls the hook. No such path
exists in this package.
"""

from __future__ import annotations

import importlib
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pymupdf

_LAYOUT_LOCK = threading.RLock()


def _missing_hook(*args: Any, **kwargs: Any) -> Any:
    """Sentinel distinguishing "attribute absent" from "attribute is None".

    Callable, not a bare ``object()``: ``pymupdf._get_layout`` is declared
    ``Callable[..., Any] | None``, so restoring an ``object()`` sentinel back
    onto it does not type-check. Only this sentinel's *identity* is ever used;
    it must never be invoked.
    """
    raise AssertionError("_MISSING sentinel must never be called")


_MISSING = _missing_hook


@contextmanager
def classic_tables() -> Iterator[None]:
    """Pin ``find_tables()`` to PyMuPDF's classic detector for the duration."""
    with _LAYOUT_LOCK:
        saved = getattr(pymupdf, "_get_layout", _MISSING)
        if saved is _MISSING:
            # No hook in this PyMuPDF: find_tables() is already classic, and
            # assigning None here would leave an attribute behind.
            yield
            return
        pymupdf._get_layout = None
        try:
            yield
        finally:
            pymupdf._get_layout = saved


@contextmanager
def layout_engine() -> Iterator[Any]:
    """Import ``pymupdf4llm`` under the lock and yield it, hook installed.

    Raises ``ImportError`` when the optional ``[layout]`` extra is absent.
    """
    with _LAYOUT_LOCK:
        try:
            module = importlib.import_module("pymupdf4llm")
        except ImportError:
            raise ImportError(
                "pymupdf4llm is required for table markdown extraction. "
                "Install it with: uv sync --extra layout"
            ) from None
        # Invariant: if pymupdf4llm believes layout is on, the hook must exist.
        # Gated on _use_layout so a pymupdf4llm whose `import pymupdf.layout`
        # failed legitimately is not "repaired" into an engine it lacks.
        if (
            getattr(module, "_use_layout", False)
            and getattr(pymupdf, "_get_layout", None) is None
        ):
            module.use_layout(True)
        yield module


def layout_active(module: Any) -> bool:
    """Whether ``module.to_markdown`` will take pymupdf4llm's layout branch.

    ``layout_engine()`` yielding successfully does **not** imply the layout
    path. ``to_markdown`` dispatches on the module global ``_use_layout`` at
    *call* time, and that global is ``False`` whenever pymupdf4llm's own
    ``import pymupdf.layout`` failed -- a broken or unimportable
    ``onnxruntime`` is enough, and the package still imports fine. In that
    state ``to_markdown`` silently routes to the classic renderer in
    ``helpers/pymupdf_rag.py``, which accepts ``**kwargs`` and ``print()``s
    "Warning - arguments ignored in legacy mode: ..." to **stdout** for any it
    does not know -- the channel the MCP stdio transport carries JSON-RPC on.
    So callers must ask before passing a layout-only keyword.

    Upstream draws the same line: ``to_markdown``'s ``table_output="html"``
    branch pops ``header`` and ``footer`` out of the kwargs before it calls
    the classic function.
    """
    return bool(getattr(module, "_use_layout", False))
