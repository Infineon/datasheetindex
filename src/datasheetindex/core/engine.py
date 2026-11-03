"""Ownership of the process-wide PyMuPDF state that ``pymupdf4llm`` changes.

PyMuPDF 2.x bundles ``pymupdf4llm`` and the ONNX layout engine. Since 2.1 it
imports them on first use rather than with ``import pymupdf``, and that first
use can be inside ``find_tables()`` (see :func:`find_classic_tables`). The
import changes two module globals, and both alter what *every* later PyMuPDF
call in the process returns:

* ``pymupdf._get_layout`` -- an ONNX-backed callable that
  ``Page.find_tables()`` consults on every call, switching it to the ML engine.
* ``pymupdf.TOOLS.unset_quad_corrections`` -- set to ``True`` by three
  ``pymupdf4llm.helpers`` modules at import. With it on, character boxes skip
  PyMuPDF's quad corrections, which mangles classic table cell text
  (``"1.71"`` -> ``"171\\n."``, ``"Supply Voltage"`` -> ``"SupplyVoltage"``)
  and shifts ``get_text("blocks")`` coordinates. Across a 30-document corpus
  it changed the cells on 2,058 of 3,961 pages and block coordinates on 484.
  pymupdf4llm sets it for itself, but its layout-branch markdown barely
  depends on it (see :func:`layout_engine`), so this package keeps it off.

So which table engine runs, and what text comes back, are properties of the
*process*, not of the document. This module is the only place in the package
that imports ``pymupdf4llm`` or writes either global. It resets the flag to
PyMuPDF's own default at import -- a host may have imported pymupdf4llm
before us -- and never turns it back on, and:

* :func:`find_classic_tables` is the scan's ``find_tables()``. It enters
  :func:`classic_tables`, which suppresses the hook, holds the flag off and
  holds ``pymupdf._layout_enabled`` on, and passes ``use_layout=False`` so the
  call cannot re-import the engine: the scan uses PyMuPDF's classic geometric
  detector on corrected character boxes.
* :func:`layout_engine` imports ``pymupdf4llm`` and the engine and yields
  the former with the hook installed, and resets the flag again after
  importing, since a first import sets it.

Both serialize on one re-entrant lock. The import inside
:func:`layout_engine` must stay under it: the import *is* what installs the
hook, and one racing :func:`classic_tables` lets the guard restore a stale
``None`` while ``pymupdf4llm._use_layout`` stays ``True``.
``to_markdown`` then routes into ``_layout_to_markdown``, iterates a ``None``
``page.layout_information``, and raises ``TypeError`` -- permanently, since the
module is cached.

Nesting :func:`layout_engine` inside :func:`classic_tables` on one thread is
silently wrong rather than loud: the lock is re-entrant so it will not
deadlock, and the invariant check below reinstalls the hook. No such path
exists in this package.
"""

from __future__ import annotations

import importlib
import logging
import threading
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import pymupdf

logger = logging.getLogger(__name__)

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


def _set_quad_corrections_unset(on: bool) -> bool:
    """Set ``TOOLS.unset_quad_corrections`` and return its previous value."""
    previous = bool(pymupdf.TOOLS.unset_quad_corrections())
    pymupdf.TOOLS.unset_quad_corrections(on)
    return previous


# PyMuPDF's default, which pymupdf4llm overrode while `import pymupdf` ran.
# Every PyMuPDF call this package makes outside layout_engine() -- page text,
# search, rendering, the table scan -- expects it. This is deliberately a
# change to the *host's* process state too: a host that calls pymupdf4llm
# itself, outside layout_engine(), now does so with PyMuPDF's default rather
# than the flag pymupdf4llm set. Either value is wrong for someone in a shared
# process; this one is right for every PyMuPDF call that is not pymupdf4llm's.
_set_quad_corrections_unset(False)


@contextmanager
def classic_tables() -> Generator[None]:
    """Pin ``find_tables()`` to PyMuPDF's classic detector for the duration.

    Also holds the quad-corrections flag off, though the import-time reset
    already did: the host process may have set it since, and the table counts
    must not depend on that.

    And holds ``pymupdf._layout_enabled`` on. PyMuPDF 2.1 builds the scan's
    text page with ``TEXT_ACCURATE_BBOXES`` only while it is on, and
    ``pymupdf4llm.use_layout(False)`` -- a host's, or :func:`layout_engine`
    degrading -- turns it off process-wide. A sequential scan in that process
    would then count with different character boxes than a fresh pool worker.
    With ``use_layout=False`` (see :func:`find_classic_tables`) the switch
    affects nothing else in ``find_tables()``.
    """
    with _LAYOUT_LOCK:
        saved_quads = _set_quad_corrections_unset(False)
        # Any: PyMuPDF leaves the switch unannotated, so a type checker reads
        # its initial `True` as Literal[True] and rejects restoring a bool.
        switch: Any = pymupdf
        saved_enabled = bool(switch._layout_enabled)
        switch._layout_enabled = True
        try:
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
        finally:
            pymupdf.TOOLS.unset_quad_corrections(saved_quads)
            switch._layout_enabled = saved_enabled


def find_classic_tables(page: Any) -> Any:
    """``page.find_tables()`` on the classic detector, under :func:`classic_tables`.

    Enters the guard itself (the lock is re-entrant), so no caller can read
    with whatever process state the host left behind.

    The guard alone is not enough from PyMuPDF 2.1 on. There the default
    ``use_layout=True`` branch of ``find_tables()`` does two things no guard
    can undo from outside the call:

    * it calls ``page.get_layout()``, which imports ``pymupdf.layout`` on first
      use; the import's ``activate()`` installs the hook the guard has just
      cleared, so every page scanned under that guard ran the ML engine;
    * it turns quad corrections off for the rest of the call, so the classic
      detector read uncorrected character boxes.

    On a sequential build of the TI ADS1115 that took the count from 392
    tables to 99, and the Bosch BME280 build segfaulted.
    ``use_layout=False`` skips that branch, and with it the engine: a scan
    process that only calls this never imports ``pymupdf.layout``.
    """
    with classic_tables():
        return page.find_tables(use_layout=False)


@contextmanager
def layout_engine() -> Generator[Any]:
    """Import ``pymupdf4llm`` under the lock and yield it, hook installed.

    The quad-corrections flag stays **off** throughout, although pymupdf4llm
    sets it on for itself. The flag is process-global and every other PyMuPDF
    read in the package takes no lock, so turning it on here -- even only for
    the duration -- would let a build on another thread read its pages with
    mangled cells and shifted blocks and cache them in its artifact. Off costs
    the markdown almost nothing, measured with ``header``/``footer``/``use_ocr``
    as ``extract_table_markdown`` passes them: identical on every page of the
    BME280 (60/60) and on 20/20 sampled PSoC 6 pages, and on 45/57 TI ADS1115
    pages, where the rest differ only in line breaks inside plot-axis text
    (``"Bus VCM"`` vs ``"Bus<br>VCM"``). The layout branch never calls
    ``find_tables``, the only other thing that toggles the flag.

    The reset comes *after* the import, in a ``finally``: when this is
    pymupdf4llm's first import in the process, the import itself is what turns
    the flag on, whether or not it then succeeds.

    Also imports ``pymupdf.layout`` here, under the lock, whenever pymupdf4llm
    means to use it. Since PyMuPDF 2.1, pymupdf4llm decides that from
    ``find_spec`` alone and the engine would otherwise be imported by the first
    ``page.get_layout()`` inside ``to_markdown``: outside the lock, and with a
    missing or broken ``onnxruntime`` as a raw error from deep inside the call.
    Importing it here keeps the hook installation under the lock and makes
    :func:`layout_active` truthful again: when the engine cannot be imported,
    ``use_layout(False)`` routes ``to_markdown`` to the classic renderer, as
    PyMuPDF 2.0 did on its own. That also turns ``pymupdf._layout_enabled``
    off process-wide, which :func:`classic_tables` holds on for the scan.

    Raises ``ImportError`` when pymupdf4llm cannot be imported, which with
    ``pymupdf>=2.1`` means ``pymupdf-lite`` was installed in its place.
    """
    with _LAYOUT_LOCK:
        try:
            module = importlib.import_module("pymupdf4llm")
            if layout_active(module):
                try:
                    engine = importlib.import_module("pymupdf.layout")
                except ImportError as exc:
                    logger.warning(
                        "PyMuPDF's layout engine could not be imported (%s); "
                        "table markdown falls back to the classic renderer. "
                        "Check that onnxruntime imports.",
                        exc,
                    )
                    module.use_layout(False)
                else:
                    # Invariant: the engine is imported, so the hook must exist.
                    # Only a stale restore can have cleared it; activate()
                    # reinstalls it and is a no-op when it is already there.
                    if getattr(pymupdf, "_get_layout", None) is None:
                        engine.activate()
        except ImportError as exc:
            raise ImportError(
                f"pymupdf4llm could not be imported ({exc}); table markdown "
                "extraction needs it. It ships inside the full pymupdf wheel; "
                "pymupdf-lite does not have it. Install pymupdf instead of "
                "pymupdf-lite."
            ) from None
        finally:
            pymupdf.TOOLS.unset_quad_corrections(False)
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
