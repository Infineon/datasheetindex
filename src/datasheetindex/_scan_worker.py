"""What runs in a table-scan process before that process imports PyMuPDF.

Two entry points: :func:`init_pool_worker`, the initializer of every scan pool
worker, and :func:`main`, the Windows helper below. Both must run before
``import pymupdf``, so this module imports nothing but the standard library,
and it lives outside ``datasheetindex.core``, whose ``__init__`` imports
PyMuPDF. A worker resolves its initializer by importing the module that
defines it, and ``spawn`` re-imports a ``-m`` entry point by name, so either
import reaching PyMuPDF would load the engine this module exists to skip.

**The Windows helper.** A ``ProcessPoolExecutor`` created *inside* an MCP stdio
server deadlocks on Windows: the workers are created but freeze before their
interpreter finishes initialising, and only unblock when the server dies, so
``shutdown(wait=True)`` never returns. It is not our bug -- the MCP Python SDK
tracks it as modelcontextprotocol/python-sdk#817, still open -- and it is not
fixable from inside the worker: :func:`init_pool_worker` runs as a pool
*initializer*, long after the point where the child is already stuck.

The fix is to not build the pool in the server process at all. :func:`main`
runs as a plain, stdio-detached child (see ``_build_table_count_cache_helper``),
and a plain process pools normally on Windows: measured 31.4s on a 148-page
datasheet that previously hung forever.

Invoked as ``python -m datasheetindex._scan_worker`` rather than by file path,
so ``__main__.__spec__`` is set and ``spawn`` re-imports this module *by name*.
Re-importing by path would re-execute the file in every worker, which is what
the ``__main__`` guard below guards against.

Results travel through a temp file, not stdout: the caller keeps this process's
stdout on devnull so nothing it or its own children print can be mistaken for
JSON-RPC by a parent MCP server.
"""

from __future__ import annotations

import json
import os
import sys
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from types import ModuleType

# Both imported by PyMuPDF 2.x's own __init__, each inside a try/except
# ImportError. A None entry in sys.modules makes such an import raise, so
# PyMuPDF comes up exactly as pymupdf-lite does: no ONNX engine, no hook.
_LAYOUT_MODULES = ("pymupdf.layout", "pymupdf4llm")


def skip_layout_engine() -> None:
    """Keep PyMuPDF's ONNX layout engine out of this process.

    Under PyMuPDF 2.x a plain ``import pymupdf`` imports ``pymupdf.layout``
    and ``pymupdf4llm``, which load onnxruntime and the ONNX model (~2s and
    ~90MB per process) and turn quad corrections off. The table scan runs only
    the classic detector, so it uses neither. Measured on a 134-page build
    whose host ``__main__`` does not import PyMuPDF, i.e. whose workers each
    paid it: 18.3s before, 6.8s after, artifacts byte-identical.

    A no-op once PyMuPDF is imported: under ``spawn`` the host's ``__main__``
    is re-imported before the initializer runs, and if that imports PyMuPDF
    the engine is already loaded. ``classic_tables()`` still guards the scan
    then, which is why it stays load-bearing.
    """
    if "pymupdf" in sys.modules:
        return
    # None is sys.modules' documented "import blocked" value, which its
    # dict[str, ModuleType] annotation does not admit.
    modules = cast("dict[str, ModuleType | None]", sys.modules)
    for name in _LAYOUT_MODULES:
        modules.setdefault(name, None)
    # With layout looking uninstalled, find_tables() would print() PyMuPDF's
    # "Consider using the pymupdf_layout package" advice. Process-local: this
    # is our own scan process, never the host.
    os.environ["PYMUPDF_SUGGEST_LAYOUT_ANALYZER"] = "0"


def init_pool_worker() -> None:
    """Initializer of every table-scan pool worker.

    Redirects stdin/stdout to devnull, then :func:`skip_layout_engine`.

    Workers start under a non-fork method (see ``structure._mp_context``), so
    they do not inherit the parent's open file descriptors -- except 0, 1 and 2,
    which survive ``exec`` on every platform. When the parent is an MCP stdio
    server, an inherited stdin/stdout collides with JSON-RPC communication,
    causing deadlocks. Only 0 and 1 are redirected; stderr is deliberately left
    intact so worker tracebacks still reach the parent's logs.
    """
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)  # stdin
    os.dup2(devnull, 1)  # stdout
    os.close(devnull)
    skip_layout_engine()


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 3:
        print(
            "usage: python -m datasheetindex._scan_worker "
            "<pdf_path> <total_pages> <out_json>",
            file=sys.stderr,
        )
        return 2

    # This process only scans, so it skips the engine too -- before the import
    # below, which is what would load it.
    skip_layout_engine()
    from datasheetindex.core.structure import _build_table_count_cache_pool

    pdf_path, total_pages, out_path = args[0], int(args[1]), args[2]
    cache = _build_table_count_cache_pool(pdf_path, total_pages)

    # Keys are page indices; JSON object keys are always strings, and the
    # caller converts them back.
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump({str(page): count for page, count in cache.items()}, handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
