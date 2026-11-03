"""What runs in a table-scan process before that process imports PyMuPDF.

Two entry points: :func:`init_pool_worker`, the initializer of every scan pool
worker, and :func:`main`, the Windows helper below. The module imports nothing
but the standard library and lives outside ``datasheetindex.core``, whose
``__init__`` imports PyMuPDF, so resolving the initializer stays cheap.

Scan processes never load PyMuPDF's ONNX layout engine. That needs nothing
here: PyMuPDF 2.1 imports the engine on first use only, and the scan reaches
``find_tables()`` through ``engine.find_classic_tables``, which does not ask
for it. ``tests/test_scan_worker.py`` pins it for the pool and the helper.

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


def init_pool_worker() -> None:
    """Initializer of every table-scan pool worker.

    Redirects stdin/stdout to devnull.

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


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 3:
        print(
            "usage: python -m datasheetindex._scan_worker "
            "<pdf_path> <total_pages> <out_json>",
            file=sys.stderr,
        )
        return 2

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
