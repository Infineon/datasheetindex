"""Out-of-process entry point for the parallel table-count scan.

Exists for Windows. A ``ProcessPoolExecutor`` created *inside* an MCP stdio
server deadlocks there: the workers are created but freeze before their
interpreter finishes initialising, and only unblock when the server dies, so
``shutdown(wait=True)`` never returns. It is not our bug -- the MCP Python SDK
tracks it as modelcontextprotocol/python-sdk#817, still open -- and it is not
fixable from inside the worker: ``_subprocess_init`` runs as a pool
*initializer*, long after the point where the child is already stuck.

The fix is to not build the pool in the server process at all. This module runs
as a plain, stdio-detached child (see ``_build_table_count_cache_helper``), and
a plain process pools normally on Windows: measured 31.4s on a 148-page
datasheet that previously hung forever.

Invoked as ``python -m datasheetindex.core._scan_worker`` rather than by file
path, so ``__main__.__spec__`` is set and ``spawn`` re-imports this module *by
name*. Re-importing by path would re-execute the file in every worker, which is
what the ``__main__`` guard below guards against.

Results travel through a temp file, not stdout: the caller keeps this process's
stdout on devnull so nothing it or its own children print can be mistaken for
JSON-RPC by a parent MCP server.
"""

from __future__ import annotations

import json
import sys


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 3:
        print(
            "usage: python -m datasheetindex.core._scan_worker "
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
