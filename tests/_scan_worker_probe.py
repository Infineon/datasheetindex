"""Report what a real table-scan pool worker has imported. Never collected.

Run as a script by tests/test_scan_worker.py. Its top level imports only the
standard library on purpose: under ``forkserver`` this file is preloaded into
the fork server as ``__mp_main__`` and under ``spawn`` it is re-imported in
every worker, so a top-level ``import pymupdf`` here would load the ONNX engine
before the initializer runs and make the probe report the host, not the fix.
That is the library-host case the fix targets -- a ``__main__`` that does not
import PyMuPDF -- and it is the case where the engine cost was paid per worker.

``structure._count_tables_on_page`` is swapped for :func:`probe` in the parent,
so the pool is the production one, ``_build_table_count_cache_pool``, with its
real initializer and start method. The probe imports what the real task's
unpickling imports, then reports. Prints one JSON object: page -> modules.

    python tests/_scan_worker_probe.py
"""

import json
import sys

_WATCHED = ("pymupdf", "pymupdf.layout", "pymupdf4llm", "onnxruntime")


def probe(args: tuple[str, int]) -> tuple[int, list[str]]:
    import datasheetindex.core.structure  # noqa: F401 -- what the real task imports

    return args[1], [name for name in _WATCHED if sys.modules.get(name)]


if __name__ == "__main__":
    from datasheetindex.core import structure

    # Its "counts" are module lists, which is the point; the annotation says int.
    structure._count_tables_on_page = probe  # ty: ignore[invalid-assignment]
    pages = 2
    cache = structure._build_table_count_cache_pool("unused.pdf", pages)
    print(json.dumps({str(page): cache[page] for page in range(pages)}))
