"""Drive the real public path in a fresh interpreter, from PyMuPDF's import state.

Run as a script by tests/test_layout_integration.py, never collected. A fresh
interpreter is the only way to observe the state ``import pymupdf`` leaves
behind: PyMuPDF 2.x imports pymupdf4llm and the ONNX engine with it, which
installs ``pymupdf._get_layout`` and turns ``TOOLS.unset_quad_corrections`` on,
and by the time pytest reaches this file other tests have long since been
through ``layout_engine()``.

What this proves, precisely:

* importing the package resets the quad-corrections flag that pymupdf4llm set,
  and leaves the layout hook PyMuPDF installed alone;
* a build in that process reports the *classic* counts, and afterwards the hook
  is the very callable it was and the flag is still off -- so the build's
  classic_tables() round-trip restored both;
* after that round-trip, extract_table_markdown still returns layout-aware
  markdown, and puts the flag back off when it is done -- the leak this guards
  is what made every later read in a long-lived process see different table
  cells and text-block coordinates;
* the whole thing runs through DatasheetTools.build_datasheet and
  DatasheetTools.extract_table_markdown, so a regression in that wiring fails
  here even if engine.py itself is fine.

What this does NOT prove: the permanent-TypeError corruption needs a
pymupdf4llm import to land *between* classic_tables()'s save and restore, which
cannot happen on one thread. That interleaving is guarded deterministically by
test_layout_engine_installs_the_hook_under_the_lock in tests/test_engine.py,
which asserts the lock is held across the import.

Exits non-zero with a message on any failure. Usage:

    python tests/_fresh_layout_process.py <pdf-path> <output-dir>
"""

import sys

import pymupdf

if not pymupdf.TOOLS.unset_quad_corrections():
    raise AssertionError("precondition failed: import pymupdf left the flag off")
if not callable(getattr(pymupdf, "_get_layout", None)):
    raise AssertionError("precondition failed: import pymupdf installed no hook")
HOOK = pymupdf._get_layout

from datasheetindex.tools.bound import DatasheetTools  # noqa: E402

EXPECTED_TOTAL_TABLES = 9  # sum([1, 2, 1, 2, 1, 2]), the classic detector


def _sum_table_counts(nodes: list[dict]) -> int:
    total = 0
    for node in nodes:
        total += node.get("table_count", 0)
        total += _sum_table_counts(node.get("nodes", []))
    return total


def _assert_process_state(when: str) -> None:
    if pymupdf.TOOLS.unset_quad_corrections():
        raise AssertionError(f"quad corrections are unset {when}")
    if pymupdf._get_layout is not HOOK:
        raise AssertionError(f"the layout hook was replaced {when}")


def main(pdf_path: str, output_dir: str) -> None:
    _assert_process_state("after importing datasheetindex")

    with DatasheetTools(pdf_path) as tools:
        artifacts = tools.build_datasheet(output_dir=output_dir)
        _assert_process_state("after build_datasheet")

        total = _sum_table_counts(artifacts.json_data.get("toc", []))
        if total != EXPECTED_TOTAL_TABLES:
            raise AssertionError(
                f"classic table_count wrong: {total} != {EXPECTED_TOTAL_TABLES}"
            )

        # First layout use in this process, after the build's guard round-trip.
        markdown = tools.extract_table_markdown(1)
        _assert_process_state("after extract_table_markdown")

    if "|" not in markdown:
        raise AssertionError(f"markdown is not layout-aware: {markdown!r}")

    print("OK")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
