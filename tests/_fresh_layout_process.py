"""Drive the real public path in an interpreter that has not imported pymupdf4llm.

Run as a script by tests/test_layout_integration.py, never collected. A fresh
interpreter is the only way to establish the precondition: pytest collects
alphabetically, and tests/test_defs.py imports pymupdf4llm (via
extract_table_markdown) long before tests/test_layout_integration.py runs.

What this proves, precisely:

* a build in a pristine process reports the *classic* counts, and afterwards
  neither pymupdf4llm nor pymupdf.layout is in sys.modules and
  pymupdf._get_layout is still None -- so the build activated no engine by any
  route, not merely by the import we happen to grep for;
* after that build's classic_tables() round-trip, the first layout use in the
  process still installs the hook and returns layout-aware markdown -- i.e. the
  guard does not leave pymupdf._get_layout permanently nulled;
* the whole thing runs through DatasheetTools.build_datasheet and
  DatasheetTools.extract_table_markdown, so a regression in that wiring fails
  here even if engine.py itself is fine.

What this does NOT prove: the permanent-TypeError corruption needs the
pymupdf4llm import to land *between* classic_tables()'s save and restore, which
cannot happen on one thread -- a build's round-trip completes before
extract_table_markdown imports anything. That interleaving is guarded
deterministically by test_layout_engine_installs_the_hook_under_the_lock in
tests/test_engine.py, which asserts the lock is held across the import.

Exits non-zero with a message on any failure. Usage:

    python tests/_fresh_layout_process.py <pdf-path> <output-dir>
"""

import sys

import pymupdf

from datasheetindex.tools.bound import DatasheetTools

EXPECTED_TOTAL_TABLES = 9  # sum([1, 2, 1, 2, 1, 2]), the classic detector


def _sum_table_counts(nodes: list[dict]) -> int:
    total = 0
    for node in nodes:
        total += node.get("table_count", 0)
        total += _sum_table_counts(node.get("nodes", []))
    return total


def main(pdf_path: str, output_dir: str) -> None:
    if "pymupdf4llm" in sys.modules:
        raise AssertionError("precondition failed: pymupdf4llm already imported")

    with DatasheetTools(pdf_path) as tools:
        artifacts = tools.build_datasheet(output_dir=output_dir)

        # Assert the hook's *state*, not just the module's absence. A regression
        # that reached the engine via `import pymupdf.layout` or by assigning
        # pymupdf._get_layout directly would leave sys.modules clean.
        if "pymupdf4llm" in sys.modules:
            raise AssertionError("build_datasheet must not import pymupdf4llm")
        if "pymupdf.layout" in sys.modules:
            raise AssertionError("build_datasheet must not import pymupdf.layout")
        if getattr(pymupdf, "_get_layout", None) is not None:
            raise AssertionError("build_datasheet must not activate the layout hook")

        total = _sum_table_counts(artifacts.json_data.get("toc", []))
        if total != EXPECTED_TOTAL_TABLES:
            raise AssertionError(
                f"classic table_count wrong: {total} != {EXPECTED_TOTAL_TABLES}"
            )

        # First layout use in this process, after the build's guard round-trip.
        markdown = tools.extract_table_markdown(1)

    if "|" not in markdown:
        raise AssertionError(f"markdown is not layout-aware: {markdown!r}")

    print("OK")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
