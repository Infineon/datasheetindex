# Pinning `table_count` to one table engine

Design for GitHub issue #12: `enrich_with_table_counts` parallel and sequential
paths disagree once `pymupdf4llm` is imported.

Status: approved, not yet implemented. Target: `datasheetindex` 0.17.3.

## Problem

`table_count` (and `has_tables`, derived from it in `structure.py:435`) is not a
stable property of a document. The same PDF yields different numbers depending
on the history of the process that indexed it.

`enrich_with_table_counts` has two paths:

- **parallel** (`_build_table_count_cache_parallel`, used when `pdf_path` is
  available and the document has >= 12 pages) runs `find_tables()` in fresh
  worker processes started under `forkserver`/`spawn`. Those workers import only
  `pymupdf`, so they always use PyMuPDF's classic geometric detector.
- **sequential** (`_build_table_count_cache_sequential`, the fallback) runs
  `find_tables()` in the caller's process, which uses the ML layout engine if
  anything in that process has imported `pymupdf4llm`.

Which path runs depends on page count, on whether `pdf_path` is available, and
on whether the pool happened to fail and fall back. So the reported count
depends on facts that have nothing to do with the document.

## Mechanism (corrected)

The issue states that importing `pymupdf4llm` "replaces PyMuPDF's
`find_tables()`". That is not what happens, and the difference matters for the
fix.

`pymupdf.Page.find_tables` is never rebound; its identity is unchanged across
the import. What changes is a single module-level global, `pymupdf._get_layout`
(`pymupdf/__init__.py:340`, default `None`):

- `pymupdf/layout/__init__.py` calls `activate()` at import time, which assigns
  an ONNX-backed callable to `pymupdf._get_layout`.
- `pymupdf4llm/__init__.py` does `import pymupdf.layout` at import time and
  calls `use_layout(True)`. `pymupdf4llm` declares `pymupdf_layout==1.28.0` as a
  hard dependency, so installing the `[layout]` extra always brings the engine.
- `find_tables()` (`pymupdf/table.py:2671-2686`) calls `page.get_layout()`,
  which consults `pymupdf._get_layout` **at call time**. When the hook returns
  boxes, they become the tables; when it runs and finds none, `find_tables()`
  short-circuits to an empty result without ever trying the classic detector.

Three consequences follow, all verified:

1. The engine is selected per call, from a global. It can be toggled.
2. Saving and restoring `pymupdf._get_layout` costs ~20 microseconds in each
   direction. (`pymupdf4llm.use_layout(True)`, the public re-enable, costs
   ~1.1 s because it re-fetches the model. It is the wrong primitive here.)
3. `find_tables()` exposes **no per-call argument** to choose the engine.
   Forcing the classic detector in-process therefore requires touching the
   global. There is no cleaner in-process alternative.

## Evidence

Measured with `pymupdf` 1.28.0, `pymupdf4llm` 1.28.0, `pymupdf-layout` 1.28.0,
Python 3.13, Linux.

**The divergence reproduces**: on a 13-page synthetic document whose tables are
drawn with horizontal rules only, the parallel path totals 0 tables and the
sequential path totals 13.

**The issue's quality claim does not generalize.** Its table ("classic 0 /
ML 1" for unruled and horizontal-rule-only styles) is derived from synthetic
fixtures. On a real 68-page datasheet (TI LM358, `ti.com/lit/ds/symlink/lm358.pdf`):

| | classic | ML layout |
|---|---|---|
| tables found | 75 | 39 |
| pages with >= 1 table | 45 | 32 |
| full-document scan | 23.6 s | 104.2 s |
| pages where the *other* engine finds nothing | 3 | 16 |

The 16 pages where ML finds nothing are pages 14-23, the "Typical
Characteristics" sections: the classic detector is false-positiving on chart
gridlines, which `docs/datasheetindex_architecture.md:155` already documents as
expected ("false positives on block diagrams are expected"). The 3 pages the
classic detector misses are the table of contents and the unruled revision
history.

Neither engine dominates. ML is more precise and 4.4x slower; classic is noisier
and fast. **The defect is the instability, not the engine choice.**

## Why the bug hid

A plain `uv sync` installs the `dev` group, which pulls `datasheetindex[mcp]`
but not `[layout]` -- it actively uninstalls `pymupdf-layout`. CI therefore runs
the classic engine everywhere and the two paths always agree.

A developer who has run `uv sync --extra layout` (or `--all-extras`) has the ML
engine available locally. `tests/test_registry.py:371` calls the
`extract_table_markdown` handler, which imports `pymupdf4llm` and activates the
hook for **every test that runs afterwards in that process**. The existing table
fixtures are fully ruled grids, the one style both engines agree on, so this
order-dependent contamination has never produced a failure.

## Decision

**`table_count` means "PyMuPDF's classic geometric detector", always.**

Both counting paths are pinned to the classic engine. The count becomes
identical whether or not `[layout]` is installed, which path ran, and what the
process imported earlier. The ML engine remains available, isolated to
`extract_table_markdown`, which is where it was actually wanted.

Rationale:

- `table_count` is already specified as a heuristic navigational hint, not a
  precise count (`docs/datasheetindex_architecture.md:155`).
- Classic is the engine every install without the optional `[layout]` extra
  already uses. Pinning to it keeps counts comparable across environments and
  changes nothing for the default install.
- It preserves the parallel path's speed. Adopting ML instead would require each
  worker to import `pymupdf4llm` (a ~1-2 s ONNX load per worker) on top of a
  4.4x slower scan, which would plausibly make the parallel path slower than the
  sequential one it exists to replace.
- The evidence does not support ML being uniformly better, so making the number
  depend on which optional extra happens to be installed would itself be a bug.

### Rejected alternatives

- **Make the workers match the parent** (issue option 1). Pays the ONNX load per
  worker and defeats the parallel path's purpose. Also makes counts depend on
  whether `[layout]` is installed.
- **Subprocess the sequential fallback** (issue option 2, one variant). Cannot
  work when `pdf_path` is `None` (in-memory documents), so an in-process guard
  would still be needed for that case, leaving two mechanisms.
- **An explicit `table_engine=` parameter** with engine provenance recorded in
  the JSON. Deterministic and honest, but adds public API surface and a second
  code path to test, for a number that is documented as a hint. YAGNI.

### Non-goals

This change does **not** improve the counts. The classic detector still
over-counts plot pages. Making `table_count` more accurate is a separate
question, and one the ML engine answers only by trading a 4.4x slowdown for a
different set of errors. This change makes the number trustworthy and stable.

## Design

### 1. New module: `src/datasheetindex/core/engine.py`

The single owner of PyMuPDF's layout global **and of the `pymupdf4llm` import
that installs it**. Roughly 55 lines.

```python
_LAYOUT_LOCK = threading.RLock()
_MISSING = object()


@contextmanager
def classic_tables() -> Iterator[None]:
    """Pin find_tables() to PyMuPDF's classic detector for the duration."""
    with _LAYOUT_LOCK:
        saved = getattr(pymupdf, "_get_layout", _MISSING)
        if saved is _MISSING:
            yield  # no hook in this PyMuPDF: find_tables() is already classic
            return
        pymupdf._get_layout = None
        try:
            yield
        finally:
            pymupdf._get_layout = saved


@contextmanager
def layout_engine() -> Iterator[Any]:
    """Import pymupdf4llm under the lock and yield it with its hook installed."""
    with _LAYOUT_LOCK:
        try:
            module = importlib.import_module("pymupdf4llm")
        except ImportError:
            raise ImportError(
                "pymupdf4llm is required for table markdown extraction. "
                "Install it with: uv sync --extra layout"
            ) from None
        # Invariant: if pymupdf4llm believes layout is on, the hook must exist.
        # Anything else makes to_markdown() raise inside _layout_to_markdown.
        if getattr(module, "_use_layout", False) and (
            getattr(pymupdf, "_get_layout", None) is None
        ):
            module.use_layout(True)  # reinstalls the hook (~1.1 s, rare)
        yield module
```

Notes:

- **`layout_engine()` performs the import.** This is the whole point, not an
  incidental tidy-up. `pymupdf4llm`'s import is what calls
  `pymupdf.layout.activate()` and installs the hook, so an import outside the
  lock can interleave with `classic_tables()`: A saves `None`, B's import
  installs the hook, A restores the stale `None`. Because
  `pymupdf4llm._use_layout` remains `True`, `to_markdown()` then routes into
  `_layout_to_markdown`, which iterates `page.layout_information` -- now `None`
  -- and raises `TypeError: 'NoneType' object is not iterable`. The module is
  cached, so re-importing never reactivates: **every subsequent
  `extract_table_markdown()` call in that process raises.** Verified.
- The `_use_layout` invariant check makes that state unreachable and
  self-healing even if a third party nulls the global. It is gated on
  `_use_layout` so that a legitimate non-layout `pymupdf4llm` install (one whose
  `import pymupdf.layout` failed, leaving `use_layout(False)`) is not
  "repaired" into an engine it does not have.
- `_MISSING` sentinel, not a `None` default: a PyMuPDF with no `_get_layout`
  attribute is a genuine no-op, and the early `return` leaves no attribute
  behind. Assigning `None` and restoring `None` would have created one.
- `RLock`, not `Lock`, so that nesting cannot deadlock.
- Restoring the saved callable directly, rather than calling
  `pymupdf4llm.use_layout(True)`, keeps `classic_tables()`'s exit free
  (~20 us vs ~1.1 s) and avoids importing `pymupdf4llm` in a process that never
  needed it.

`core/` is the right home: `tools/bound.py` already imports from `core/`
(`core.locate`, `core.structure`, `core.textfile`), so this introduces no new
dependency direction.

### 2. `src/datasheetindex/core/structure.py`

Wrap the `find_tables()` call in `classic_tables()` in **both** places:

- `_build_table_count_cache_sequential` (line ~370) -- this is the actual fix.
- `_count_tables_on_page` (line ~187), the worker body -- a no-op today, because
  workers never import `pymupdf4llm`. It is included so that "classic" is a
  property of the counting function rather than an accident of the worker's
  import graph: a future `set_forkserver_preload`, or a PyMuPDF that activates
  layout on import, cannot silently reintroduce the drift.

Update `enrich_with_table_counts`'s docstring to state that counts come from the
classic detector regardless of process state.

### 3. `src/datasheetindex/tools/bound.py`

`extract_table_markdown` delegates both the import and the call to
`layout_engine()`, so the activation and the use of the hook happen inside one
critical section:

```python
with layout_engine() as pymupdf4llm:
    return pymupdf4llm.to_markdown(self.doc, pages=[page - 1], show_progress=False)
```

The local `importlib.import_module("pymupdf4llm")` and its `ImportError`
message move into `engine.py`, leaving `bound.py` with no direct knowledge of
the optional dependency. Keeping the import here -- outside the lock -- is
precisely the defect described above, so this is not a stylistic choice.

This is necessary because `build_datasheet` and `extract_table_markdown` both
run under `asyncio.to_thread` (`tools/defs.py:156` and `:240`), so they can
genuinely execute concurrently in different threads. Without the lock, an
`extract_table_markdown` overlapping an in-flight sequential scan would silently
return non-layout markdown -- the same silent-wrong-answer class this issue is
about.

`tools/defs.py:106` already documents that handlers are not safe under
concurrent invocation *within one session*, but an SSE host running two sessions
in one process is outside that contract, and the global is process-wide.

Cost: when `[layout]` is absent -- the default install, and CI -- the lock is
uncontended and the guard is a no-op, so the common path pays nothing. When
`[layout]` is present, a table-markdown call blocks behind an in-flight
sequential scan. Acceptable: the sequential path runs only for documents under
12 pages, for in-memory documents, or after a pool failure.

### 4. `src/datasheetindex/mcp_server.py`

`_preload_layout_model()` keeps its purpose and its docstring -- warming the
ONNX model at server start, which once counting is pinned benefits only
`extract_table_markdown` -- but stops importing `pymupdf4llm` itself:

```python
def _preload_layout_model() -> None:
    with contextlib.suppress(ImportError):
        with layout_engine():
            pass
```

It runs before serving begins, so it cannot actually race today. Routing it
through `engine.py` anyway leaves exactly one import site for `pymupdf4llm` in
the package, which is what makes "the hook is only ever installed under the
lock" a property you can check by grepping rather than by reasoning.

## Testing

The regression test the issue asks for must fail before the fix and pass after,
**in CI, where `[layout]` is not installed**. This is possible: `_get_layout`
exists in stock `pymupdf` 1.28.0 (defaulting to `None`), and assigning a fake
hook drives the real layout branch of `find_tables()` with no ML engine present.
Verified: with a stub returning one `"table"` box per page, `find_tables()`
reports 1 table per page on a fixture where the classic detector reports 0.

New `tests/test_engine.py`:

- `classic_tables()` clears the hook inside the block and restores it after.
- It restores the hook when the body raises.
- It nests without deadlocking, and the innermost exit restores correctly.
- When `pymupdf` has no `_get_layout` attribute (simulate with
  `monkeypatch.delattr`), `classic_tables()` is a true no-op: the body runs and
  the attribute is **still absent afterwards** (`not hasattr(pymupdf,
  "_get_layout")`). This is the assertion the previous sketch would have failed.
- `layout_engine()` yields the module and holds `_LAYOUT_LOCK` for the whole
  body (assert `_LAYOUT_LOCK` is held inside; a full thread-race test is not
  worth its flakiness).
- `layout_engine()` raises the friendly `ImportError` when `pymupdf4llm` is
  absent (simulate by making `importlib.import_module` raise).
- **The stale-restore regression**, with a stub `pymupdf4llm` module: enter
  `classic_tables()` while no hook is installed, have the stub's import install
  one, exit, and assert `layout_engine()` still yields a module whose hook is
  live. Reproduces the permanent-`TypeError` corruption if the import ever
  escapes the lock.
- The `_use_layout` invariant repair fires when the hook is `None` but the
  module reports `_use_layout = True`, and does **not** fire when `_use_layout`
  is `False`.

In `tests/test_structure.py`:

- A new **unruled** fixture (text columns with horizontal rules only, no
  vertical cell borders) alongside the existing ruled-grid fixture. The existing
  fixture cannot catch this bug, because it is the one style both engines agree
  on.
- With a fake `_get_layout` hook installed (via `monkeypatch.setattr`),
  `_build_table_count_cache_sequential` must return the **classic** counts on
  the unruled fixture. This fails before the fix (it would report one table per
  page from the stub) and passes after.
- A direct parallel-vs-sequential agreement test on the unruled fixture, with
  the fake hook installed in the parent. This is newly possible.
- Correct the docstrings at `test_structure.py:639` and `:658`, which currently
  explain that the two paths cannot be compared because they may use different
  engines. That caveat is exactly what this change removes.

Nothing here needs the `[layout]` extra, so CI coverage is real.

## Documentation

- `docs/datasheetindex_architecture.md:155`: extend the existing note to say
  that `has_tables` and `table_count` come from PyMuPDF's classic geometric
  detector, are identical whether or not the `[layout]` extra is installed, and
  that plots and block diagrams produce expected false positives.
- `CHANGELOG.md`: a `Fixed` entry under a new `0.17.3`. It should also correct
  the 0.17.2 entry's claim that importing `pymupdf4llm` "replaces PyMuPDF's
  `find_tables()`" -- it sets a global hook that `find_tables()` consults.

## Risks

- **A future PyMuPDF renames the hook.** Layout would be active while
  `classic_tables()` silently no-ops, reintroducing the drift. Not detectable
  from CI, which has no layout engine. Mitigated by the fact that the counting
  fixture would still be internally consistent; accepted, and noted in the
  module docstring.
- **The lock serializes table-markdown behind index builds.** Bounded by the
  sequential path's rarity (see above), and absent entirely without `[layout]`.
- **`RLock` permits `layout_engine()` nested inside `classic_tables()` on one
  thread**, which would call `to_markdown()` with the hook suppressed. No such
  path exists, and the `_use_layout` invariant check reinstalls the hook rather
  than crashing, but the nesting is silently wrong rather than loud. Noted in
  the module docstring; not worth a re-entrancy guard for a path nothing takes.
- **A third party importing `pymupdf4llm` directly** (a host application, a
  notebook) still installs the hook outside our lock. `engine.py` owns every
  import inside the package, which is all it can own.

## Acceptance criteria

1. On a document whose tables are drawn with horizontal rules only, the parallel
   and sequential paths return identical counts, in a process that has imported
   `pymupdf4llm`.
2. `table_count` for a given PDF is unchanged by installing `[layout]`.
3. `extract_table_markdown` still returns layout-aware markdown (pipe-delimited
   tables) when `[layout]` is installed.
4. In a process that has **never** imported `pymupdf4llm`, an index build
   (which enters and exits `classic_tables()`) followed by an
   `extract_table_markdown()` call returns layout-aware markdown rather than
   raising `TypeError`. This is the corruption the original design admitted.
5. `pymupdf4llm` is imported in exactly one place in `src/`:
   `core/engine.py`. Enforceable by grep.
6. The full test suite passes under a plain `uv sync` and under
   `uv sync --extra layout`.
