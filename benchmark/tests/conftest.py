"""Session-wide protection for the archive.

Four separate ways of silently overwriting `archive/` have now been found
across three review rounds, plus one bypass of the guard added to stop them.
Every one was a different script, so per-script guards keep missing the next
one; the class needs a check that does not depend on knowing where the write
will come from.

This digests the whole archive before any test runs and again at the end. If a
single byte moved, the run fails -- whatever caused it, including a test that
shells out to a script whose defaults point at the wrong place, which is what
happened last time.
"""

from __future__ import annotations

import hashlib

import pytest

from chamberbench.claimsio import archive_dir


def _digest() -> dict[str, str]:
    root = archive_dir()
    if not root.exists():
        return {}
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


@pytest.fixture(scope="session", autouse=True)
def _archive_is_read_only():
    """Fail the run if anything under `archive/` changed while tests ran."""
    before = _digest()
    yield
    after = _digest()
    if before == after:
        return

    changed = sorted(k for k in before.keys() & after.keys() if before[k] != after[k])
    added = sorted(after.keys() - before.keys())
    removed = sorted(before.keys() - after.keys())
    detail = "\n".join(
        [f"  modified: {k}" for k in changed]
        + [f"  added:    {k}" for k in added]
        + [f"  removed:  {k}" for k in removed]
    )
    pytest.fail(
        "the test run modified the archive, which is primary evidence:\n"
        + detail
        + "\n\nA test must never write there. Point the code under test at a "
        "tmp_path via CHAMBERBENCH_ARCHIVE_DIR instead.",
        pytrace=False,
    )
