"""The re-grading tool actually re-grades, and says when it graded less.

`regrade_archive.py` is what the documentation offers an outside researcher as
the way to attack the published results, and it shipped with no tests. A tool
that reports agreement regardless of its input would manufacture confidence in
exactly the place the work is weakest, so these pin both halves: that it is
sensitive to the claim set, and that it reports reduced coverage rather than a
cleaner-looking number.
"""

from __future__ import annotations

import copy
import subprocess
import sys

import pytest
import yaml

from chamberbench.claimsio import BENCHMARK_ROOT, archive_dir, data_dir

SCRIPT = BENCHMARK_ROOT / "scripts" / "regrade_archive.py"


def _run(data_dir_override=None):
    env = {"PATH": "/usr/bin:/bin"}
    import os

    env = dict(os.environ)
    if data_dir_override is not None:
        env["CHAMBERBENCH_DATA_DIR"] = str(data_dir_override)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        cwd=BENCHMARK_ROOT,
        env=env,
        check=False,
    )
    return proc


def _counts(stdout):
    out = {}
    for line in stdout.splitlines():
        if "agree with published verdict:" in line:
            out["agree"] = int(line.split(":")[1])
        elif "disagree (verdict would change):" in line:
            out["disagree"] = int(line.split(":")[1])
        elif line.strip().startswith("re-graded:"):
            out["regraded"] = int(line.split(":")[1])
    return out


@pytest.fixture
def claims_copy(tmp_path):
    """A writable copy of the shipped claim set."""
    src = data_dir() / "claims.yaml"
    doc = yaml.safe_load(src.read_text(encoding="utf-8"))

    def write(mutate):
        d = copy.deepcopy(doc)
        mutate(d)
        (tmp_path / "claims.yaml").write_text(
            yaml.safe_dump(d, sort_keys=False), encoding="utf-8"
        )
        return tmp_path

    return write


def test_self_check_reproduces_the_published_verdicts():
    """148 of 149 agree; the one flip is the documented needle repair."""
    proc = _run()
    assert proc.returncode == 0, proc.stderr[-1500:]
    c = _counts(proc.stdout)
    assert c["regraded"] == 149
    assert c["agree"] == 148
    assert c["disagree"] == 1
    assert "acs70331-saturation-low" in proc.stdout


def test_cell_accounting_is_checkable():
    """The denominator must be reconcilable with the archive's own cell count."""
    proc = _run()
    assert "cells considered:      150" in proc.stdout
    assert "with an extraction:  149" in proc.stdout
    assert "without one:         1" in proc.stdout


def test_it_is_sensitive_to_a_broken_needle(claims_copy):
    """The core property: a claim set that cannot match must produce flips.

    A tool that reports 148/1 no matter what it is fed is worse than no tool.
    """

    def break_one(d):
        for c in d["claims"]:
            if c["id"] == "dps310-relative-accuracy":
                c["value_contains"] = ["ZZZ_IMPOSSIBLE"]

    proc = _run(claims_copy(break_one))
    c = _counts(proc.stdout)
    assert c["disagree"] > 1, f"breaking a needle produced no new flips: {c}"


def test_it_is_sensitive_to_the_confidence_floor(claims_copy):
    """Raising every floor to 1.0 must reject nearly everything."""

    def raise_floors(d):
        for c in d["claims"]:
            c["confidence_min"] = 1.0

    proc = _run(claims_copy(raise_floors))
    c = _counts(proc.stdout)
    assert c["disagree"] > 100, f"an impossible floor barely changed anything: {c}"


def test_a_dropped_claim_reports_partial_rather_than_a_clean_score(claims_copy):
    """An abstention must not look like a better result than our own.

    Dropping the one disagreeing claim yields 143/143/0 -- cleaner than the
    shipped self-check. Reporting that at exit 0, with no note that six cells
    went ungraded, is how a tool manufactures false confidence.
    """

    def drop(d):
        d["claims"] = [c for c in d["claims"] if c["id"] != "acs70331-saturation-low"]

    proc = _run(claims_copy(drop))
    assert proc.returncode == 2, "a partial claim set must not exit 0"
    assert "NOT GRADED" in proc.stdout
    assert "acs70331-saturation-low" in proc.stdout
    assert "PARTIAL" in proc.stdout


def test_an_unreadable_claim_set_fails_with_an_actionable_message(tmp_path):
    (tmp_path / "claims.yaml").write_text(
        "claims:\n  - id: x\n    value_contains: [20, mV]\n", encoding="utf-8"
    )
    proc = _run(tmp_path)
    assert proc.returncode == 1
    assert "could not be parsed" in proc.stderr
    assert "quote numeric needles" in proc.stderr


def test_a_missing_claim_file_says_what_is_required(tmp_path):
    proc = _run(tmp_path / "nope")
    assert proc.returncode == 1
    assert "no claim file at" in proc.stderr


def test_the_archive_is_not_written_to():
    """Re-grading is read-only; it must never touch the evidence."""
    before = {p.name: p.stat().st_mtime_ns for p in archive_dir().glob("*.json")}
    _run()
    after = {p.name: p.stat().st_mtime_ns for p in archive_dir().glob("*.json")}
    assert before == after, "regrade_archive modified the archive"
