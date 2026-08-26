"""What a full reproduction costs, computed from the archive itself."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "reproduction_cost.py"


def _load():
    spec = importlib.util.spec_from_file_location("reproduction_cost", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_reports_every_archived_arm():
    summary = _load().summarise()
    assert set(summary) >= {"claudesonnet4.6", "gpt-5.1", "qwen3.6-27b"}


def test_token_counts_are_positive():
    for arm, usage in _load().summarise().items():
        assert usage["input_tokens"] > 0, arm
        assert usage["output_tokens"] > 0, arm


def test_runs_offline_and_writes_nothing():
    before = {
        p: p.stat().st_mtime
        for p in (SCRIPT.parents[1] / "archive").rglob("*")
        if p.is_file()
    }
    _load().summarise()
    after = {
        p: p.stat().st_mtime
        for p in (SCRIPT.parents[1] / "archive").rglob("*")
        if p.is_file()
    }
    assert before == after
