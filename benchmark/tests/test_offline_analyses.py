"""Three paper-backing analyses that Tier 1 omitted. All offline: they
read the archive and need no credentials."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

ANALYSES = [
    "grounding_wrong_document.py",
    "dispatch_stats.py",
    "compute_paper_numbers.py",
]

_KEY_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "LITELLM_MASTER_KEY")


@pytest.mark.parametrize("name", ANALYSES)
def test_analysis_runs_offline(name, monkeypatch):
    """No API key in the environment; these must still work.

    Setting dummy keys first, rather than trusting the ambient test
    environment to already be clean, is what makes this real evidence: a
    filter that silently no-ops (a typo'd key name, say) would still pass
    this test if the parent process happened to have no keys set, which
    proves nothing about a reader's machine.
    """
    for var in _KEY_VARS:
        monkeypatch.setenv(var, "sk-test-should-never-reach-the-subprocess")
    env = {k: v for k, v in os.environ.items() if k not in _KEY_VARS}
    assert not any(key in env for key in _KEY_VARS), env.keys() & set(_KEY_VARS)
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / name)],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    if name == "grounding_wrong_document.py":
        # returncode == 0 cannot tell "a guard fired and printed its message"
        # apart from "something else caused an early silent return" -- pin
        # which branch actually ran. There are three, and exactly one must:
        # the corpus PDFs are third-party datasheets deliberately not
        # redistributed with the archive, and datasheetindex ships with the
        # harness extra rather than with Tier 1, so on a default install a
        # guard is what runs. Enumerating all three is what keeps this test
        # from going quietly inert: a reader who fetches the corpus on a
        # Tier-1 install used to reach an unguarded import and a traceback,
        # and this test could not see it because it only ever ran in the
        # corpus-absent state.
        branches = {
            "corpus": "corpus datasheet(s) not found" in result.stdout,
            "library": "datasheetindex not installed" in result.stdout,
            "analysis": "decoy served:" in result.stdout,
        }
        ran = [k for k, v in branches.items() if v]
        assert len(ran) == 1, (ran, result.stdout)


@pytest.mark.parametrize("name", ANALYSES)
def test_analysis_makes_no_api_calls(name):
    text = (SCRIPTS / name).read_text(encoding="utf-8")
    for forbidden in ("anthropic", "openai", "extract_chamber", "anthropic_path"):
        assert forbidden not in text.lower(), (name, forbidden)
