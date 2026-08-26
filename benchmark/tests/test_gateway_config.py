"""The reference gateway config is the whole model-naming fix, so the
aliases it declares must be the aliases the archive was produced under."""

from __future__ import annotations

from pathlib import Path

import yaml

from chamberbench.claimsio import archive_dir
from chamberbench.harness import CHAMBER_MODEL_CONFIG

CONFIG = Path(__file__).resolve().parents[1] / "gateway" / "litellm_config.yaml"

ARCHIVED_ALIASES = {"claudesonnet4.6", "gpt-5.1", "qwen3.6-27b"}


def test_config_parses():
    assert yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["model_list"]


def test_every_archived_alias_is_declared():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    declared = {entry["model_name"] for entry in cfg["model_list"]}
    assert ARCHIVED_ALIASES <= declared


def test_declared_aliases_are_known_to_the_harness():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    for entry in cfg["model_list"]:
        assert entry["model_name"] in CHAMBER_MODEL_CONFIG, entry["model_name"]


def test_aliases_match_archive_filenames():
    """`latest_chamber.<alias>.json` must keep resolving after the port."""
    for alias in ARCHIVED_ALIASES:
        assert (archive_dir() / ("latest_chamber." + alias + ".json")).exists(), alias


def test_no_internal_hostnames():
    text = CONFIG.read_text(encoding="utf-8") + (CONFIG.parent / "README.md").read_text(
        encoding="utf-8"
    )
    assert "infineon" not in text.lower()
