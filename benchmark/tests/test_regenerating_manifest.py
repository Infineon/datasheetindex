"""Manifest coverage is enforced, not asserted.

A name grep cannot build this manifest: producers construct output names at
runtime (fault_injection_multimodel writes
f"fault_injection_{model.replace('.','_')}.json"), and a grep cannot tell a
producer from a consumer. So the manifest is written by hand and this test
is what keeps it honest.
"""

from __future__ import annotations

from pathlib import Path

from chamberbench.claimsio import archive_dir

MANIFEST = Path(__file__).resolve().parents[1] / "docs" / "regenerating.md"


#: The rendered figures under archive/figures/ are NOT shipped. They are derived
#: from the tracked archive/*.json by the commands in the manifest's Figures
#: section, and matplotlib does not render them bit-for-bit reproducibly across
#: versions, so a committed PNG would drift and invite false "this does not
#: match" conclusions. They are excluded here so that this suite measures what a
#: fresh clone actually has, whether or not a working tree happens to hold a
#: locally regenerated copy.
SHIPPED_SUFFIXES = (".json", ".jsonl", ".md")


def _archive_files() -> list[Path]:
    """Every shipped artifact under archive/, excluding the directory's own README.

    Locally regenerated figures are skipped -- see SHIPPED_SUFFIXES.
    """
    return sorted(
        p
        for p in archive_dir().rglob("*")
        if p.is_file() and p.name != "README.md" and p.suffix in SHIPPED_SUFFIXES
    )


def test_expected_artifact_count():
    """37: 33 json, 3 jsonl, and classifier_agreement.md. Figures are not shipped."""
    assert len(_archive_files()) == 37


def test_every_artifact_has_a_manifest_row():
    text = MANIFEST.read_text(encoding="utf-8")
    missing = [p.name for p in _archive_files() if p.name not in text]
    assert not missing, "no manifest row for: " + ", ".join(sorted(missing))


def test_manifest_names_no_artifact_that_does_not_exist():
    """Catches a row left behind after an artifact is renamed."""
    text = MANIFEST.read_text(encoding="utf-8")
    names = {p.name for p in _archive_files()}
    for line in text.splitlines():
        if not line.startswith("| `"):
            continue
        cited = line.split("`")[1]
        # .png is deliberately absent from this tuple: the Figures rows document
        # how to regenerate artifacts the archive does not ship, so citing one is
        # not a stale row. See SHIPPED_SUFFIXES.
        if cited.endswith(SHIPPED_SUFFIXES):
            assert cited in names, cited


def test_unrecovered_provenance_is_declared_not_omitted():
    """variance_qwen_no_think.json's exact invocation was not recovered.
    The manifest must say so rather than leave the row looking complete."""
    text = MANIFEST.read_text(encoding="utf-8")
    assert "variance_qwen_no_think.json" in text
    assert "not recovered" in text.lower()
