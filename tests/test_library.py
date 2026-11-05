"""Tests for library discovery: root parsing, scanning and matching."""

import os

import pytest

from datasheetindex import library
from datasheetindex.library import (
    DocumentEntry,
    match_entries,
    parse_library_roots,
    scan_root,
)


def _touch(path, size=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def _entry(relpath):
    return DocumentEntry(
        relpath=relpath, path=os.path.join("root", relpath), mtime=1.0, size=1
    )


def test_parse_roots_none_and_blank_are_empty():
    assert parse_library_roots(None) == []
    assert parse_library_roots("  " + os.pathsep + " ") == []


def test_parse_roots_strips_quotes_whitespace_and_trailing_separator(tmp_path):
    raw = f' "{tmp_path}{os.sep}" {os.pathsep}{tmp_path / "b"}'
    assert parse_library_roots(raw) == [str(tmp_path), str(tmp_path / "b")]


def test_parse_roots_dedupes_and_makes_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = f"lib{os.pathsep}{tmp_path / 'lib'}"
    assert parse_library_roots(raw) == [str(tmp_path / "lib")]


def test_scan_lists_pdfs_recursively_with_stat_fields(tmp_path):
    name = "Infineon-TLE987x-UM-v01_92-EN.pdf"
    _touch(tmp_path / "TLE" / "TLE987x" / name, size=2048)
    _touch(tmp_path / "notes.txt")
    _touch(tmp_path / "UPPER.PDF")
    result = scan_root(str(tmp_path), clock=lambda: 1000.0)
    rel = sorted(e.relpath for e in result.entries)
    assert rel == [os.path.join("TLE", "TLE987x", name), "UPPER.PDF"]
    um = next(e for e in result.entries if e.relpath.endswith("UM-v01_92-EN.pdf"))
    assert um.path == str(tmp_path / "TLE" / "TLE987x" / name)
    assert um.size == 2048 and um.mtime > 0
    assert result.root == str(tmp_path)
    assert result.listed_at == 1000.0 and result.skipped_folders == 0


def test_scan_does_not_follow_directory_symlinks(tmp_path):
    _touch(tmp_path / "real" / "a.pdf")
    try:
        (tmp_path / "loop").symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks not permitted")
    result = scan_root(str(tmp_path))
    assert [e.relpath for e in result.entries] == [os.path.join("real", "a.pdf")]


def test_scan_skips_and_counts_unreadable_folder(tmp_path, monkeypatch):
    _touch(tmp_path / "ok.pdf")
    _touch(tmp_path / "locked" / "hidden.pdf")
    real_scandir = os.scandir
    locked = str(tmp_path / "locked")

    def fake_scandir(path):
        if os.fspath(path) == locked:
            raise PermissionError(13, "denied", locked)
        return real_scandir(path)

    monkeypatch.setattr(library.os, "scandir", fake_scandir)
    result = scan_root(str(tmp_path))
    assert result.skipped_folders == 1
    assert [e.relpath for e in result.entries] == ["ok.pdf"]


def test_scan_raises_when_root_is_missing(tmp_path):
    with pytest.raises(OSError):
        scan_root(str(tmp_path / "absent"))


def test_match_requires_every_term_case_insensitive():
    um = os.path.join("TLE987x", "Infineon-TLE987x-UM.pdf")
    entries = [
        _entry(um),
        _entry(os.path.join("TLE987x", "Infineon-TLE987x-DS.pdf")),
        _entry(os.path.join("TLE9140", "Infineon-TLE9140-UM.pdf")),
    ]
    assert [e.relpath for e in match_entries(entries, "tle987 um")] == [um]


def test_match_empty_query_returns_all_sorted():
    entries = [_entry("b.pdf"), _entry("A.pdf"), _entry("c.pdf")]
    assert [e.relpath for e in match_entries(entries, "  ")] == [
        "A.pdf",
        "b.pdf",
        "c.pdf",
    ]


def test_match_searches_folders_too():
    entries = [_entry(os.path.join("TLE987x", "4711.pdf")), _entry("other.pdf")]
    got = match_entries(entries, "tle987x")
    assert [e.relpath for e in got] == [os.path.join("TLE987x", "4711.pdf")]


@pytest.mark.parametrize("query", ["TLE987x/um", "tle987x\\UM"])
@pytest.mark.parametrize("sep", ["/", "\\"])
def test_match_normalises_separators_and_case(sep, query):
    rel = "TLE987x" + sep + "UM-guide.pdf"
    assert [e.relpath for e in match_entries([_entry(rel)], query)] == [rel]
