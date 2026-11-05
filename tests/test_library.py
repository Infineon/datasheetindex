"""Tests for library discovery: root parsing, scanning and matching."""

import json
import os
import re

import pytest

from datasheetindex import library
from datasheetindex.library import (
    DocumentEntry,
    ScanResult,
    cache_file,
    default_cache_dir,
    load_cache,
    match_entries,
    parse_library_roots,
    save_cache,
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


def _result(root, relpaths=("a.pdf",)):
    return ScanResult(
        root=root,
        listed_at=123.5,
        entries=tuple(_entry(r) for r in relpaths),
        skipped_folders=2,
    )


def test_cache_round_trip(tmp_path):
    path = tmp_path / "cache" / "x.json"
    result = _result(str(tmp_path / "lib"), ("a.pdf", os.path.join("d", "b.pdf")))
    save_cache(path, result)
    assert load_cache(path, result.root) == result


def test_cache_round_trip_preserves_non_ascii_and_upper_case_names(tmp_path):
    path = tmp_path / "x.json"
    result = _result(str(tmp_path / "lib"), ("Datenblatt-Gr\u00f6\u00dfe.PDF",))
    save_cache(path, result)
    loaded = load_cache(path, result.root)
    assert loaded is not None
    assert loaded.entries[0].relpath == "Datenblatt-Gr\u00f6\u00dfe.PDF"


def test_load_cache_missing_corrupt_or_old_schema_is_none(tmp_path):
    root = str(tmp_path / "lib")
    assert load_cache(tmp_path / "absent.json", root) is None
    bad = tmp_path / "bad.json"
    bad.write_bytes(b"{not json")
    assert load_cache(bad, root) is None
    old = tmp_path / "old.json"
    old.write_text(
        json.dumps(
            {
                "schema": 0,
                "root": root,
                "listed_at": 1.0,
                "skipped_folders": 0,
                "entries": [],
            }
        ),
        encoding="utf-8",
    )
    assert load_cache(old, root) is None


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"schema": 1},
        {"schema": 1, "root": "R", "listed_at": 1.0, "skipped_folders": 0},
        {
            "schema": 1,
            "root": "R",
            "listed_at": 1.0,
            "skipped_folders": 0,
            "entries": ["notalist"],
        },
        {
            "schema": 1,
            "root": "R",
            "listed_at": "x",
            "skipped_folders": 0,
            "entries": [],
        },
        {
            "schema": 1,
            "root": "R",
            "listed_at": 1.0,
            "skipped_folders": 0,
            "entries": [["a", "b", 1.0]],
        },
    ],
)
def test_load_cache_wrong_shape_is_none(tmp_path, payload):
    path = tmp_path / "x.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_cache(path, "R") is None


def test_load_cache_for_other_root_is_none(tmp_path):
    path = tmp_path / "x.json"
    save_cache(path, _result(str(tmp_path / "one")))
    assert load_cache(path, str(tmp_path / "two")) is None


def test_cache_file_is_stable_and_distinct_per_root(tmp_path):
    a = cache_file(tmp_path, str(tmp_path / "a"))
    assert a == cache_file(tmp_path, str(tmp_path / "a"))
    assert a != cache_file(tmp_path, str(tmp_path / "b"))
    assert a.parent == tmp_path
    assert re.fullmatch(r"[0-9a-f]{16}\.json", a.name)


def test_default_cache_dir_is_under_output_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DATASHEETINDEX_OUTPUT_DIR", str(tmp_path))
    assert default_cache_dir() == tmp_path / "library"


def test_save_cache_is_atomic(tmp_path, monkeypatch):
    path = tmp_path / "x.json"
    root = str(tmp_path / "lib")
    old = _result(root, ("old.pdf",))
    save_cache(path, old)

    def boom(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        save_cache(path, _result(root, ("new.pdf",)))
    monkeypatch.undo()
    assert load_cache(path, root) == old
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]
