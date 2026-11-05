"""Tests for library discovery: root parsing, scanning, matching and serving."""

import json
import logging
import os
import re
import threading
import time
from datetime import datetime

import pytest

from datasheetindex import library
from datasheetindex.library import (
    MAX_LIMIT,
    DocumentEntry,
    DocumentLibrary,
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


class _UnstattableEntry:
    """A scandir entry whose ``stat`` fails, as a file deleted mid-scan does."""

    def __init__(self, entry):
        self._entry = entry
        self.name, self.path = entry.name, entry.path

    def __getattr__(self, attr):
        return getattr(self._entry, attr)

    def stat(self, *, follow_symlinks=True):
        raise PermissionError(13, "denied", self.path)


def test_scan_skips_and_counts_pdf_that_cannot_be_read(tmp_path, monkeypatch):
    _touch(tmp_path / "ok.pdf")
    _touch(tmp_path / "gone.pdf")
    real_scandir = os.scandir

    class _Listing:
        def __init__(self, path):
            self._it = real_scandir(path)

        def __enter__(self):
            return [
                _UnstattableEntry(e) if e.name == "gone.pdf" else e
                for e in self._it.__enter__()
            ]

        def __exit__(self, *exc):
            return self._it.__exit__(*exc)

    monkeypatch.setattr(library.os, "scandir", _Listing)
    result = scan_root(str(tmp_path))
    assert [e.relpath for e in result.entries] == ["ok.pdf"]
    assert result.skipped_files == 1 and result.skipped_folders == 0


def _make_unencodable_pdf(folder):
    """Create ``TLE\xff-DS.pdf``, a name that is not valid UTF-8, or skip."""
    name = b"TLE\xff-DS.pdf"
    try:
        with open(os.path.join(os.fsencode(folder), name), "wb") as f:
            f.write(b"x")
    except (OSError, ValueError):
        pytest.skip("this filesystem cannot hold a non-UTF-8 file name")
    if os.fsencode(os.fsdecode(name)) != name:
        pytest.skip("file names do not round-trip through surrogateescape")


@pytest.mark.skipif(os.name != "posix", reason="POSIX byte file names only")
def test_scan_skips_and_counts_name_not_encodable_as_utf8(tmp_path):
    _touch(tmp_path / "ok.pdf")
    _make_unencodable_pdf(tmp_path)
    result = scan_root(str(tmp_path))
    assert [e.relpath for e in result.entries] == ["ok.pdf"]
    assert result.skipped_files == 1
    # The listing it produces must survive the cache and the JSON response.
    save_cache(tmp_path / "c.json", result)
    assert load_cache(tmp_path / "c.json", result.root) == result


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
        {"schema": library.CACHE_SCHEMA_VERSION},
        {
            "schema": library.CACHE_SCHEMA_VERSION,
            "root": "R",
            "listed_at": 1.0,
            "skipped_folders": 0,
            "skipped_files": 0,
        },
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


def _cache_text(root, *, listed_at="1.0", size="1", mtime="1.0"):
    """A cache file's text with raw JSON literals spliced into its numbers."""
    return (
        f'{{"schema": {library.CACHE_SCHEMA_VERSION}, "root": {json.dumps(root)}, '
        f'"listed_at": {listed_at}, "skipped_folders": 0, "skipped_files": 0, '
        f'"entries": [["a.pdf", "a.pdf", {mtime}, {size}]]}}'
    )


@pytest.mark.parametrize(
    "text",
    [
        _cache_text("R", size="Infinity"),
        _cache_text("R", size="1e400"),
        _cache_text("R", listed_at="1e400"),
        _cache_text("R", listed_at="Infinity"),
        _cache_text("R", listed_at="NaN"),
        "[" * 100_000 + "]" * 100_000,
    ],
    ids=["size-inf", "size-1e400", "listed-1e400", "listed-inf", "listed-nan", "deep"],
)
def test_load_cache_damaged_numbers_or_nesting_is_none(tmp_path, text):
    path = tmp_path / "x.json"
    path.write_text(text, encoding="utf-8")
    assert load_cache(path, "R") is None


def test_load_cache_unrepresentable_mtime_is_kept(tmp_path):
    # One file's date is unknown; the response renders it as None (_local_time).
    path = tmp_path / "x.json"
    path.write_text(_cache_text("R", mtime="1e400"), encoding="utf-8")
    loaded = load_cache(path, "R")
    assert loaded is not None and loaded.entries[0].mtime == float("inf")


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


# --- DocumentLibrary -------------------------------------------------------

NOW = 1_000_000.0
DAY = 86400.0


def _listing(root, relpaths, *, listed_at=NOW, size=1):
    return ScanResult(
        root=root,
        listed_at=listed_at,
        entries=tuple(
            DocumentEntry(relpath=r, path=os.path.join(root, r), mtime=NOW, size=size)
            for r in relpaths
        ),
        skipped_folders=0,
    )


def _lists(*relpaths):
    """A scanner outcome: list ``relpaths`` under whichever root is scanned."""
    return lambda root: _listing(root, relpaths)


class FakeScanner:
    """Canned outcomes in call order (the last repeats); optionally gated."""

    def __init__(self, *outcomes, gated=False):
        self.outcomes = list(outcomes)
        self.gated = gated
        self.gate = threading.Event()
        self.calls = []

    def __call__(self, root):
        self.calls.append(root)
        if self.gated:
            assert self.gate.wait(5), "test never released the scanner gate"
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome(root)


@pytest.fixture
def cache_dir(tmp_path):
    return tmp_path / "cache"


@pytest.fixture
def make_library(cache_dir):
    made = []

    def make(roots, scanner, **kwargs):
        kwargs.setdefault("clock", lambda: NOW)
        kwargs.setdefault("wait_seconds", 5.0)
        lib = DocumentLibrary(list(roots), cache_dir, scanner=scanner, **kwargs)
        made.append((lib, scanner))
        return lib

    yield make
    for lib, scanner in made:
        if isinstance(scanner, FakeScanner):
            scanner.gate.set()
        assert lib.wait_for_scans(timeout=5)


def _seed(cache_dir, root, relpaths, *, age):
    save_cache(
        cache_file(cache_dir, root), _listing(root, relpaths, listed_at=NOW - age)
    )


def _names(response):
    return [os.path.basename(d["path"]) for d in response["documents"]]


def test_find_serves_fresh_cache_without_scanning(tmp_path, cache_dir, make_library):
    root = str(tmp_path / "lib")
    _seed(cache_dir, root, ["a.pdf", "b.pdf"], age=60)
    scanner = FakeScanner(_lists("new.pdf"))
    got = make_library([root], scanner).find()
    assert scanner.calls == []
    assert _names(got) == ["a.pdf", "b.pdf"]
    [status] = got["roots"]
    assert status["root"] == root and status["status"] == "ok"
    assert status["documents"] == 2 and "error" not in status


def test_warm_scans_missing_and_stale_roots_in_background(
    tmp_path, cache_dir, make_library
):
    missing, stale, fresh = (str(tmp_path / n) for n in ("missing", "stale", "fresh"))
    _seed(cache_dir, stale, ["old.pdf"], age=DAY + 1)
    _seed(cache_dir, fresh, ["kept.pdf"], age=60)
    scanner = FakeScanner(_lists("new.pdf"), gated=True)
    lib = make_library([missing, stale, fresh], scanner)
    lib.warm()  # returns although both scans are blocked on the gate
    assert not scanner.gate.is_set()
    scanner.gate.set()
    assert lib.wait_for_scans(timeout=5)
    assert sorted(scanner.calls) == sorted([missing, stale])
    got = lib.find()
    assert [r["status"] for r in got["roots"]] == ["ok", "ok", "ok"]
    assert _names(got) == ["new.pdf", "new.pdf", "kept.pdf"]


def test_stale_list_is_served_at_once_and_refreshed(tmp_path, cache_dir, make_library):
    root = str(tmp_path / "lib")
    _seed(cache_dir, root, ["old.pdf"], age=DAY + 1)
    scanner = FakeScanner(_lists("new.pdf"), gated=True)
    lib = make_library([root], scanner)
    start = time.monotonic()
    got = lib.find()
    assert time.monotonic() - start < 1.0
    assert _names(got) == ["old.pdf"]
    assert got["roots"][0]["status"] == "refreshing"
    assert "scan_elapsed_seconds" not in got["roots"][0]
    scanner.gate.set()
    assert lib.wait_for_scans(timeout=5)
    got = lib.find()
    assert _names(got) == ["new.pdf"]
    assert got["roots"][0]["status"] == "ok"
    assert scanner.calls == [root]


def test_refresh_true_waits_for_the_new_list(tmp_path, cache_dir, make_library):
    root = str(tmp_path / "lib")
    _seed(cache_dir, root, ["old.pdf"], age=60)
    scanner = FakeScanner(_lists("new.pdf"), gated=True)
    lib = make_library([root], scanner)
    timer = threading.Timer(0.05, scanner.gate.set)
    timer.start()
    try:
        got = lib.find(refresh=True)
    finally:
        timer.join()
    assert _names(got) == ["new.pdf"]
    assert got["roots"][0]["status"] == "ok"


def test_call_during_running_scan_does_not_start_another(
    tmp_path, cache_dir, make_library
):
    root = str(tmp_path / "lib")
    scanner = FakeScanner(_lists("a.pdf"), gated=True)
    lib = make_library([root], scanner, wait_seconds=0.05)
    lib.warm()
    assert lib.find()["roots"][0]["status"] == "scanning"
    assert lib.find(refresh=True)["roots"][0]["status"] == "scanning"
    scanner.gate.set()
    assert lib.wait_for_scans(timeout=5)
    assert _names(lib.find()) == ["a.pdf"]
    assert scanner.calls == [root]


def test_wait_is_bounded_and_reports_scanning(tmp_path, cache_dir, make_library):
    root = str(tmp_path / "lib")
    now = [NOW]
    scanner = FakeScanner(_lists("a.pdf"), gated=True)
    lib = make_library([root], scanner, clock=lambda: now[0], wait_seconds=0.2)
    lib.warm()
    now[0] += 7.0
    start = time.monotonic()
    got = lib.find()
    elapsed = time.monotonic() - start
    assert 0.15 <= elapsed < 1.0
    [status] = got["roots"]
    assert status["status"] == "scanning"
    assert status["scan_elapsed_seconds"] == 7
    assert status["listed_at"] is None and status["documents"] == 0
    assert got["documents"] == [] and got["total_matches"] == 0


def test_unreachable_root_without_cache_is_unavailable_not_error(
    tmp_path, cache_dir, make_library
):
    root = str(tmp_path / "lib")
    scanner = FakeScanner(OSError("share offline"))
    got = make_library([root], scanner).find()
    [status] = got["roots"]
    assert status["status"] == "unavailable"
    assert status["error"].startswith("OSError")
    assert "share offline" in status["error"]
    assert status["listed_at"] is None
    assert got["documents"] == []


def test_any_scan_exception_is_recorded_not_raised(tmp_path, cache_dir, make_library):
    root = str(tmp_path / "lib")
    scanner = FakeScanner(ValueError("bad entry"))
    status = make_library([root], scanner).find()["roots"][0]
    assert status["status"] == "unavailable"
    assert status["error"] == "ValueError: bad entry"


def test_unreachable_root_with_cache_serves_old_list(tmp_path, cache_dir, make_library):
    root = str(tmp_path / "lib")
    _seed(cache_dir, root, ["old.pdf"], age=60)
    scanner = FakeScanner(OSError("share offline"))
    lib = make_library([root], scanner)
    got = lib.find(refresh=True)
    assert _names(got) == ["old.pdf"]
    [status] = got["roots"]
    assert status["status"] == "ok"
    assert status["error"].startswith("OSError")
    assert status["listed_at"] is not None
    # The fresh list is still served without another attempt; the error stays.
    got = lib.find()
    assert _names(got) == ["old.pdf"]
    assert got["roots"][0]["error"].startswith("OSError")
    assert scanner.calls == [root]


def test_failed_scan_without_cache_is_retried_on_next_call(
    tmp_path, cache_dir, make_library
):
    root = str(tmp_path / "lib")
    scanner = FakeScanner(OSError("share offline"), _lists("a.pdf"))
    lib = make_library([root], scanner)
    assert lib.find()["roots"][0]["status"] == "unavailable"
    got = lib.find()
    assert got["roots"][0]["status"] == "ok"
    assert "error" not in got["roots"][0]
    assert _names(got) == ["a.pdf"]
    assert scanner.calls == [root, root]


def test_overlapping_roots_list_each_pdf_once(tmp_path, cache_dir, make_library):
    lib_dir = tmp_path / "lib"
    _touch(lib_dir / "Datasheet" / "a.pdf")
    _touch(lib_dir / "b.pdf")
    outer, inner = str(lib_dir), str(lib_dir / "Datasheet")
    got = make_library([outer, inner], scan_root).find()
    assert [d["path"] for d in got["documents"]] == [
        str(lib_dir / "b.pdf"),
        str(lib_dir / "Datasheet" / "a.pdf"),
    ]
    assert got["total_matches"] == 2
    assert [r["documents"] for r in got["roots"]] == [2, 1]


def test_limit_truncates_and_reports_total(tmp_path, cache_dir, make_library):
    root = str(tmp_path / "lib")
    _seed(cache_dir, root, ["tle-a.pdf", "tle-b.pdf", "tle-c.pdf", "x.pdf"], age=60)
    lib = make_library([root], FakeScanner(_lists()))
    got = lib.find("tle", limit=2)
    assert _names(got) == ["tle-a.pdf", "tle-b.pdf"]
    assert got["total_matches"] == 3 and got["truncated"] is True
    assert len(lib.find(limit=0)["documents"]) == 1
    assert lib.find(limit=MAX_LIMIT + 1)["truncated"] is False


def test_result_fields_are_formatted(tmp_path, cache_dir, make_library):
    root = str(tmp_path / "lib")
    save_cache(cache_file(cache_dir, root), _listing(root, ["a.pdf"], size=1025))
    lib = make_library([root], FakeScanner(_lists()))
    got = lib.find()
    json.dumps(got)
    [doc] = got["documents"]
    assert doc == {
        "path": os.path.join(root, "a.pdf"),
        "modified": datetime.fromtimestamp(NOW).strftime("%Y-%m-%d"),
        "size_kb": 2,
    }
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", doc["modified"])
    listed_at = datetime.fromisoformat(got["roots"][0]["listed_at"])
    assert listed_at.tzinfo is not None
    assert listed_at.timestamp() == NOW
    assert got["roots"][0]["skipped_folders"] == 0
    assert lib.roots == [root]


def test_successful_scan_is_persisted_for_the_next_process(
    tmp_path, cache_dir, make_library
):
    root = str(tmp_path / "lib")
    assert _names(make_library([root], FakeScanner(_lists("a.pdf"))).find()) == [
        "a.pdf"
    ]
    second = FakeScanner(OSError("must not scan"))
    got = make_library([root], second).find()
    assert _names(got) == ["a.pdf"]
    assert second.calls == []


def test_cache_save_failure_is_logged_not_raised(
    tmp_path, cache_dir, make_library, monkeypatch, caplog
):
    def fail(path, result):
        raise OSError("disk full")

    monkeypatch.setattr(library, "save_cache", fail)
    root = str(tmp_path / "lib")
    with caplog.at_level(logging.WARNING, logger="datasheetindex.library"):
        got = make_library([root], FakeScanner(_lists("a.pdf"))).find()
    assert _names(got) == ["a.pdf"]
    assert got["roots"][0]["status"] == "ok"
    assert "disk full" in caplog.text


@pytest.mark.parametrize(
    "text",
    [
        _cache_text("ROOT", size="Infinity"),
        _cache_text("ROOT", listed_at="NaN"),
        "[" * 100_000 + "]" * 100_000,
    ],
    ids=["size-inf", "listed-nan", "deep"],
)
def test_warm_on_a_damaged_cache_rescans_instead_of_raising(
    tmp_path, cache_dir, make_library, text
):
    root = str(tmp_path / "lib")
    path = cache_file(cache_dir, root)
    path.parent.mkdir(parents=True)
    path.write_text(text.replace('"ROOT"', json.dumps(root)), encoding="utf-8")
    scanner = FakeScanner(_lists("new.pdf"))
    lib = make_library([root], scanner)
    lib.warm()
    assert lib.wait_for_scans(timeout=5)
    assert scanner.calls == [root]
    assert _names(lib.find()) == ["new.pdf"]


def test_list_from_the_future_is_stale(tmp_path, cache_dir, make_library):
    # Another host sharing the cache folder, with its clock ahead of ours.
    root = str(tmp_path / "lib")
    _seed(cache_dir, root, ["old.pdf"], age=-3600)
    scanner = FakeScanner(_lists("new.pdf"))
    lib = make_library([root], scanner)
    lib.warm()
    assert lib.wait_for_scans(timeout=5)
    assert scanner.calls == [root]
    assert _names(lib.find()) == ["new.pdf"]


def test_skipped_files_are_reported_per_root(tmp_path, cache_dir, make_library):
    root = str(tmp_path / "lib")
    listing = _listing(root, ["a.pdf"])
    save_cache(
        cache_file(cache_dir, root),
        ScanResult(listing.root, listing.listed_at, listing.entries, 0, 3),
    )
    got = make_library([root], FakeScanner(_lists())).find()
    assert got["roots"][0]["skipped_files"] == 3


@pytest.mark.skipif(os.name != "posix", reason="POSIX byte file names only")
def test_unencodable_name_is_left_out_of_the_response(
    tmp_path, cache_dir, make_library
):
    lib_dir = tmp_path / "lib"
    _touch(lib_dir / "TLE-UM.pdf")
    _make_unencodable_pdf(lib_dir)
    lib = make_library([str(lib_dir)], scan_root)
    got = lib.find()
    assert _names(got) == ["TLE-UM.pdf"]
    assert got["roots"][0]["skipped_files"] == 1
    json.dumps(got, ensure_ascii=False).encode("utf-8")
    assert load_cache(cache_file(cache_dir, str(lib_dir)), str(lib_dir)) is not None


def test_tool_def_description_names_roots_and_schema_caps_limit(tmp_path):
    from datasheetindex.library import create_find_documents_def

    lib = DocumentLibrary(["/a", "/b"], tmp_path)
    d = create_find_documents_def(lib)
    assert d.name == "find_documents"
    assert "Library roots: /a; /b." in d.description
    props = d.input_schema["properties"]
    assert props["limit"]["maximum"] == MAX_LIMIT
    assert props["limit"]["default"] == 50
    assert props["query"]["default"] == ""
    assert props["refresh"]["default"] is False
