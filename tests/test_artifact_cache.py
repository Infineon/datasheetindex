"""Tests for the build-artifact cache primitives, sidecar and validity rule."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import replace

import pytest

from datasheetindex.core.artifact_cache import (
    SIDECAR_SUFFIX,
    ArtifactRecord,
    atomic_write_text,
    is_editable_install,
    read_sidecar,
    remove_sidecar,
    reuse_blocker,
    sha256_file,
    sha256_text,
    sidecar_path,
    write_sidecar,
)


class FakeDistribution:
    """Stand-in for importlib.metadata.Distribution."""

    def __init__(self, direct_url_payload):
        self._payload = direct_url_payload

    def read_text(self, name):
        assert name == "direct_url.json"
        return self._payload


def test_sha256_file_matches_hashlib(tmp_path):
    target = tmp_path / "payload.bin"
    target.write_bytes(b"datasheet bytes")

    assert sha256_file(target) == hashlib.sha256(b"datasheet bytes").hexdigest()


def test_sha256_file_reads_in_chunks(tmp_path):
    """A file larger than one chunk must hash the same as one shot."""
    payload = b"x" * (3 * 1024 * 1024 + 7)
    target = tmp_path / "big.bin"
    target.write_bytes(payload)

    assert sha256_file(target) == hashlib.sha256(payload).hexdigest()


def test_sha256_text_is_utf8():
    """A multi-byte codepoint, so a single-byte codec would give a different digest."""
    payload = "5 \u00b5A"  # U+00B5 MICRO SIGN, as datasheet units are written
    expected = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    assert sha256_text(payload) == expected
    assert sha256_text(payload) != hashlib.sha256(payload.encode("latin-1")).hexdigest()


def test_atomic_write_text_creates_file(tmp_path):
    target = tmp_path / "out.txt"

    atomic_write_text(target, "--- PAGE 1 ---\nbody")

    assert target.read_text(encoding="utf-8") == "--- PAGE 1 ---\nbody"


def test_atomic_write_text_creates_missing_parents(tmp_path):
    target = tmp_path / "nested" / "out.txt"

    atomic_write_text(target, "content")

    assert target.read_text(encoding="utf-8") == "content"


def test_atomic_write_text_leaves_previous_content_on_failure(tmp_path, monkeypatch):
    """The whole reason for the temp-then-replace: no truncated file."""
    target = tmp_path / "out.txt"
    atomic_write_text(target, "generation one")

    def boom(_src, _dst):
        raise OSError("replace failed")

    monkeypatch.setattr("datasheetindex.core.artifact_cache.os.replace", boom)

    with pytest.raises(OSError):
        atomic_write_text(target, "generation two")

    assert target.read_text(encoding="utf-8") == "generation one"


def test_atomic_write_text_removes_its_temp_file_on_failure(tmp_path, monkeypatch):
    """No residue in a directory consumers enumerate."""
    target = tmp_path / "out.txt"

    def boom(_src, _dst):
        raise OSError("replace failed")

    monkeypatch.setattr("datasheetindex.core.artifact_cache.os.replace", boom)

    with pytest.raises(OSError):
        atomic_write_text(target, "content")

    assert list(tmp_path.iterdir()) == []


def test_atomic_write_text_overwrites_existing_file(tmp_path):
    """Happy-path overwrite of an existing destination."""
    target = tmp_path / "out.txt"
    atomic_write_text(target, "first write")
    atomic_write_text(target, "second write")

    assert target.read_text(encoding="utf-8") == "second write"
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_write_text_uses_a_unique_temp_path_per_write(tmp_path, monkeypatch):
    """Two writers must never derive the same temp name.

    A shared temp name lets one writer's partial content be replaced into
    place by another, and leaves the loser's os.replace with nothing to move.
    Asserted without threads so it fails deterministically, which a race
    cannot promise.
    """
    target = tmp_path / "out.txt"
    seen: list[str] = []
    real_replace = os.replace

    def recording_replace(src, dst):
        seen.append(str(src))
        real_replace(src, dst)

    patch_target = "datasheetindex.core.artifact_cache.os.replace"
    monkeypatch.setattr(patch_target, recording_replace)

    atomic_write_text(target, "first")
    atomic_write_text(target, "second")

    assert len(seen) == 2
    assert seen[0] != seen[1], "both writes used the same temp path"
    assert target.read_text(encoding="utf-8") == "second"


def test_atomic_write_text_concurrent_writers(tmp_path):
    """Two concurrent writers both complete, and neither corrupts the result."""
    target = tmp_path / "out.txt"
    barrier = threading.Barrier(2)
    results = {}
    errors: list[BaseException] = []

    def writer(writer_id, payload):
        try:
            barrier.wait()
            atomic_write_text(target, payload)
            results[writer_id] = payload
        except BaseException as exc:
            errors.append(exc)

    thread1 = threading.Thread(target=writer, args=(1, "payload from thread 1"))
    thread2 = threading.Thread(target=writer, args=(2, "payload from thread 2"))

    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()

    assert errors == [], f"a writer thread raised: {errors}"
    final_content = target.read_text(encoding="utf-8")
    assert final_content in (results[1], results[2])
    assert list(tmp_path.iterdir()) == [target]


def test_is_editable_install_reads_direct_url_metadata(monkeypatch):
    """The probe must state the rule, not observe this checkout."""
    monkeypatch.setattr(
        "datasheetindex.core.artifact_cache.Distribution.from_name",
        lambda _name: FakeDistribution(json.dumps({"dir_info": {"editable": True}})),
    )
    assert is_editable_install() is True

    # A non-editable directory install still carries dir_info -- editable is
    # absent/false, but the install came from a directory just the same, so
    # this must also disable reuse.
    monkeypatch.setattr(
        "datasheetindex.core.artifact_cache.Distribution.from_name",
        lambda _name: FakeDistribution(json.dumps({"dir_info": {"editable": False}})),
    )
    assert is_editable_install() is True


def test_is_editable_install_is_true_for_a_non_editable_directory_install(
    monkeypatch,
):
    """``pip install .`` writes ``dir_info: {}`` with no ``editable`` key.

    A contributor iterating with a plain (non-editable) directory install and
    no version bump between installs must not be served pre-edit artifacts --
    the same failure the editable check exists to prevent, one workflow over.
    """
    monkeypatch.setattr(
        "datasheetindex.core.artifact_cache.Distribution.from_name",
        lambda _name: FakeDistribution(json.dumps({"dir_info": {}})),
    )

    assert is_editable_install() is True


def test_is_editable_install_is_false_for_a_local_archive_install(monkeypatch):
    """``pip install ./dist/foo.whl`` writes a ``url`` with no ``dir_info`` key.

    That install is immutable, exactly like an index-installed wheel, so it
    must stay reusable.
    """
    monkeypatch.setattr(
        "datasheetindex.core.artifact_cache.Distribution.from_name",
        lambda _name: FakeDistribution(
            json.dumps({"url": "file:///tmp/dist/foo-1.0-py3-none-any.whl"})
        ),
    )

    assert is_editable_install() is False


def test_is_editable_install_treats_a_wheel_without_direct_url_as_not_editable(
    monkeypatch,
):
    """A PyPI/Artifactory wheel has no direct_url.json at all.

    That is the immutable case where version equality genuinely suffices, so it
    must be reusable -- this is the whole point of the feature in production.
    """
    monkeypatch.setattr(
        "datasheetindex.core.artifact_cache.Distribution.from_name",
        lambda _name: FakeDistribution(None),
    )

    assert is_editable_install() is False


def test_is_editable_install_is_true_without_distribution_metadata(monkeypatch):
    """A source tree on ``pythonpath`` has no distribution at all.

    That is the development case, so it fails safe towards 'editable'.
    """

    def no_distribution(_name):
        raise ValueError("no distribution metadata")

    monkeypatch.setattr(
        "datasheetindex.core.artifact_cache.Distribution.from_name", no_distribution
    )

    assert is_editable_install() is True


def test_is_editable_install_is_true_on_corrupt_metadata(monkeypatch):
    monkeypatch.setattr(
        "datasheetindex.core.artifact_cache.Distribution.from_name",
        lambda _name: FakeDistribution("{not json"),
    )

    assert is_editable_install() is True


def test_sidecar_path_sits_beside_the_deliverables(tmp_path):
    assert SIDECAR_SUFFIX == ".build.json"
    assert sidecar_path(tmp_path, "ds") == tmp_path / "ds.build.json"


def make_record(**overrides) -> ArtifactRecord:
    """A complete record; override one field per invalidation test.

    Overrides are applied with dataclasses.replace rather than coerced, so a
    wrongly-typed override surfaces instead of being silently converted --
    bool("False") is True, which would invert the flag that governs whether a
    cached artifact may be reused at all.
    """
    base = ArtifactRecord(
        source_sha256="a" * 64,
        source_size=1024,
        build_options={
            "output_dir": "/tmp/out",
            "output_stem": None,
            "include_summaries": False,
            "model": None,
        },
        datasheetindex_version="0.24.0",
        json_name="ds.json",
        json_sha256="b" * 64,
        text_name="ds.txt",
        text_sha256="c" * 64,
        toc_quality={
            "score": 0.62,
            "entry_count": 2,
            "max_depth": 1,
            "page_coverage": 1.0,
            "recommend_summaries": True,
            "details": "2 entries",
        },
    )
    return replace(base, **overrides) if overrides else base


def test_record_round_trips_through_json(tmp_path):
    record = make_record(
        llm_enrichment_incomplete=True,
        llm_enrichment_notes=("toc_fallback_raised",),
    )
    path = tmp_path / f"ds{SIDECAR_SUFFIX}"

    write_sidecar(path, record)

    assert read_sidecar(path) == record


def test_record_json_nests_artifacts_and_keeps_quality_details(tmp_path):
    path = tmp_path / f"ds{SIDECAR_SUFFIX}"
    write_sidecar(path, make_record())

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["artifacts"]["json"]["name"] == "ds.json"
    assert payload["artifacts"]["text"]["sha256"] == "c" * 64
    assert payload["toc_quality"]["details"] == "2 entries"
    assert payload["llm_enrichment_notes"] == []


def test_read_sidecar_returns_none_when_missing(tmp_path):
    assert read_sidecar(tmp_path / f"absent{SIDECAR_SUFFIX}") is None


def test_read_sidecar_returns_none_on_corrupt_json(tmp_path):
    path = tmp_path / f"ds{SIDECAR_SUFFIX}"
    path.write_text("{not json", encoding="utf-8")

    assert read_sidecar(path) is None


def test_read_sidecar_warns_on_bad_shape(tmp_path, caplog):
    """A parseable file with the wrong shape means to_dict/from_dict diverged.

    That is a bug, not a routine miss, so it is logged louder.
    """
    path = tmp_path / f"ds{SIDECAR_SUFFIX}"
    path.write_text(json.dumps({"source_sha256": "a" * 64}), encoding="utf-8")

    with caplog.at_level("WARNING"):
        assert read_sidecar(path) is None

    assert any(entry.levelname == "WARNING" for entry in caplog.records)


def test_remove_sidecar_is_silent_when_absent(tmp_path):
    remove_sidecar(tmp_path / f"absent{SIDECAR_SUFFIX}")
    remove_sidecar(tmp_path / "no-such-dir" / f"ds{SIDECAR_SUFFIX}")


def test_remove_sidecar_deletes(tmp_path):
    path = tmp_path / f"ds{SIDECAR_SUFFIX}"
    write_sidecar(path, make_record())

    remove_sidecar(path)

    assert not path.exists()


@pytest.fixture
def source_file(tmp_path):
    """A stand-in source; this module never needs a real PDF."""
    path = tmp_path / "ds.pdf"
    path.write_bytes(b"%PDF-1.7 fake source bytes")
    return path


def matching_record(source_file, **overrides) -> ArtifactRecord:
    return make_record(
        source_sha256=sha256_file(source_file),
        source_size=source_file.stat().st_size,
        **overrides,
    )


def test_reuse_blocker_returns_none_when_everything_matches(source_file):
    record = matching_record(source_file)

    blocker = reuse_blocker(
        record,
        source_path=source_file,
        build_options=record.build_options,
        running_version="0.24.0",
    )

    assert blocker is None


def test_reuse_blocker_rejects_a_different_version(source_file):
    record = matching_record(source_file)

    blocker = reuse_blocker(
        record,
        source_path=source_file,
        build_options=record.build_options,
        running_version="0.25.0",
    )

    assert blocker == "version_changed"


def test_reuse_blocker_rejects_changed_source_bytes(source_file):
    record = matching_record(source_file)
    source_file.write_bytes(b"%PDF-1.7 different bytes here")

    blocker = reuse_blocker(
        record,
        source_path=source_file,
        build_options=record.build_options,
        running_version="0.24.0",
    )

    assert blocker in {"source_size_changed", "source_content_changed"}


def test_reuse_blocker_detects_a_same_size_source_edit(source_file):
    """The size pre-check must not be the only source check."""
    record = matching_record(source_file)
    original = source_file.read_bytes()
    source_file.write_bytes(b"X" + original[1:])

    blocker = reuse_blocker(
        record,
        source_path=source_file,
        build_options=record.build_options,
        running_version="0.24.0",
    )

    assert blocker == "source_content_changed"


def test_reuse_blocker_rejects_a_missing_source(source_file):
    record = matching_record(source_file)
    source_file.unlink()

    blocker = reuse_blocker(
        record,
        source_path=source_file,
        build_options=record.build_options,
        running_version="0.24.0",
    )

    assert blocker == "source_missing"


def test_reuse_blocker_rejects_changed_build_options(source_file):
    record = matching_record(source_file)
    changed = dict(record.build_options)
    changed["include_summaries"] = True

    blocker = reuse_blocker(
        record,
        source_path=source_file,
        build_options=changed,
        running_version="0.24.0",
    )

    assert blocker == "build_options_changed"


def test_reuse_blocker_rejects_incomplete_enrichment(source_file):
    """A degraded artifact must never be pinned in place."""
    record = matching_record(
        source_file,
        llm_enrichment_incomplete=True,
        llm_enrichment_notes=("toc_fallback_raised",),
    )

    blocker = reuse_blocker(
        record,
        source_path=source_file,
        build_options=record.build_options,
        running_version="0.24.0",
    )

    assert blocker == "llm_enrichment_incomplete"


def test_reuse_blocker_checks_the_cheap_fields_before_hashing(source_file, monkeypatch):
    """A version mismatch must not pay for a 2.6 MB hash."""

    def unexpected(_path):
        raise AssertionError("hashed the source despite a version mismatch")

    monkeypatch.setattr("datasheetindex.core.artifact_cache.sha256_file", unexpected)
    record = matching_record(source_file)

    blocker = reuse_blocker(
        record,
        source_path=source_file,
        build_options=record.build_options,
        running_version="9.9.9",
    )

    assert blocker == "version_changed"
