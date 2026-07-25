"""Tests for the build-artifact cache primitives, sidecar and validity rule."""

import hashlib
import json
import threading

import pytest

from datasheetindex.core.artifact_cache import (
    SIDECAR_SUFFIX,
    atomic_write_text,
    is_editable_install,
    sha256_file,
    sha256_text,
    sidecar_path,
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
    payload = "5 µA"  # MICRO SIGN, as datasheet units are written
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


def test_atomic_write_text_concurrent_writers(tmp_path):
    """Two concurrent writers to one destination both complete without truncation."""
    target = tmp_path / "out.txt"
    barrier = threading.Barrier(2)
    results = {}

    def writer(writer_id, payload):
        barrier.wait()
        atomic_write_text(target, payload)
        results[writer_id] = payload

    thread1 = threading.Thread(target=writer, args=(1, "payload from thread 1"))
    thread2 = threading.Thread(target=writer, args=(2, "payload from thread 2"))

    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()

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

    monkeypatch.setattr(
        "datasheetindex.core.artifact_cache.Distribution.from_name",
        lambda _name: FakeDistribution(json.dumps({"dir_info": {"editable": False}})),
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
