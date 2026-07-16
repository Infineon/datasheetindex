"""Tests for the shared package-version helper."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from datasheetindex._version import package_version


def test_package_version_matches_installed_metadata():
    assert package_version() == version("datasheetindex")


def test_package_version_falls_back_when_distribution_missing(monkeypatch):
    """A source tree without installed metadata must not raise at server start.

    pyproject sets pythonpath = ["src"], so the package is importable with no
    installed distribution behind it. A visibly wrong version beats an
    ImportError, and both beat a confidently wrong 1.0.0.
    """
    import datasheetindex._version as version_module

    def _raise(_name):
        raise PackageNotFoundError(_name)

    monkeypatch.setattr(version_module, "version", _raise)

    assert package_version() == "0+unknown"
