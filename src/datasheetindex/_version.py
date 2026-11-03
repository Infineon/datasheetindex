"""The installed distribution version, shared by both MCP server surfaces.

Kept out of ``__init__.py`` deliberately: that module imports ``mcp_server``,
which needs this helper, so defining it there would be a circular import.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

_FALLBACK = "0+unknown"


def package_version() -> str:
    """Return the installed ``datasheetindex`` version.

    Both MCP surfaces report this rather than a literal, so a release cannot
    leave one of them advertising a stale version. Falls back to ``0+unknown``
    when the package is importable but has no installed distribution metadata,
    which is the case for a source tree relying on ``pythonpath = ["src"]``.
    """
    try:
        return version("datasheetindex")
    except PackageNotFoundError:
        return _FALLBACK


def pymupdf_version() -> str:
    """Return the version of the PyMuPDF this process imported.

    From the module, not the distribution metadata: ``pymupdf-lite`` installs
    the same ``pymupdf`` module under another distribution name. Imported here
    rather than at module level so importing this helper stays cheap.
    """
    import pymupdf

    return str(pymupdf.VersionBind)
