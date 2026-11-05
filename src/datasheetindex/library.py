"""Library discovery for the MCP server: list PDFs under configured folders."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

LIBRARY_ROOTS_ENV_VAR = "DATASHEETINDEX_LIBRARY_ROOTS"


@dataclass(frozen=True)
class DocumentEntry:
    relpath: str
    path: str
    mtime: float
    size: int


@dataclass(frozen=True)
class ScanResult:
    root: str
    listed_at: float
    entries: tuple[DocumentEntry, ...]
    skipped_folders: int


def parse_library_roots(value: str | None) -> list[str]:
    """Split a roots setting into absolute, de-duplicated folder paths."""
    roots: list[str] = []
    seen: set[str] = set()
    for part in (value or "").split(os.pathsep):
        part = part.strip()
        if len(part) >= 2 and part[0] == part[-1] == '"':
            part = part[1:-1].strip()
        if not part:
            continue
        root = os.path.abspath(part)
        key = os.path.normcase(root)
        if key not in seen:
            seen.add(key)
            roots.append(root)
    return roots


def scan_root(root: str, *, clock: Callable[[], float] = time.time) -> ScanResult:
    """List every PDF under ``root``, never following symlinks or junctions.

    Raises ``OSError`` only when ``root`` itself cannot be listed; an unreadable
    subfolder is skipped and counted.
    """
    entries: list[DocumentEntry] = []
    skipped = 0
    stack = [root]
    while stack:
        folder = stack.pop()
        try:
            with os.scandir(folder) as it:
                children = list(it)
        except OSError:
            if folder == root:
                raise
            skipped += 1
            continue
        for child in children:
            try:
                if child.is_symlink() or child.is_junction():
                    continue
                if child.is_dir(follow_symlinks=False):
                    stack.append(child.path)
                elif child.is_file(follow_symlinks=False) and (
                    child.name.lower().endswith(".pdf")
                ):
                    st = child.stat(follow_symlinks=False)
                    entries.append(
                        DocumentEntry(
                            relpath=os.path.relpath(child.path, root),
                            path=child.path,
                            mtime=st.st_mtime,
                            size=st.st_size,
                        )
                    )
            except OSError:
                continue
    return ScanResult(
        root=root,
        listed_at=clock(),
        entries=tuple(entries),
        skipped_folders=skipped,
    )


def _fold(text: str) -> str:
    return text.replace("\\", "/").casefold()


def match_entries(entries: Iterable[DocumentEntry], query: str) -> list[DocumentEntry]:
    """Entries whose relpath contains every query term, sorted by relpath."""
    terms = [_fold(t) for t in query.split()]
    hits = [e for e in entries if all(t in _fold(e.relpath) for t in terms)]
    return sorted(hits, key=lambda e: e.relpath.casefold())
