"""Library discovery for the MCP server: list PDFs under configured folders."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from datasheetindex.index import resolve_default_output_dir

LIBRARY_ROOTS_ENV_VAR = "DATASHEETINDEX_LIBRARY_ROOTS"
CACHE_SCHEMA_VERSION = 1


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


def default_cache_dir() -> Path:
    return Path(resolve_default_output_dir()) / "library"


def cache_file(cache_dir: Path, root: str) -> Path:
    digest = hashlib.sha256(os.path.normcase(root).encode("utf-8")).hexdigest()
    return cache_dir / f"{digest[:16]}.json"


def save_cache(path: Path, result: ScanResult) -> None:
    """Write ``result`` atomically so a concurrent reader never sees a partial file."""
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "root": result.root,
        "listed_at": result.listed_at,
        "skipped_folders": result.skipped_folders,
        "entries": [[e.relpath, e.path, e.mtime, e.size] for e in result.entries],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as tmp:
        tmp_name = tmp.name
        try:
            json.dump(payload, tmp, ensure_ascii=False)
        except BaseException:
            tmp.close()
            os.unlink(tmp_name)
            raise
    try:
        os.replace(tmp_name, path)
    except BaseException:
        os.unlink(tmp_name)
        raise


def load_cache(path: Path, root: str) -> ScanResult | None:
    """Read a cache written by ``save_cache``; ``None`` if absent or unusable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["schema"] != CACHE_SCHEMA_VERSION:
            return None
        if os.path.normcase(data["root"]) != os.path.normcase(root):
            return None
        entries = tuple(
            DocumentEntry(relpath=str(r), path=str(p), mtime=float(m), size=int(z))
            for r, p, m, z in data["entries"]
        )
        return ScanResult(
            root=root,
            listed_at=float(data["listed_at"]),
            entries=entries,
            skipped_folders=int(data["skipped_folders"]),
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None
