"""Library discovery for the MCP server: list PDFs under configured folders."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from datasheetindex.index import resolve_default_output_dir

LIBRARY_ROOTS_ENV_VAR = "DATASHEETINDEX_LIBRARY_ROOTS"
CACHE_SCHEMA_VERSION = 1
CACHE_TTL_SECONDS = 86400.0
SCAN_WAIT_SECONDS = 30.0
DEFAULT_LIMIT = 50
MAX_LIMIT = 500
SCAN_THREAD_NAME = "datasheetindex-library-scan"

logger = logging.getLogger(__name__)


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


def _local_time(ts: float) -> datetime | None:
    # A share can report an mtime the platform cannot represent; that file's
    # date is unknown, which must not fail the whole listing.
    try:
        return datetime.fromtimestamp(ts).astimezone()
    except (OverflowError, OSError, ValueError):
        return None


def _document(entry: DocumentEntry) -> dict[str, Any]:
    modified = _local_time(entry.mtime)
    return {
        "path": entry.path,
        "modified": modified.strftime("%Y-%m-%d") if modified else None,
        "size_kb": math.ceil(entry.size / 1024),
    }


@dataclass
class _RootState:
    """One root's listing; every field is read and written under the library lock."""

    root: str
    cache_path: Path
    loaded: bool = False
    result: ScanResult | None = None
    error: str | None = None
    done: threading.Event | None = None
    thread: threading.Thread | None = None
    started_at: float = 0.0

    @property
    def scanning(self) -> bool:
        return self.done is not None and not self.done.is_set()


class DocumentLibrary:
    """PDFs under configured roots, listed by background scans and cached on disk.

    A root has at most one scan thread at a time. ``find`` serves a fresh list at
    once and a stale one at once while it is rescanned; it waits, for at most
    ``wait_seconds``, only for a root with no list or when ``refresh`` is asked
    for. A root that cannot be scanned is reported in its status entry and never
    raised. The lock guards state only: nothing scans or waits while holding it.
    """

    def __init__(
        self,
        roots: list[str],
        cache_dir: Path,
        *,
        scanner: Callable[[str], ScanResult] = scan_root,
        clock: Callable[[], float] = time.time,
        ttl: float = CACHE_TTL_SECONDS,
        wait_seconds: float = SCAN_WAIT_SECONDS,
    ) -> None:
        self._roots = list(roots)
        self._states = [_RootState(r, cache_file(cache_dir, r)) for r in self._roots]
        self._scanner = scanner
        self._clock = clock
        self._ttl = ttl
        self._wait_seconds = wait_seconds
        self._lock = threading.Lock()

    @property
    def roots(self) -> list[str]:
        return list(self._roots)

    def warm(self) -> None:
        """Load cached lists and rescan, in the background, any missing or stale."""
        for state in self._states:
            self._load(state)
            with self._lock:
                if state.result is None or self._is_stale(state.result):
                    self._start_scan_locked(state)

    def find(
        self, query: str = "", limit: int = DEFAULT_LIMIT, refresh: bool = False
    ) -> dict[str, Any]:
        """Documents whose path under a root contains every term of ``query``."""
        limit = max(1, min(int(limit), MAX_LIMIT))
        pending: list[threading.Event] = []
        for state in self._states:
            self._load(state)
            with self._lock:
                result = state.result
                if refresh or result is None:
                    pending.append(self._start_scan_locked(state))
                elif self._is_stale(result):
                    self._start_scan_locked(state)
        # One deadline for all roots, so the call waits wait_seconds at most.
        deadline = time.monotonic() + self._wait_seconds
        for done in pending:
            done.wait(max(0.0, deadline - time.monotonic()))
        with self._lock:
            snapshot = [
                (s.root, s.result, s.error, s.scanning, s.started_at)
                for s in self._states
            ]
        return self._response(snapshot, query, limit)

    def wait_for_scans(self, timeout: float | None = None) -> bool:
        """Join the scan threads running now; ``False`` if ``timeout`` ran out.

        The server never needs this (scan threads are daemons). It lets tests
        and embedding code finish without leaving a scan behind.
        """
        with self._lock:
            threads = [s.thread for s in self._states if s.thread is not None]
        deadline = None if timeout is None else time.monotonic() + timeout
        for thread in threads:
            if deadline is None:
                thread.join()
            else:
                thread.join(max(0.0, deadline - time.monotonic()))
        return not any(t.is_alive() for t in threads)

    def _is_stale(self, result: ScanResult) -> bool:
        return self._clock() - result.listed_at > self._ttl

    def _load(self, state: _RootState) -> None:
        """Adopt the root's on-disk cache once; read outside the lock."""
        with self._lock:
            if state.loaded:
                return
        cached = load_cache(state.cache_path, state.root)
        with self._lock:
            if not state.loaded:
                state.loaded = True
                # A scan that finished meanwhile is newer than the file.
                if state.result is None:
                    state.result = cached

    def _start_scan_locked(self, state: _RootState) -> threading.Event:
        """Start a scan of ``state.root`` unless one runs; return its done event."""
        if state.done is not None and not state.done.is_set():
            return state.done
        done = threading.Event()
        thread = threading.Thread(
            target=self._scan, args=(state, done), name=SCAN_THREAD_NAME, daemon=True
        )
        state.done, state.thread, state.started_at = done, thread, self._clock()
        try:
            thread.start()
        except RuntimeError as exc:  # interpreter cannot start another thread
            state.error = f"{type(exc).__name__}: {exc}"
            state.thread = None
            done.set()
        return done

    def _scan(self, state: _RootState, done: threading.Event) -> None:
        result: ScanResult | None = None
        error = "scan ended without a result"
        started = time.monotonic()
        try:
            result = self._scanner(state.root)
            try:
                save_cache(state.cache_path, result)
            except Exception:
                logger.warning(
                    "could not save the library cache for %s",
                    state.root,
                    exc_info=True,
                )
        except BaseException as exc:
            # Nothing above this thread would see it: record it for the status.
            error = f"{type(exc).__name__}: {exc}"
        finally:
            # Publishing and setting `done` together means a waiter that wakes
            # always sees the outcome, and the next call can start a new scan.
            with self._lock:
                if result is None:
                    state.error = error
                else:
                    state.result, state.error = result, None
                done.set()
        elapsed = time.monotonic() - started
        if result is None:
            logger.warning("library scan of %s failed: %s", state.root, error)
        else:
            logger.info(
                "library scan of %s listed %d PDFs in %.1fs",
                state.root,
                len(result.entries),
                elapsed,
            )

    def _response(
        self,
        snapshot: list[tuple[str, ScanResult | None, str | None, bool, float]],
        query: str,
        limit: int,
    ) -> dict[str, Any]:
        now = self._clock()
        roots: list[dict[str, Any]] = []
        matches: list[DocumentEntry] = []
        seen: set[str] = set()
        for root, result, error, scanning, started_at in snapshot:
            if scanning:
                status = "scanning" if result is None else "refreshing"
            else:
                status = "unavailable" if result is None else "ok"
            listed_at = _local_time(result.listed_at) if result else None
            entry: dict[str, Any] = {
                "root": root,
                "status": status,
                "listed_at": (
                    listed_at.isoformat(timespec="seconds") if listed_at else None
                ),
                "documents": len(result.entries) if result else 0,
                "skipped_folders": result.skipped_folders if result else 0,
            }
            if error is not None:
                entry["error"] = error
            if status == "scanning":
                entry["scan_elapsed_seconds"] = int(max(0.0, now - started_at))
            roots.append(entry)
            if result is None:
                continue
            # Overlapping roots list a file twice; the first root keeps it.
            for doc in match_entries(result.entries, query):
                key = os.path.normcase(doc.path)
                if key not in seen:
                    seen.add(key)
                    matches.append(doc)
        return {
            "roots": roots,
            "total_matches": len(matches),
            "truncated": len(matches) > limit,
            "documents": [_document(e) for e in matches[:limit]],
        }
