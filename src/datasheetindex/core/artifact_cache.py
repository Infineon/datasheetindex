"""Sidecar fingerprint for reusing on-disk build artifacts.

Owns computing the fingerprint, reading and writing the sidecar, and deciding
validity. Imports no PyMuPDF and knows nothing about how a build works, so it
is testable without a PDF.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from importlib.metadata import Distribution
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)

#: The sidecar's filename suffix, appended to the artifact stem.
SIDECAR_SUFFIX = ".build.json"

_HASH_CHUNK_SIZE = 1 << 20


def sidecar_path(output_dir: str | Path, output_stem: str) -> Path:
    """Return the sidecar path beside the two deliverables."""
    return Path(output_dir) / f"{output_stem}{SIDECAR_SUFFIX}"


def sha256_file(path: str | Path) -> str:
    """Hex sha256 of a file's bytes, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """Hex sha256 of text, UTF-8 encoded.

    Used to hash artifact content *after it has been read*, which is what makes
    a straddled or crash-mixed pair of deliverables fail validation rather than
    be served as a coherent artifact.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
    """Write text via a temp file in the same directory, then ``os.replace``.

    A crash or failure leaves the previous generation intact instead of a
    truncated file. The temp file shares the destination's directory so the
    replace stays on one filesystem. The temp name is unique per writing thread
    so concurrent writers to the same destination do not share a temp path and
    truncate each other's content.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        temp_path.write_text(content, encoding="utf-8")
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def is_editable_install() -> bool:
    """True when this package is editable or is an uninstalled source tree.

    Reuse is disabled for both. ``package_version()`` returns the real version
    under an editable install, so a source edit without a version bump would
    otherwise serve pre-edit artifacts, and ``0+unknown == 0+unknown`` would
    match anyway -- exact version equality could never have forced a rebuild on
    its own.

    Note the asymmetry: **no** ``direct_url.json`` means an index-installed
    wheel, which is immutable, so version equality suffices and reuse is on.
    No distribution *at all* means a source tree, where it is not.
    """
    try:
        raw = Distribution.from_name("datasheetindex").read_text("direct_url.json")
    except Exception:
        return True
    if raw is None:
        return False
    try:
        dir_info = json.loads(raw).get("dir_info") or {}
    except json.JSONDecodeError:
        return True
    return bool(dir_info.get("editable"))
