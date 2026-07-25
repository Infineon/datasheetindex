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
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ArtifactRecord:
    """What the sidecar stores about one build.

    Everything needed to decide whether the artifacts on disk are the ones a
    fresh build would produce. Recorded rather than inferred from the
    deliverables: ``TocNode.to_dict`` omits empty fields, and the emitted
    ``toc_quality`` block drops ``details``, so the deliverables alone cannot
    reconstruct the build.
    """

    source_sha256: str
    source_size: int
    build_options: dict[str, object]
    datasheetindex_version: str
    json_name: str
    json_sha256: str
    text_name: str
    text_sha256: str
    toc_quality: dict[str, object]
    llm_enrichment_incomplete: bool = False
    llm_enrichment_notes: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "source_sha256": self.source_sha256,
            "source_size": self.source_size,
            "build_options": dict(self.build_options),
            "datasheetindex_version": self.datasheetindex_version,
            "artifacts": {
                "json": {"name": self.json_name, "sha256": self.json_sha256},
                "text": {"name": self.text_name, "sha256": self.text_sha256},
            },
            "toc_quality": dict(self.toc_quality),
            "llm_enrichment_incomplete": self.llm_enrichment_incomplete,
            "llm_enrichment_notes": list(self.llm_enrichment_notes),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ArtifactRecord:
        """Rebuild from ``to_dict`` output.

        Raises on a missing key rather than defaulting: a sidecar that does not
        carry a fingerprint field cannot be validated against it, and guessing
        would turn a bug into a false cache hit.
        """
        artifacts = data["artifacts"]
        return cls(
            source_sha256=data["source_sha256"],
            source_size=data["source_size"],
            build_options=dict(data["build_options"]),
            datasheetindex_version=data["datasheetindex_version"],
            json_name=artifacts["json"]["name"],
            json_sha256=artifacts["json"]["sha256"],
            text_name=artifacts["text"]["name"],
            text_sha256=artifacts["text"]["sha256"],
            toc_quality=dict(data["toc_quality"]),
            llm_enrichment_incomplete=bool(data["llm_enrichment_incomplete"]),
            llm_enrichment_notes=tuple(data["llm_enrichment_notes"]),
        )


def write_sidecar(path: Path, record: ArtifactRecord) -> None:
    """Write the sidecar atomically. Raises; the caller decides how to degrade."""
    atomic_write_text(path, json.dumps(record.to_dict(), indent=2, ensure_ascii=False))


def read_sidecar(path: Path) -> ArtifactRecord | None:
    """Load the sidecar, or None when it cannot be used.

    A missing or corrupt sidecar is routine and logged at debug -- the failure
    direction is a rebuild, which is safe. A parseable file with the wrong shape
    is logged at warning, because it means ``to_dict`` and ``from_dict`` have
    diverged, which is a bug.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        logger.debug("No readable build sidecar at %s", path)
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        logger.debug("Build sidecar at %s is not valid JSON", path)
        return None
    try:
        return ArtifactRecord.from_dict(payload)
    except (KeyError, TypeError, ValueError):
        logger.warning("Build sidecar at %s has an unexpected shape; rebuilding", path)
        return None


def remove_sidecar(path: Path) -> None:
    """Delete the sidecar, best effort.

    Called before the deliverables are rewritten, so a concurrent reader finds
    no sidecar and rebuilds rather than validating new data against an old
    record.
    """
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.debug("Could not remove build sidecar at %s", path)
