"""
On-disk store for ingested papers.

Why this module exists: the original pipeline kept each paper's raw
content (text strings, table strings, base64 images) only in a
LangChain InMemoryStore, as bare strings with no metadata. Two things
followed from that:

- A restart lost every paper, while Chroma kept the now-orphaned
  summary vectors on disk.
- Page numbers were dropped before they reached the docstore, so an
  answer could never cite a page, and retrieve() had to guess whether
  a string was an image by sniffing base64 header bytes.

Here each paper gets a directory under config.PAPERS_DIR holding

    docstore.json   {doc_id: record}   - the raw content
    manifest.json   counts, file name, hash, cached summary, ...

where a record is a dict:

    {"type": "text" | "table" | "image",
     "content": <text, table string, or base64 image>,
     "page_number": <int or None>,
     "image": <base64 crop of a table, tables only, may be None>}

The manifest is written last, so its presence means the paper was
ingested completely. A paper directory without a manifest is debris
from an interrupted ingestion and is treated as not ingested.

Papers are keyed by a hash of the PDF's bytes (see pipeline.py), which
is what makes a re-upload of the same file instant.
"""

import json
import logging
import os
import re
import shutil
import threading
import time
from pathlib import Path

import config

logger = logging.getLogger("smartpaper.store")

_MANIFEST = "manifest.json"
_DOCSTORE = "docstore.json"

# paper_id becomes a directory name and part of a Chroma collection
# name, so it must never contain path separators or dots.
_PAPER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{3,64}$")

_write_lock = threading.RLock()


def validate_paper_id(paper_id: str) -> str:
    if not isinstance(paper_id, str) or not _PAPER_ID_RE.match(paper_id):
        raise ValueError(f"Invalid paper_id: {paper_id!r}")
    return paper_id


def paper_dir(paper_id: str) -> Path:
    return config.PAPERS_DIR / validate_paper_id(paper_id)


def _write_json_atomic(path: Path, payload) -> None:
    """Write JSON via a temp file + rename, so a crash never leaves half a file."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    os.replace(tmp, path)


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_paper(paper_id: str, manifest: dict, records: dict) -> dict:
    """
    Persist a fully ingested paper. Returns the manifest as written
    (with paper_id, schema_version and timestamps filled in).
    """
    directory = paper_dir(paper_id)
    now = time.time()
    manifest = {
        **manifest,
        "paper_id": paper_id,
        "schema_version": config.STORE_SCHEMA_VERSION,
        "embedding_model": config.EMBEDDING_MODEL,
        "created_at": now,
        "last_used": now,
    }
    with _write_lock:
        directory.mkdir(parents=True, exist_ok=True)
        # Remove a stale manifest first: until the new one is written,
        # the paper must read as "not ingested".
        (directory / _MANIFEST).unlink(missing_ok=True)
        _write_json_atomic(directory / _DOCSTORE, records)
        _write_json_atomic(directory / _MANIFEST, manifest)
    return manifest


def load_manifest(paper_id: str) -> dict | None:
    """
    Return a paper's manifest, or None if the paper is not (completely)
    ingested, is unreadable, or was written by an incompatible version.
    """
    path = paper_dir(paper_id) / _MANIFEST
    if not path.is_file():
        return None
    try:
        manifest = _read_json(path)
    except (OSError, ValueError) as exc:
        logger.warning("Unreadable manifest for %s: %s", paper_id, exc)
        return None
    if not isinstance(manifest, dict):
        return None
    if manifest.get("schema_version") != config.STORE_SCHEMA_VERSION:
        return None
    # Vectors embedded with a different model are not comparable.
    if manifest.get("embedding_model") != config.EMBEDDING_MODEL:
        return None
    return manifest


def load_records(paper_id: str) -> dict | None:
    """Return {doc_id: record} for a paper, or None if it cannot be read."""
    path = paper_dir(paper_id) / _DOCSTORE
    if not path.is_file():
        return None
    try:
        records = _read_json(path)
    except (OSError, ValueError) as exc:
        logger.warning("Unreadable docstore for %s: %s", paper_id, exc)
        return None
    return records if isinstance(records, dict) else None


def update_manifest(paper_id: str, **fields) -> dict | None:
    """Merge fields into an existing manifest. Returns it, or None if absent."""
    with _write_lock:
        manifest = load_manifest(paper_id)
        if manifest is None:
            return None
        manifest.update(fields)
        _write_json_atomic(paper_dir(paper_id) / _MANIFEST, manifest)
        return manifest


def touch(paper_id: str) -> None:
    """Record that a paper was just used, for least-recently-used eviction."""
    try:
        update_manifest(paper_id, last_used=time.time())
    except OSError as exc:
        logger.warning("Could not update last_used for %s: %s", paper_id, exc)


def delete_paper_files(paper_id: str) -> None:
    """Remove a paper's directory. Missing directories are not an error."""
    with _write_lock:
        shutil.rmtree(paper_dir(paper_id), ignore_errors=True)


def list_manifests() -> list[dict]:
    """Manifests of all completely ingested papers."""
    if not config.PAPERS_DIR.is_dir():
        return []
    manifests = []
    for entry in config.PAPERS_DIR.iterdir():
        if not entry.is_dir() or not _PAPER_ID_RE.match(entry.name):
            continue
        manifest = load_manifest(entry.name)
        if manifest is not None:
            manifests.append(manifest)
    return manifests
