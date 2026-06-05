"""Filesystem browse — pick a local folder under the datasets root to ingest.

The raw data store is the local filesystem (no S3). These endpoints let the
Studio ingest screen list and drill into ``settings.datasets_root``. Every path
is resolved and checked to live under the root (no traversal escapes).
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query

from ..config import settings
from ..deps import require_auth
from ...ingest.kinds import IMAGE_EXTS, VIDEO_EXTS

router = APIRouter(dependencies=[Depends(require_auth)])


def _root() -> Path:
    return Path(settings.datasets_root).resolve()


def safe_resolve(rel: str) -> Path:
    """Resolve ``rel`` under the datasets root, rejecting anything that escapes
    it (``..`` / absolute paths / symlink breakouts)."""
    root = _root()
    target = (root / (rel or "")).resolve()
    if target != root and root not in target.parents:
        raise HTTPException(400, "path escapes datasets root")
    if not target.exists():
        raise HTTPException(404, f"path not found: {rel}")
    return target


def media_counts(path: Path, max_files: int = 5_000, max_depth: int = 4) -> dict:
    """Count images/videos under ``path`` (bounded) and note a labels dir.

    Bounded on purpose: this powers a folder picker, so an exact count over
    hundreds of thousands of files is unnecessary — stop at ``max_files`` and
    flag ``truncated``."""
    n_img = n_vid = 0
    has_labels = False
    base_depth = len(path.parts)
    seen = 0
    for dirpath, dirnames, filenames in os.walk(path):
        depth = len(Path(dirpath).parts) - base_depth
        if depth > max_depth:
            dirnames[:] = []
            continue
        if "labels" in {d.lower() for d in dirnames} or Path(dirpath).name.lower() == "labels":
            has_labels = True
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMAGE_EXTS:
                n_img += 1
            elif ext in VIDEO_EXTS:
                n_vid += 1
            seen += 1
            if seen >= max_files:
                return {"images": n_img, "videos": n_vid, "has_labels": has_labels, "truncated": True}
    return {"images": n_img, "videos": n_vid, "has_labels": has_labels, "truncated": False}


def _kind(counts: dict) -> str:
    if counts["images"] and counts["videos"]:
        return "mixed"
    if counts["videos"]:
        return "video"
    return "image"


@router.get("/fs/datasets")
def list_fs_datasets():
    """Immediate subfolders of the datasets root that contain images or videos.

    Sync def on purpose: the os.walk scan is blocking, so FastAPI runs it in a
    threadpool rather than stalling the event loop (and other requests)."""
    root = _root()
    if not root.exists():
        raise HTTPException(404, f"datasets root not found: {root}")
    out = []
    for entry in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        counts = media_counts(entry)
        if counts["images"] == 0 and counts["videos"] == 0:
            continue
        out.append(
            {
                "name": entry.name,
                "path": entry.name,  # relative to datasets_root
                "images": counts["images"],
                "videos": counts["videos"],
                "has_labels": counts["has_labels"],
                "kind": _kind(counts),
            }
        )
    return {"root": str(root), "entries": out}


@router.get("/fs/browse")
def browse(path: str = Query("")):
    """Drill into a folder: list subfolders (with media counts) and a sample of files."""
    target = safe_resolve(path)
    if not target.is_dir():
        raise HTTPException(400, "not a directory")
    root = _root()
    folders, files = [], []
    for entry in sorted(target.iterdir(), key=lambda p: p.name.lower()):
        if entry.name.startswith("."):
            continue
        rel = str(entry.relative_to(root))
        if entry.is_dir():
            counts = media_counts(entry)
            folders.append({"name": entry.name, "path": rel, "images": counts["images"],
                            "videos": counts["videos"], "has_labels": counts["has_labels"],
                            "kind": _kind(counts)})
        else:
            ext = entry.suffix.lower()
            if ext in IMAGE_EXTS or ext in VIDEO_EXTS:
                files.append({"name": entry.name, "path": rel,
                              "kind": "video" if ext in VIDEO_EXTS else "image"})
    return {"root": str(root), "path": path, "folders": folders, "files": files[:50],
            "n_files": len(files)}
