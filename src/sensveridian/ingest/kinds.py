"""Media kind constants + discovery — stdlib only (no cv2), so the API can
import the extension sets without pulling in OpenCV."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def discover_images(root: str | Path, limit: Optional[int] = None) -> list[Path]:
    """Image files under ``root`` (recursive, sorted). ``limit`` caps the count."""
    root = Path(root)
    if not root.exists():
        return []
    if root.is_file():
        return [root] if root.suffix.lower() in IMAGE_EXTS else []
    out: list[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out.append(p)
            if limit is not None and len(out) >= limit:
                break
    return out


def discover_videos(root: str | Path) -> list[Path]:
    """Video files under ``root`` (recursive, sorted)."""
    root = Path(root)
    if not root.exists():
        return []
    if root.is_file():
        return [root] if root.suffix.lower() in VIDEO_EXTS else []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
