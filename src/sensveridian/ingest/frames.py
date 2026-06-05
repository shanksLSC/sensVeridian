"""Frame sampling for video ingest.

Follows the organisation's existing dataset-generator convention rather than
ffmpeg (see ``dataset-generator/src/extract_frames.py`` and
``dataset-generator/src/select_every_nth.py``, which are the source of truth):

* **Decode** with OpenCV ``VideoCapture`` and **target-fps stride** sampling:
  ``stride = max(1, round(src_fps / target_fps))``; keep frames where
  ``frame_idx % stride == 0``. Frames are written as
  ``<video_stem>__frame_NNNNNN.jpg`` (JPEG quality 95).
* **Near-duplicate dedup** via ``select_every_nth``: group by source-video stem,
  keep one frame out of every ``stride`` — a cheap temporal dedup that
  complements the Orchestrator's exact sha-256 content dedup.

cv2 is already a sensVeridian dependency, so there is no external ffmpeg
requirement. The pure helpers (``compute_stride``, ``group_frames``,
``select_every_nth``) are unit-testable without a video file.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import cv2

from .kinds import IMAGE_EXTS, VIDEO_EXTS, discover_images, discover_videos  # noqa: F401

DEFAULT_JPEG_QUALITY = 95
FRAME_RE = re.compile(r"^(?P<stem>.+)__frame_(?P<idx>\d+)\.jpg$")


def compute_stride(src_fps: float, target_fps: float) -> int:
    """Frames to skip to approximate ``target_fps`` from ``src_fps`` (>= 1)."""
    if src_fps <= 0 or target_fps <= 0:
        return 1
    return max(1, int(round(src_fps / target_fps)))


def frame_name(video_stem: str, frame_idx: int) -> str:
    return f"{video_stem}__frame_{frame_idx:06d}.jpg"


def decode_video_frames(
    video_path: str | Path,
    out_dir: str | Path,
    target_fps: float,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    resume: bool = True,
) -> list[Path]:
    """Decode ``video_path`` to ``<stem>__frame_NNNNNN.jpg`` at ~``target_fps``.

    Mirrors ``dataset-generator`` ``extract_frames.process_video`` but writes to
    a single output directory. Returns the produced frame paths (sorted).
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if src_fps <= 0:
        cap.release()
        raise RuntimeError(f"invalid source fps for {video_path.name}: {src_fps}")

    stride = compute_stride(src_fps, target_fps)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    written: list[Path] = []
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride == 0:
                out_path = out_dir / frame_name(video_path.stem, frame_idx)
                if resume and out_path.exists():
                    written.append(out_path)
                else:
                    ok_w, buf = cv2.imencode(".jpg", frame, encode_params)
                    if ok_w:
                        out_path.write_bytes(buf.tobytes())
                        written.append(out_path)
            frame_idx += 1
    finally:
        cap.release()
    return sorted(written)


def group_frames(paths: list[Path]) -> dict[str, list[Path]]:
    """Group ``<stem>__frame_NNNNNN.jpg`` paths by source-video stem, ordered by
    numeric frame index (mirrors select_every_nth.group_frames)."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in paths:
        m = FRAME_RE.match(p.name)
        if m:
            groups[m.group("stem")].append(p)
    for stem, ps in groups.items():
        ps.sort(key=lambda q: int(FRAME_RE.match(q.name).group("idx")))
    return dict(groups)


def select_every_nth(paths: list[Path], stride: int) -> list[Path]:
    """Keep one out of every ``stride`` frames per source-video stem — the cheap
    temporal near-duplicate dropper from dataset-generator."""
    if stride <= 1:
        return list(paths)
    kept: list[Path] = []
    for _stem, frames in sorted(group_frames(paths).items()):
        kept.extend(frames[::stride])
    return kept


def sample_video(
    video_path: str | Path,
    out_dir: str | Path,
    target_fps: float,
    dedup_stride: int = 1,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> list[Path]:
    """Decode a video to frames at ``target_fps``, then apply the optional
    ``select_every_nth`` near-duplicate dedup. Returns the final frame paths."""
    frames = decode_video_frames(video_path, out_dir, target_fps, jpeg_quality=jpeg_quality)
    return select_every_nth(frames, dedup_stride)
