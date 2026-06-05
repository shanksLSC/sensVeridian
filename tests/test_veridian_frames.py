"""Tests for the frame-sampling helpers (sensveridian.ingest.frames).

Only the pure helpers are tested here (no video decode): the stride math, the
frame-name convention, group-by-stem ordering, and select_every_nth — which
mirror the dataset-generator pipeline.
"""
from __future__ import annotations

from pathlib import Path

from sensveridian.ingest import frames


def test_compute_stride():
    assert frames.compute_stride(30.0, 2.0) == 15
    assert frames.compute_stride(30.0, 30.0) == 1
    assert frames.compute_stride(24.0, 60.0) == 1   # never below 1
    assert frames.compute_stride(0.0, 2.0) == 1     # guards bad fps


def test_frame_name_convention():
    assert frames.frame_name("clip01", 42) == "clip01__frame_000042.jpg"


def test_group_frames_orders_by_index():
    paths = [
        Path("clipA__frame_000010.jpg"),
        Path("clipA__frame_000002.jpg"),
        Path("clipB__frame_000001.jpg"),
        Path("not-a-frame.png"),
    ]
    groups = frames.group_frames(paths)
    assert set(groups) == {"clipA", "clipB"}
    assert [p.name for p in groups["clipA"]] == ["clipA__frame_000002.jpg", "clipA__frame_000010.jpg"]


def test_select_every_nth_per_stem():
    paths = [Path(f"v__frame_{i:06d}.jpg") for i in range(10)]
    kept = frames.select_every_nth(paths, stride=5)
    assert [p.name for p in kept] == ["v__frame_000000.jpg", "v__frame_000005.jpg"]
    # stride 1 keeps everything
    assert frames.select_every_nth(paths, stride=1) == paths


def test_discover_videos(tmp_path: Path):
    (tmp_path / "a.mp4").write_bytes(b"x")
    (tmp_path / "b.txt").write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.mov").write_bytes(b"x")
    found = {p.name for p in frames.discover_videos(tmp_path)}
    assert found == {"a.mp4", "c.mov"}
