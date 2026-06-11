"""Tests for on-disk label read/write (sensveridian.ingest.label_io)."""
from __future__ import annotations

from pathlib import Path

import pytest

from sensveridian.ingest import label_io


def _make_yolo_dataset(root: Path):
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    (root / "dataset.yaml").write_text("names:\n  0: face\n  1: car\n", encoding="utf-8")
    img = root / "images" / "frame1.jpg"
    img.write_bytes(b"\xff\xd8\xff")
    (root / "labels" / "frame1.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.25 0.25 0.1 0.1\n", encoding="utf-8")
    return img


def test_detect_and_read_yolo(tmp_path: Path):
    img = _make_yolo_dataset(tmp_path / "ds")
    info = label_io.detect_labels(img)
    assert info is not None
    assert info["format"] == "yolo"
    assert info["class_names"] == {0: "face", 1: "car"}
    anns = label_io.read_image_labels(img, info["labels_dir"], info["format"], info["class_names"])
    assert len(anns) == 2
    # YOLO center (0.5,0.5,0.2,0.2) -> top-left xywh (0.4,0.4,0.2,0.2)
    assert anns[0]["cls"] == "face"
    assert anns[0]["box"] == pytest.approx([0.4, 0.4, 0.2, 0.2])
    assert anns[1]["cls"] == "car"


def test_labels_dir_resolution(tmp_path: Path):
    img = tmp_path / "ds" / "images" / "sub" / "x.jpg"
    img.parent.mkdir(parents=True)
    ld = label_io.labels_dir_for(img)
    assert ld == tmp_path / "ds" / "labels" / "sub"


def test_write_back_overwrites_with_backup(tmp_path: Path):
    img = _make_yolo_dataset(tmp_path / "ds")
    labels_dir = tmp_path / "ds" / "labels"
    original = (labels_dir / "frame1.txt").read_text(encoding="utf-8")

    written = label_io.write_image_labels(
        img, labels_dir, "yolo", {0: "face", 1: "car"},
        boxes=[{"cls": "car", "box": [0.1, 0.1, 0.4, 0.4]}],  # verified GT (one car)
    )
    assert written == labels_dir / "frame1.txt"
    # .bak holds the original
    assert (labels_dir / "frame1.txt.bak").read_text(encoding="utf-8") == original
    # new content: class index 1 (car), center (0.3, 0.3), size (0.4, 0.4)
    lines = written.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    idx, cx, cy, w, h = lines[0].split()
    assert idx == "1"
    assert float(cx) == pytest.approx(0.3) and float(cy) == pytest.approx(0.3)
    assert float(w) == pytest.approx(0.4) and float(h) == pytest.approx(0.4)


def test_read_missing_label_returns_empty(tmp_path: Path):
    img = _make_yolo_dataset(tmp_path / "ds")
    (img.parent.parent / "labels" / "frame1.txt").unlink()
    assert label_io.read_image_labels(img, img.parent.parent / "labels", "yolo", {0: "face"}) == []


def test_labels_dir_for_handles_named_variants(tmp_path: Path):
    # extracted_images -> extracted_labels (the qr_som layout)
    root = tmp_path / "ds"
    (root / "extracted_images").mkdir(parents=True)
    (root / "extracted_labels").mkdir(parents=True)
    img = root / "extracted_images" / "a.png"
    img.write_bytes(b"\x89PNG")
    (root / "extracted_labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    ld = label_io.labels_dir_for(img)
    assert ld is not None and ld.name == "extracted_labels"
    anns = label_io.read_image_labels(img, ld, "yolo", {0: "qr"})
    assert anns and anns[0]["cls"] == "qr"
