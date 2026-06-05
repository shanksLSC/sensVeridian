"""Tests for the label-lint health check (sensveridian.api.lint)."""
from __future__ import annotations

from types import SimpleNamespace

from sensveridian.api import lint


def _vision_spec():
    return SimpleNamespace(kind="vision")


def test_vision_checks_count_real_issues():
    anns = [
        {"image": "a.jpg", "cls": "car", "box": [0.1, 0.1, 0.2, 0.2], "img_w": 1000, "img_h": 1000},
        {"image": "a.jpg", "cls": "car", "box": [0.1, 0.1, 0.2, 0.2], "img_w": 1000, "img_h": 1000},  # duplicate
        {"image": "a.jpg", "cls": "person", "box": [0.5, 0.5, 0.6, 0.6]},   # off-image (x+w>1)
        {"image": "b.jpg", "cls": "car", "box": [0.2, 0.2, 0.004, 0.004], "img_w": 1000, "img_h": 1000},  # sub-8px
    ]
    report = lint.run_checks(_vision_spec(), annotations=anns, image_refs=["a.jpg"])  # b.jpg unmatched
    checks = {c["label"]: c for c in report["checks"]}
    assert report["boxes"] == 4
    assert checks["Duplicate / overlapping (IoU > .95)"]["n"] == 1
    assert checks["Boxes clipped to image bounds"]["n"] == 1
    assert checks["Sub-8px boxes"]["n"] == 1
    assert checks["Label files without an image"]["n"] == 1  # b.jpg
    assert "car" in checks["Class imbalance"]["note"]
    # score penalised by 1 error (*4) and warns
    assert report["score"] < 100


def test_clean_set_scores_100():
    anns = [{"image": "a.jpg", "cls": "car", "box": [0.1, 0.1, 0.2, 0.2], "img_w": 1000, "img_h": 1000}]
    report = lint.run_checks(_vision_spec(), annotations=anns, image_refs=["a.jpg"])
    assert report["score"] == 100


def test_audio_checks():
    spec = SimpleNamespace(kind="audio")
    segs = [
        {"clip": "c1", "start": 0.0, "end": 0.2, "label": "speech", "dur": 1.0},
        {"clip": "c1", "start": 0.1, "end": 0.3, "label": "speech", "dur": 1.0},  # overlaps previous
        {"clip": "c1", "start": 0.5, "end": 0.5, "label": "noise", "dur": 1.0},   # zero-length
        {"clip": "c1", "start": 0.9, "end": 1.5, "label": "siren", "dur": 1.0},   # out-of-range
    ]
    report = lint.run_checks(spec, segments=segs)
    checks = {c["label"]: c for c in report["checks"]}
    assert checks["Overlapping segments"]["n"] == 1
    assert checks["Zero-length segments"]["n"] == 1
    assert checks["Out-of-range timestamps"]["n"] == 1
