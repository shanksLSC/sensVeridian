"""Tests for the per-model output interpreters (sensveridian.postprocessors).

These exercise the decode logic on synthetic, well-formed head tensors so the
ported math (anchor/FCOS decode, sigmoid/softmax gating, NMS, normalization) is
verified independently of the TensorFlow models.
"""
from __future__ import annotations

import numpy as np
import pytest

import sensveridian.postprocessors as pp
from sensveridian.postprocessors import common, constants as K, embedding


# ---- shared math -----------------------------------------------------------
def test_common_math():
    assert common.sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)
    sm = common.softmax(np.array([1.0, 1.0, 1.0]))
    assert sm.sum() == pytest.approx(1.0) and sm[0] == pytest.approx(1 / 3)
    assert common.iou_xyxy([0, 0, 2, 2], [0, 0, 2, 2]) == pytest.approx(1.0)
    assert common.iou_xyxy([0, 0, 1, 1], [5, 5, 6, 6]) == 0.0


def test_nms_greedy_suppresses_overlap():
    boxes = [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]]
    keep = common.nms_greedy(boxes, [0.9, 0.8, 0.7], 0.45, 10)
    assert keep == [0, 2]  # box 1 overlaps box 0 and is dropped


# ---- AMOD (FCOS, 6 tensors) ------------------------------------------------
def test_amod_decode_one_detection():
    outs = []
    for (gh, gw, _stride) in K.MOD_SCALES:
        outs.append(np.zeros((1, gh, gw, 4), np.float32))            # bbox
        outs.append(np.full((1, gh, gw, K.MOD_NUM_CLASSES + 1), -10.0, np.float32))  # logits
    # light up one cell at the coarsest (9x12) scale: bbox=outs[4], logits=outs[5]
    outs[4][0, 4, 5, :] = 1.0
    outs[5][0, 4, 5, 0] = 10.0      # objectness
    outs[5][0, 4, 5, 1 + 2] = 10.0  # class 2 (truck)
    dets = pp.interpret("amod", outs, 384, 288, 0.3)
    assert len(dets) == 1
    d = dets[0]
    assert d["class_id"] == 2 and d["conf"] > 0.9
    assert all(0.0 <= v <= 1.0 for v in d["bbox"]) and d["bbox"][2] > d["bbox"][0]


# ---- QR (4 anchors) --------------------------------------------------------
def test_qr_decode_one_detection():
    a = np.full((1, 9, 16, 36), -10.0, np.float32)
    a[0, 4, 8, 0] = 10.0                 # anchor 0 confidence
    a[0, 4, 8, 4:8] = [0.0, 0.0, 1.0, 1.0]  # dx,dy,dw,dh
    dets = pp.interpret("qrcode", a, 256, 144, 0.5)
    assert len(dets) == 1
    assert dets[0]["class_id"] == 0 and dets[0]["conf"] > 0.9
    assert dets[0]["bbox"][2] > dets[0]["bbox"][0]


# ---- FD (2 anchors + landmarks) --------------------------------------------
def test_fd_decode_one_detection_with_landmarks():
    a = np.full((1, 9, 16, 38), -10.0, np.float32)
    a[0, 4, 8, 0] = 10.0                  # anchor 0 confidence
    a[0, 4, 8, 2:6] = [0.0, 0.0, 1.0, 1.0]   # bbox deltas (base = num_conf=2)
    dets = pp.interpret("fd", a, 256, 144, 0.5)
    assert len(dets) == 1
    assert "landmarks" in dets[0] and len(dets[0]["landmarks"]) == K.FD_NUM_LANDMARKS


# ---- registry --------------------------------------------------------------
def test_registry():
    assert pp.has_interpreter("amod") and pp.has_interpreter("qrcode") and pp.has_interpreter("fd")
    assert not pp.has_interpreter("aed")
    with pytest.raises(KeyError):
        pp.interpret("nope", [], 1, 1)


# ---- FR embedding ----------------------------------------------------------
def test_embedding_normalize_and_cosine():
    v = embedding.interpret_embedding(np.array([3.0, 4.0]))
    assert np.linalg.norm(v) == pytest.approx(1.0)
    assert embedding.cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)
    assert embedding.cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
