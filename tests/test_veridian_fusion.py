"""Tests for the detection-fusion core (sensveridian.api.fusion)."""
from __future__ import annotations

import pytest

from sensveridian.api import fusion


def test_iou_xywh_identity_and_disjoint():
    assert fusion.iou_xywh([0, 0, 1, 1], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert fusion.iou_xywh([0, 0, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1]) == 0.0
    assert fusion.iou_xywh(None, [0, 0, 1, 1]) == 0.0


def test_normalize_box_spaces():
    # normalized xyxy -> xywh
    assert fusion.normalize_box([0.3, 0.4, 0.5, 0.6], 1000, 800, "norm") == pytest.approx([0.3, 0.4, 0.2, 0.2])
    # absolute pixels -> normalized
    assert fusion.normalize_box([100, 80, 300, 240], 1000, 800, "abs_px") == pytest.approx([0.1, 0.1, 0.2, 0.2])
    # auto: large coords detected as pixels
    assert fusion.normalize_box([100, 80, 300, 240], 1000, 800, "auto") == pytest.approx([0.1, 0.1, 0.2, 0.2])
    # auto: small coords detected as normalized
    assert fusion.normalize_box([0.1, 0.1, 0.3, 0.3], 1000, 800, "auto") == pytest.approx([0.1, 0.1, 0.2, 0.2])
    # clamps to [0,1]
    assert fusion.normalize_box([-10, -10, 2000, 2000], 1000, 800, "abs_px") == pytest.approx([0.0, 0.0, 1.0, 1.0])


def test_extract_amod_and_qr():
    amod = fusion.extract_pred_detections(
        "amod", {"detections": [{"bbox": [0.3, 0.4, 0.5, 0.6], "conf": 0.9, "class_id": 1}]}, 1000, 800
    )
    assert amod[0]["cls"] == "car" and amod[0]["model"] == "amod"  # class_id 1 = car
    assert amod[0]["box"] == pytest.approx([0.3, 0.4, 0.2, 0.2])
    assert amod[0]["conf"] == 0.9

    qr = fusion.extract_pred_detections(
        "qrcode", {"detections": [{"bbox": [0.1, 0.1, 0.2, 0.2], "conf": 0.6}], "decoded_texts": ["PKG-1"]}, 1000, 800
    )
    assert qr[0]["cls"] == "qr"
    assert qr[0]["decoded"] == {"pred": "PKG-1"}


def test_extract_fd_abs_px_and_fr_recognized():
    fd = fusion.extract_pred_detections(
        "fd", {"detections": [{"bbox": [100, 80, 300, 240], "conf": 0.8}]}, 1000, 800
    )
    assert fd[0]["cls"] == "face"
    assert fd[0]["box"] == pytest.approx([0.1, 0.1, 0.2, 0.2])

    # FR uses 'recognized', not 'detections'
    fr = fusion.extract_pred_detections(
        "fr", {"recognized": [{"bbox": [100, 80, 300, 240], "matched_person_id": "P-1042", "score": 0.88}]},
        1000, 800, name_map={"P-1042": "A. Okafor"},
    )
    assert fr[0]["cls"] == "face"
    assert fr[0]["identity"]["pred"] == "A. Okafor"
    assert fr[0]["identity"]["person_id"] == "P-1042"
    assert fr[0]["conf"] == pytest.approx(0.88)


def test_fuse_no_gt_layer_is_provisional_match():
    preds = {"amod": {"detections": [{"bbox": [0.3, 0.4, 0.5, 0.6], "conf": 0.9, "class_id": 0}]}}
    dets = fusion.fuse_detections("abc123", preds, 1000, 800)
    assert len(dets) == 1
    d = dets[0]
    assert d["state"] == "match"
    assert d["gt"] == d["pred"]  # provisional: prediction is the working ground truth
    assert d["id"] == "abc123_amod_0"


def test_fuse_with_gt_match_mismatch_fp_miss():
    preds = {
        "amod": {
            "detections": [
                {"bbox": [0.30, 0.40, 0.50, 0.60], "conf": 0.9, "class_id": 1},  # car, matches gt0
                {"bbox": [0.80, 0.80, 0.90, 0.90], "conf": 0.5, "class_id": 2},  # truck, no gt -> fp
            ]
        }
    }
    gt_items = [
        {"cls": "car", "box": [0.30, 0.40, 0.20, 0.20]},   # overlaps pred0 -> match
        {"cls": "person", "box": [0.05, 0.05, 0.05, 0.05]},  # no pred -> miss
    ]
    dets = fusion.fuse_detections("img0", preds, 1000, 800, gt_items=gt_items)
    by_state = {}
    for d in dets:
        by_state.setdefault(d["state"], []).append(d)
    assert len(by_state["match"]) == 1
    assert len(by_state["fp"]) == 1
    assert len(by_state["miss"]) == 1
    miss = by_state["miss"][0]
    assert miss["pred"] is None and miss["gt"] == [0.05, 0.05, 0.05, 0.05]


def test_fuse_two_models_one_gt_both_match():
    # two models detect the same QR; each is scored against GT independently, so
    # both are matches (not one match + one false positive), and no miss.
    preds = {
        "qr_gray": {"detections": [{"bbox": [0.10, 0.10, 0.60, 0.60], "conf": 0.95, "class_id": 0}]},
        "qr_rgb": {"detections": [{"bbox": [0.11, 0.11, 0.61, 0.61], "conf": 0.94, "class_id": 0}]},
    }
    gt = [{"cls": "qr", "box": [0.10, 0.10, 0.50, 0.50]}]
    dets = fusion.fuse_detections("im", preds, 100, 100, gt_items=gt)
    states = sorted(d["state"] for d in dets)
    assert states == ["match", "match"]
    assert not any(d["state"] in ("fp", "miss") for d in dets)


def test_fuse_mismatch_on_class_disagreement():
    preds = {"amod": {"detections": [{"bbox": [0.30, 0.40, 0.50, 0.60], "conf": 0.9, "class_id": 0}]}}  # car
    gt_items = [{"cls": "truck", "box": [0.30, 0.40, 0.20, 0.20]}]  # overlaps but wrong class
    dets = fusion.fuse_detections("img0", preds, 1000, 800, gt_items=gt_items)
    assert dets[0]["state"] == "mismatch"


def test_fuse_review_overrides():
    preds = {"amod": {"detections": [{"bbox": [0.30, 0.40, 0.50, 0.60], "conf": 0.9, "class_id": 0}]}}
    det_id = "img0_amod_0"
    # rejected -> fp
    rej = fusion.fuse_detections("img0", preds, 1000, 800, reviews={det_id: {"verdict": "rejected"}})
    assert rej[0]["state"] == "fp" and rej[0]["gt"] is None
    # edited -> gt replaced; far box -> mismatch
    edited = fusion.fuse_detections(
        "img0", preds, 1000, 800, reviews={det_id: {"verdict": "edited", "box": [0.0, 0.0, 0.05, 0.05]}}
    )
    assert edited[0]["gt"] == [0.0, 0.0, 0.05, 0.05]
    assert edited[0]["state"] == "mismatch"


def test_rollups_match_reference_formula():
    objects = [
        {"state": "match", "gt": [0, 0, 1, 1]},
        {"state": "match", "gt": [0, 0, 1, 1]},
        {"state": "fp", "gt": None},
        {"state": "miss", "gt": [0, 0, 1, 1]},
    ]
    r = fusion.image_rollup(objects)
    # matched=2, conflicts=2 (fp+miss) -> agreement 2/4
    assert r["conflicts"] == 2
    assert r["agreement"] == pytest.approx(0.5)
    ds = fusion.dataset_rollup([r, {"agreement": 1.0, "conflicts": 0}], reviewed=1)
    assert ds["conflicts"] == 2 and ds["reviewed"] == 1 and ds["count"] == 2


def test_detection_image_id_roundtrip():
    assert fusion.detection_image_id("abc123_amod_0") == "abc123"
    assert fusion.detection_image_id("flip:abc123_fr_2") == "abc123"
    assert fusion.detection_image_id("img:abc123") is None  # caller handles img: ids


# ---- committed_gt (Path 1 curation write-back) -----------------------------
def _amod(bbox, class_id=1, conf=0.9):
    return {"amod": {"detections": [{"bbox": bbox, "conf": conf, "class_id": class_id}]}}


def test_committed_gt_provisional_writes_unreviewed_pred():
    # no GT layer -> a non-rejected prediction is written as GT (auto-label)
    out = fusion.committed_gt("im", _amod([0.1, 0.1, 0.6, 0.6]), 100, 100)
    assert out == [{"cls": "car", "box": pytest.approx([0.1, 0.1, 0.5, 0.5])}]


def test_committed_gt_rejected_pred_is_dropped():
    out = fusion.committed_gt("im", _amod([0.1, 0.1, 0.6, 0.6]), 100, 100,
                              reviews={"im_amod_0": {"verdict": "rejected"}})
    assert out == []


def test_committed_gt_edited_box_is_used():
    out = fusion.committed_gt("im", _amod([0.1, 0.1, 0.6, 0.6]), 100, 100,
                              reviews={"im_amod_0": {"verdict": "edited", "box": [0.2, 0.2, 0.3, 0.3]}})
    assert out == [{"cls": "car", "box": [0.2, 0.2, 0.3, 0.3]}]


def test_committed_gt_match_keeps_label_vocabulary():
    # pred matches imported GT -> keep the GT's own label ("vehicle"), pred box
    gt = [{"cls": "vehicle", "box": [0.1, 0.1, 0.5, 0.5]}]
    out = fusion.committed_gt("im", _amod([0.1, 0.1, 0.6, 0.6]), 100, 100,
                              gt_items=gt, class_map={"vehicle": "car"})
    assert out == [{"cls": "vehicle", "box": pytest.approx([0.1, 0.1, 0.5, 0.5])}]


def test_committed_gt_unmatched_unaccepted_pred_skipped_gt_kept():
    gt = [{"cls": "vehicle", "box": [0.1, 0.1, 0.5, 0.5]}]
    out = fusion.committed_gt("im", _amod([0.7, 0.7, 0.9, 0.9]), 100, 100, gt_items=gt)
    assert out == [{"cls": "vehicle", "box": [0.1, 0.1, 0.5, 0.5]}]


def test_committed_gt_accepted_new_pred_uses_inverse_class_map():
    gt = [{"cls": "vehicle", "box": [0.1, 0.1, 0.5, 0.5]}]
    out = fusion.committed_gt("im", _amod([0.7, 0.7, 0.9, 0.9]), 100, 100, gt_items=gt,
                              reviews={"im_amod_0": {"verdict": "accepted"}},
                              class_map={"vehicle": "car"})
    assert {"cls": "vehicle", "box": pytest.approx([0.7, 0.7, 0.2, 0.2])} in out
    assert {"cls": "vehicle", "box": [0.1, 0.1, 0.5, 0.5]} in out
    assert len(out) == 2


def test_committed_gt_rejected_pred_overrides_matching_gt():
    gt = [{"cls": "vehicle", "box": [0.1, 0.1, 0.5, 0.5]}]
    out = fusion.committed_gt("im", _amod([0.1, 0.1, 0.6, 0.6]), 100, 100, gt_items=gt,
                              reviews={"im_amod_0": {"verdict": "rejected"}})
    assert out == []  # the GT box the rejected pred overlapped is dropped too
