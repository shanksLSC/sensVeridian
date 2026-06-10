"""Unit tests for api.metrics.compute_metrics (pure, DB-free).

Synthetic predicted/GT samples with hand-computable TP/FP/FN so the headline
P/R/F1/agreement and COCO mAP@[.5:.95] are checked against known values.
Boxes are normalized xywh; classes are already model-aligned (the class_map is
applied upstream in fusion/the store, not here).
"""
from __future__ import annotations

import math

from sensveridian.api.metrics import MAP_IOUS, compute_metrics


def _approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return math.isclose(a, b, abs_tol=tol)


def test_perfect_match():
    samples = [{"preds": [{"box": [0, 0, 0.5, 0.5], "cls": "car", "conf": 0.9}],
                "gt": [{"box": [0, 0, 0.5, 0.5], "cls": "car"}]}]
    m = compute_metrics(samples)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 0, 0)
    assert _approx(m["precision"], 1.0)
    assert _approx(m["recall"], 1.0)
    assert _approx(m["f1"], 1.0)
    assert _approx(m["agreement"], 1.0)
    assert _approx(m["AP50"], 1.0)
    assert _approx(m["mAP"], 1.0)
    assert m["per_class_ap50"] == {"car": 1.0}


def test_false_positive_only():
    samples = [{"preds": [{"box": [0, 0, 0.5, 0.5], "cls": "car", "conf": 0.9}], "gt": []}]
    m = compute_metrics(samples)
    assert (m["tp"], m["fp"], m["fn"]) == (0, 1, 0)
    assert _approx(m["precision"], 0.0)
    assert _approx(m["recall"], 0.0)
    assert _approx(m["agreement"], 0.0)
    # no GT class anywhere -> mAP averages over an empty class set -> 0
    assert _approx(m["mAP"], 0.0)
    assert m["n_pred"] == 1 and m["n_gt"] == 0


def test_false_negative_only():
    samples = [{"preds": [], "gt": [{"box": [0, 0, 0.5, 0.5], "cls": "car"}]}]
    m = compute_metrics(samples)
    assert (m["tp"], m["fp"], m["fn"]) == (0, 0, 1)
    assert _approx(m["recall"], 0.0)
    assert _approx(m["AP50"], 0.0)
    assert _approx(m["mAP"], 0.0)


def test_low_iou_is_not_a_match():
    # boxes barely overlap -> IoU < 0.5 -> FP + FN, not a match
    samples = [{"preds": [{"box": [0, 0, 0.5, 0.5], "cls": "car", "conf": 0.9}],
                "gt": [{"box": [0.45, 0.45, 0.5, 0.5], "cls": "car"}]}]
    m = compute_metrics(samples)
    assert (m["tp"], m["fp"], m["fn"]) == (0, 1, 1)


def test_wrong_class_is_not_a_match():
    samples = [{"preds": [{"box": [0, 0, 0.5, 0.5], "cls": "car", "conf": 0.9}],
                "gt": [{"box": [0, 0, 0.5, 0.5], "cls": "person"}]}]
    m = compute_metrics(samples)
    assert (m["tp"], m["fp"], m["fn"]) == (0, 1, 1)


def test_partial_dataset_pr_and_map():
    # image A: car pred matches car gt; image B: car pred but person gt
    samples = [
        {"preds": [{"box": [0, 0, 0.5, 0.5], "cls": "car", "conf": 0.9}],
         "gt": [{"box": [0, 0, 0.5, 0.5], "cls": "car"}]},
        {"preds": [{"box": [0, 0, 0.5, 0.5], "cls": "car", "conf": 0.8}],
         "gt": [{"box": [0, 0, 0.5, 0.5], "cls": "person"}]},
    ]
    m = compute_metrics(samples)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    assert _approx(m["precision"], 0.5)
    assert _approx(m["recall"], 0.5)
    assert _approx(m["f1"], 0.5)
    assert _approx(m["agreement"], round(1 / 3, 4), tol=1e-3)
    # car AP = 1.0 (its single GT is found), person AP = 0.0 (no person pred)
    assert _approx(m["per_class_ap50"]["car"], 1.0)
    assert _approx(m["per_class_ap50"]["person"], 0.0)
    assert _approx(m["AP50"], 0.5)
    assert _approx(m["mAP"], 0.5)


def test_multiple_preds_one_gt_counts_extra_as_fp():
    # two overlapping preds, one GT -> one TP, one FP
    samples = [{"preds": [{"box": [0, 0, 0.5, 0.5], "cls": "car", "conf": 0.9},
                          {"box": [0, 0, 0.5, 0.5], "cls": "car", "conf": 0.7}],
                "gt": [{"box": [0, 0, 0.5, 0.5], "cls": "car"}]}]
    m = compute_metrics(samples)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 0)
    assert _approx(m["recall"], 1.0)
    assert _approx(m["precision"], 0.5)


def test_empty_dataset():
    m = compute_metrics([])
    assert m["n_images"] == 0
    assert (m["tp"], m["fp"], m["fn"]) == (0, 0, 0)
    assert _approx(m["mAP"], 0.0)
    assert m["iou_thr"] == 0.5


def test_map_iou_grid():
    assert MAP_IOUS[0] == 0.5 and MAP_IOUS[-1] == 0.95 and len(MAP_IOUS) == 10
