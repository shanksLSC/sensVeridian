"""Predicted-vs-GT detection metrics (Path 2) — pure, NumPy-free.

Computes, from per-image predicted + ground-truth boxes (normalized xywh, with a
model-aligned class name):

- aggregate precision / recall / F1 / agreement at IoU 0.5, and
- COCO-style mAP@[.5:.95] (per-class AP averaged over IoU 0.5:0.05:0.95) plus
  AP50 / AP75.

Greedy matching per class, highest-confidence prediction first. GT class
alignment is applied upstream (fusion's class_map), so classes compare directly.
"""
from __future__ import annotations

from typing import Optional

from .fusion import iou_xywh

DEFAULT_IOU = 0.5
MAP_IOUS = [round(0.5 + 0.05 * i, 2) for i in range(10)]  # 0.50 .. 0.95


def _match_counts(preds: list[dict], gts: list[dict], iou_thr: float) -> tuple[int, int, int]:
    """Greedy TP/FP/FN for one image+class at one IoU threshold (preds sorted by
    confidence desc)."""
    used = [False] * len(gts)
    tp = 0
    for p in preds:
        best_j, best_iou = -1, iou_thr
        for j, g in enumerate(gts):
            if used[j]:
                continue
            v = iou_xywh(_xyxy(p["box"]), _xyxy(g["box"]))
            if v >= best_iou:
                best_iou, best_j = v, j
        if best_j >= 0:
            used[best_j] = True
            tp += 1
    fp = len(preds) - tp
    fn = len(gts) - tp
    return tp, fp, fn


def _xyxy(b):
    return [b[0], b[1], b[0] + b[2], b[1] + b[3]]


def _average_precision(matched: list[int], confs: list[float], n_gt: int) -> float:
    """All-point AP from per-prediction match flags ordered by confidence desc."""
    if n_gt == 0:
        return 0.0
    order = sorted(range(len(confs)), key=lambda i: confs[i], reverse=True)
    tp = fp = 0
    prev_recall = 0.0
    ap = 0.0
    # precision/recall as we walk predictions high->low confidence
    rec_prec = []
    for i in order:
        if matched[i]:
            tp += 1
        else:
            fp += 1
        rec_prec.append((tp / n_gt, tp / (tp + fp)))
    # monotonic-decreasing precision envelope, integrate over recall
    for recall, precision in rec_prec:
        # use max precision for recall >= current (interpolated)
        p_interp = max((p for r, p in rec_prec if r >= recall), default=0.0)
        ap += (recall - prev_recall) * p_interp
        prev_recall = recall
    return ap


def _ap_for_class(preds: list[tuple[int, dict]], gts_by_img: dict[int, list[dict]],
                  iou_thr: float) -> float:
    """AP for one class at one IoU threshold. preds = [(img_idx, pred)] sorted by
    conf desc; gts_by_img = {img_idx: [gt,...]}."""
    n_gt = sum(len(v) for v in gts_by_img.values())
    if n_gt == 0:
        return 0.0
    used: dict[int, list[bool]] = {i: [False] * len(v) for i, v in gts_by_img.items()}
    matched, confs = [], []
    for img_idx, p in preds:
        gts = gts_by_img.get(img_idx, [])
        best_j, best_iou = -1, iou_thr
        for j, g in enumerate(gts):
            if used[img_idx][j]:
                continue
            v = iou_xywh(_xyxy(p["box"]), _xyxy(g["box"]))
            if v >= best_iou:
                best_iou, best_j = v, j
        hit = best_j >= 0
        if hit:
            used[img_idx][best_j] = True
        matched.append(1 if hit else 0)
        confs.append(float(p.get("conf", 0.0)))
    return _average_precision(matched, confs, n_gt)


def compute_metrics(samples: list[dict], iou_thr: float = DEFAULT_IOU) -> dict:
    """``samples`` = ``[{"preds":[{box,cls,conf}], "gt":[{box,cls}]}, ...]``
    (boxes normalized xywh; classes already model-aligned)."""
    # aggregate TP/FP/FN at the headline IoU
    tp = fp = fn = 0
    classes = set()
    for s in samples:
        classes.update(p["cls"] for p in s["preds"])
        classes.update(g["cls"] for g in s["gt"])
        # per-class within the image so a car pred cannot match a person GT
        cset = {p["cls"] for p in s["preds"]} | {g["cls"] for g in s["gt"]}
        for c in cset:
            ptp, pfp, pfn = _match_counts(
                [p for p in s["preds"] if p["cls"] == c],
                [g for g in s["gt"] if g["cls"] == c],
                iou_thr,
            )
            tp += ptp; fp += pfp; fn += pfn

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    agreement = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0

    # mAP over IoU thresholds, per class
    ap_by_iou: dict[float, list[float]] = {}
    per_class_ap50: dict[str, float] = {}
    eval_classes = sorted(c for c in classes
                          if any(g["cls"] == c for s in samples for g in s["gt"]))
    for t in MAP_IOUS:
        aps = []
        for c in eval_classes:
            preds_c = []
            gts_by_img = {}
            for i, s in enumerate(samples):
                gts_c = [g for g in s["gt"] if g["cls"] == c]
                if gts_c:
                    gts_by_img[i] = gts_c
                for p in s["preds"]:
                    if p["cls"] == c:
                        preds_c.append((i, p))
            preds_c.sort(key=lambda ip: float(ip[1].get("conf", 0.0)), reverse=True)
            ap = _ap_for_class(preds_c, gts_by_img, t)
            aps.append(ap)
            if t == 0.5:
                per_class_ap50[c] = round(ap, 4)
        ap_by_iou[t] = aps

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    mAP = _mean([_mean(ap_by_iou[t]) for t in MAP_IOUS])
    ap50 = _mean(ap_by_iou.get(0.5, []))
    ap75 = _mean(ap_by_iou.get(0.75, []))

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "agreement": round(agreement, 4),
        "mAP": round(mAP, 4),
        "AP50": round(ap50, 4),
        "AP75": round(ap75, 4),
        "tp": tp, "fp": fp, "fn": fn,
        "n_images": len(samples),
        "n_pred": sum(len(s["preds"]) for s in samples),
        "n_gt": sum(len(s["gt"]) for s in samples),
        "per_class_ap50": per_class_ap50,
        "iou_thr": iou_thr,
    }
