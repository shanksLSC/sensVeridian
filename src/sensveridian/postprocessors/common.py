"""Shared post-processing math — NumPy port of the MLHILS reference
``common.c`` / ``nms_decoder.c`` float path (sigmoid, softmax, IoU, greedy NMS).

These mirror the C float-mode helpers so the Python interpreters decode model
heads identically to the on-device post-processors. See
``/data3/ssharma8/projects/lattice-internal/postptocessors_MLHILS/src``.
"""
from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.clip(np.sum(e, axis=axis, keepdims=True), 1e-12, None)


def iou_xyxy(a, b) -> float:
    """IoU of two ``[x1, y1, x2, y2]`` boxes (pp_iou_f)."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = ix2 - ix1
    ih = iy2 - iy1
    if iw <= 0.0 or ih <= 0.0:
        return 0.0
    inter = iw * ih
    denom = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / denom if denom > 1e-6 else 0.0


def nms_greedy(boxes: list, scores: list, iou_threshold: float, max_keep: int) -> list[int]:
    """Greedy NMS over score-descending boxes (pp_nms_greedy_f). Returns kept
    indices into the (already score-sorted) ``boxes``."""
    keep: list[int] = []
    for i in range(len(boxes)):
        if len(keep) >= max_keep:
            break
        if all(iou_xyxy(boxes[i], boxes[k]) < iou_threshold for k in keep):
            keep.append(i)
    return keep


def chw(arr: np.ndarray) -> np.ndarray:
    """Reorder a Keras NHWC output ``(1, H, W, C)`` (or ``(H, W, C)``) to CHW
    ``(C, H, W)`` so channel indexing matches the C reference (``chw_index``)."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 4:
        a = a[0]
    if a.ndim != 3:
        raise ValueError(f"expected HWC tensor, got shape {a.shape}")
    return np.transpose(a, (2, 0, 1))  # HWC -> CHW


def sort_desc(scores: list[float]) -> list[int]:
    """Indices that sort ``scores`` descending (stable, matches insertion sort)."""
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)


def to_normalized_detection(class_id: int, x1: float, y1: float, x2: float, y2: float,
                            conf: float, img_w: int, img_h: int, **extra) -> dict:
    """Pack a decoded pixel-space box into the runner detection contract with a
    bbox normalized to ``[0, 1]`` (xyxy) against the model input dims."""
    det = {
        "bbox": [x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h],
        "conf": float(conf),
        "class_id": int(class_id),
    }
    det.update(extra)
    return det
