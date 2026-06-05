"""AMOD interpreter — FCOS-style multi-object decode + NMS.

Python port of ``multiobject.c`` (float path). The model emits 6 tensors: a
``[H, W, 4]`` bbox head and a ``[H, W, num_classes+1]`` logits head at each of 3
scales (strides 8/16/32). Logits channel 0 is objectness; 1.. are class scores
(softmax). FCOS box: corners are distances (l, t, r, b) from the cell centre.
"""
from __future__ import annotations

import numpy as np

from . import common, constants as K


def _decode_scale(bbox: np.ndarray, logits: np.ndarray, stride: int,
                  img_w: int, img_h: int, conf_threshold: float,
                  max_out: float, num_classes: int) -> list[tuple]:
    """Decode one FCOS scale -> list of (x1, y1, x2, y2, conf, cls) in pixels."""
    H, W = bbox.shape[0], bbox.shape[1]
    obj = common.sigmoid(logits[..., 0])                       # (H, W)
    cls = common.softmax(logits[..., 1:1 + num_classes], axis=-1)  # (H, W, C)
    best_cls = np.argmax(cls, axis=-1)
    best_p = np.max(cls, axis=-1)
    conf = obj * best_p
    ys, xs = np.nonzero(conf >= conf_threshold)
    out = []
    for gy, gx in zip(ys.tolist(), xs.tolist()):
        dl, dt, dr, db = (float(v) for v in bbox[gy, gx, :4])
        cx = stride / 2.0 + stride * gx
        cy = stride / 2.0 + stride * gy
        x1 = cx - dl * img_w / max_out
        y1 = cy - dt * img_w / max_out
        x2 = cx + dr * img_w / max_out
        y2 = cy + db * img_w / max_out
        x1 = max(0.0, x1); y1 = max(0.0, y1)
        x2 = min(float(img_w - 1), x2); y2 = min(float(img_h - 1), y2)
        if x2 <= x1 or y2 <= y1:
            continue
        out.append((x1, y1, x2, y2, float(conf[gy, gx]), int(best_cls[gy, gx])))
    return out


def interpret(outputs, img_w: int, img_h: int, conf_threshold: float | None = None) -> list[dict]:
    conf_threshold = K.MOD_CONF_THRESHOLD if conf_threshold is None else conf_threshold
    num_classes = K.MOD_NUM_CLASSES
    arrs = [np.asarray(o)[0] if np.asarray(o).ndim == 4 else np.asarray(o) for o in outputs]

    # pair bbox (C==4) with logits (C==num_classes+1) by grid size
    bbox_by_grid, logit_by_grid = {}, {}
    for a in arrs:
        H, W, C = a.shape
        if C == 4:
            bbox_by_grid[(H, W)] = a
        elif C == num_classes + 1:
            logit_by_grid[(H, W)] = a

    candidates: list[tuple] = []
    for (gh, gw, stride) in K.MOD_SCALES:
        bbox = bbox_by_grid.get((gh, gw))
        logits = logit_by_grid.get((gh, gw))
        if bbox is None or logits is None:
            continue
        candidates.extend(
            _decode_scale(bbox, logits, stride, img_w, img_h, conf_threshold,
                          K.MOD_MAX_FPGA_OUTPUT, num_classes)
        )

    if not candidates:
        return []
    order = common.sort_desc([c[4] for c in candidates])
    boxes = [candidates[i][:4] for i in order]
    scores = [candidates[i][4] for i in order]
    keep = common.nms_greedy(boxes, scores, K.MOD_NMS_IOU_THRESHOLD, K.MOD_MAX_DETECTIONS)
    dets = []
    for k in keep:
        x1, y1, x2, y2, conf, cls = candidates[order[k]]
        dets.append(common.to_normalized_detection(cls, x1, y1, x2, y2, conf, img_w, img_h))
    return dets
