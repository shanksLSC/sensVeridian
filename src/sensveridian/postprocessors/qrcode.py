"""QR interpreter — 4-anchor single-class decode + NMS.

Python port of ``qrcode.c`` (float path). Output ``[H, W, 36]`` channel layout:
0-3 anchor confidences, 4-19 bbox deltas (4 anchors x dx,dy,dw,dh), 20-31 angle
(unused), 32-35 class (single class). Anchor box decode (centre + delta*anchor).
"""
from __future__ import annotations

import numpy as np

from . import common, constants as K


def interpret(outputs, img_w: int, img_h: int, conf_threshold: float | None = None) -> list[dict]:
    conf_threshold = K.QR_CONF_THRESHOLD if conf_threshold is None else conf_threshold
    a = np.asarray(outputs[0] if isinstance(outputs, (list, tuple)) else outputs, dtype=np.float32)
    if a.ndim == 4:
        a = a[0]
    H, W, _ = a.shape
    nA = K.QR_ANCHOR_PER_GRID

    candidates: list[tuple] = []
    for anc in range(nA):
        conf = common.sigmoid(a[..., anc])
        aw = ah = K.QR_ANCHOR_SIZES[anc]
        base = 4 + anc * 4
        ys, xs = np.nonzero(conf >= conf_threshold)
        for gy, gx in zip(ys.tolist(), xs.tolist()):
            dx, dy, dw, dh = (float(v) for v in a[gy, gx, base:base + 4])
            acx = (gx + 1) * img_w / (W + 1)
            acy = (gy + 1) * img_h / (H + 1)
            bcx = acx + dx * aw
            bcy = acy + dy * ah
            bw = aw * dw
            bh = ah * dh
            x1 = max(0.0, bcx - bw / 2.0)
            y1 = max(0.0, bcy - bh / 2.0)
            x2 = min(float(img_w - 1), bcx + bw / 2.0)
            y2 = min(float(img_h - 1), bcy + bh / 2.0)
            if x2 <= x1 or y2 <= y1:
                continue
            candidates.append((x1, y1, x2, y2, float(conf[gy, gx]), 0))

    if not candidates:
        return []
    order = common.sort_desc([c[4] for c in candidates])
    boxes = [candidates[i][:4] for i in order]
    keep = common.nms_greedy(boxes, [candidates[i][4] for i in order],
                             K.QR_NMS_IOU_THRESHOLD, K.QR_MAX_DETECTIONS)
    return [
        common.to_normalized_detection(0, *candidates[order[k]][:4], candidates[order[k]][4], img_w, img_h)
        for k in keep
    ]
