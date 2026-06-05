"""Face-detection interpreter — 2-anchor single-class decode (+ landmarks) + NMS.

Python port of ``detection.c`` ``face_detection_post_process`` (float path).
Output ``[H, W, 38]`` channel layout: 0-1 conf (2 anchors), 2-9 bbox deltas
(2 anchors x dx,dy,dw,dh), 10-29 landmark deltas (5 lm x (x,y) x 2 anchors),
30-35 head pose (unused), 36-37 class (single). Anchor box decode.
"""
from __future__ import annotations

import numpy as np

from . import common, constants as K


def interpret_face(outputs, img_w: int, img_h: int, conf_threshold: float | None = None) -> list[dict]:
    conf_threshold = K.FD_CONF_THRESHOLD if conf_threshold is None else conf_threshold
    a = np.asarray(outputs[0] if isinstance(outputs, (list, tuple)) else outputs, dtype=np.float32)
    if a.ndim == 4:
        a = a[0]
    H, W, _ = a.shape
    nA = K.FD_ANCHOR_PER_GRID
    num_conf = nA
    lnd_pos = num_conf + 4 * nA  # 10
    nL = K.FD_NUM_LANDMARKS

    candidates: list[tuple] = []
    for anc in range(nA):
        conf = common.sigmoid(a[..., anc])
        aw, ah = K.FD_ANCHOR_W[anc], K.FD_ANCHOR_H[anc]
        bbox_base = num_conf + anc * 4
        lm_base = lnd_pos + anc * nL * 2
        ys, xs = np.nonzero(conf >= conf_threshold)
        for gy, gx in zip(ys.tolist(), xs.tolist()):
            dx, dy, dw, dh = (float(v) for v in a[gy, gx, bbox_base:bbox_base + 4])
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
            lms = []
            for lm in range(nL):
                ldx = float(a[gy, gx, lm_base + lm * 2 + 0])
                ldy = float(a[gy, gx, lm_base + lm * 2 + 1])
                lms.append([(acx + ldx * aw) / img_w, (acy + ldy * ah) / img_h])
            candidates.append((x1, y1, x2, y2, float(conf[gy, gx]), lms))

    if not candidates:
        return []
    order = common.sort_desc([c[4] for c in candidates])
    boxes = [candidates[i][:4] for i in order]
    keep = common.nms_greedy(boxes, [candidates[i][4] for i in order],
                             K.FD_NMS_IOU_THRESHOLD, K.FD_MAX_DETECTIONS)
    dets = []
    for k in keep:
        x1, y1, x2, y2, conf, lms = candidates[order[k]]
        dets.append(common.to_normalized_detection(0, x1, y1, x2, y2, conf, img_w, img_h, landmarks=lms))
    return dets
