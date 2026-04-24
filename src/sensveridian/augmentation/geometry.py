from __future__ import annotations

import numpy as np
import cv2


def scale_for_delta(d_initial_ft: float, delta_ft: float, min_scale: float = 0.1) -> float:
    d_new = max(d_initial_ft + delta_ft, 1e-3)
    s = d_initial_ft / d_new
    return float(max(min_scale, min(1.0, s)))


def depth_sort_indices(d_initial_list: list[float], delta_ft: float) -> list[int]:
    values = [(i, d + delta_ft) for i, d in enumerate(d_initial_list)]
    values.sort(key=lambda x: x[1], reverse=True)  # farther first
    return [i for i, _ in values]


def extract_rgba_from_mask(image_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    alpha = (mask_u8 > 0).astype(np.uint8) * 255
    rgba = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
    rgba[..., 3] = alpha
    return rgba


def scaled_object_rgba(obj_rgba: np.ndarray, scale: float) -> np.ndarray:
    h, w = obj_rgba.shape[:2]
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(obj_rgba, (nw, nh), interpolation=cv2.INTER_LINEAR)


def paste_rgba_center(bg_bgr: np.ndarray, obj_rgba: np.ndarray, center_xy: tuple[int, int]) -> np.ndarray:
    out = bg_bgr.copy()
    oh, ow = obj_rgba.shape[:2]
    cx, cy = center_xy
    x1 = cx - ow // 2
    y1 = cy - oh // 2
    x2 = x1 + ow
    y2 = y1 + oh

    bh, bw = out.shape[:2]
    ix1, iy1 = max(0, x1), max(0, y1)
    ix2, iy2 = min(bw, x2), min(bh, y2)
    if ix1 >= ix2 or iy1 >= iy2:
        return out
    ox1, oy1 = ix1 - x1, iy1 - y1
    ox2, oy2 = ox1 + (ix2 - ix1), oy1 + (iy2 - iy1)

    fg = obj_rgba[oy1:oy2, ox1:ox2, :3].astype(np.float32)
    alpha = (obj_rgba[oy1:oy2, ox1:ox2, 3:4].astype(np.float32) / 255.0)
    bg = out[iy1:iy2, ix1:ix2].astype(np.float32)
    comp = fg * alpha + bg * (1.0 - alpha)
    out[iy1:iy2, ix1:ix2] = comp.astype(np.uint8)
    return out

