from __future__ import annotations

import numpy as np
import cv2


def dof_blur(image_bgr: np.ndarray, strength: float) -> np.ndarray:
    k = int(max(1, round(strength)))
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(image_bgr, (k, k), 0)


def atmospheric_haze(image_bgr: np.ndarray, strength: float, haze_color: tuple[int, int, int] = (220, 220, 220)) -> np.ndarray:
    alpha = float(max(0.0, min(0.35, strength)))
    haze = np.full_like(image_bgr, haze_color, dtype=np.uint8)
    out = cv2.addWeighted(image_bgr, 1.0 - alpha, haze, alpha, 0.0)
    return out

