from __future__ import annotations

import numpy as np


class ZoeDepthEstimator:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None

    def load(self) -> None:
        import torch

        self.model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True).to(self.device).eval()

    def estimate_depth_ft(self, image_bgr: np.ndarray) -> np.ndarray:
        if self.model is None:
            self.load()
        import cv2
        from PIL import Image

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        depth_m = self.model.infer_pil(pil)  # numpy float depth in meters
        depth_m = np.asarray(depth_m, dtype=np.float32)
        depth_ft = depth_m * 3.28084
        return depth_ft


def median_depth_in_bbox(depth_ft: np.ndarray, bbox_xyxy: list[int]) -> float:
    x1, y1, x2, y2 = bbox_xyxy
    h, w = depth_ft.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    roi = depth_ft[y1:y2, x1:x2]
    if roi.size == 0:
        return float(np.median(depth_ft))
    return float(np.median(roi))

