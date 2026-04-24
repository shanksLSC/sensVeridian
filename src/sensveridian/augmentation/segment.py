from __future__ import annotations

import numpy as np
import cv2


class SAMSegmenter:
    def __init__(self, checkpoint_path: str, model_type: str = "vit_b", device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.device = device
        self.predictor = None

    def load(self) -> None:
        from segment_anything import sam_model_registry, SamPredictor  # type: ignore

        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)

    def segment(self, image_bgr: np.ndarray, bboxes_xyxy: list[list[int]]) -> list[np.ndarray]:
        if self.predictor is None:
            self.load()
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)
        masks = []
        for bbox in bboxes_xyxy:
            box = np.array(bbox, dtype=np.float32)
            mask, _, _ = self.predictor.predict(box=box, multimask_output=False)
            masks.append(mask[0].astype(np.uint8))
        return masks


def union_masks(masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w), dtype=np.uint8)
    for m in masks:
        mm = (m > 0).astype(np.uint8)
        if mm.shape != out.shape:
            mm = cv2.resize(mm, (w, h), interpolation=cv2.INTER_NEAREST)
        out = np.maximum(out, mm)
    return out

