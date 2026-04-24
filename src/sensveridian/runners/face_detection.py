from __future__ import annotations

import cv2
import numpy as np
from .base import RunnerOutput, Summary
from .common import load_sensai_h5_model, preprocess_for_model, as_list_of_arrays, extract_detection_candidates, safe_bbox_xyxy


class FaceDetectionRunner:
    model_id = "fd"
    display_name = "FaceDetection"
    version = "8.1.0"
    depends_on: tuple[str, ...] = ()

    def __init__(self, weights_path: str, conf_threshold: float = 0.3):
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.model = None
        self.input_spec = (320, 320, 3)

    def load(self) -> None:
        self.model = load_sensai_h5_model(self.weights_path)
        shape = self.model.input_shape
        self.input_spec = (int(shape[1]), int(shape[2]), int(shape[3]))

    def predict(self, image_bgr: np.ndarray, deps: dict[str, RunnerOutput]) -> RunnerOutput:
        if self.model is None:
            self.load()
        ih, iw = image_bgr.shape[:2]
        x = preprocess_for_model(image_bgr, self.input_spec)
        pred = self.model.predict(x, verbose=0)
        outputs = as_list_of_arrays(pred)
        dets = extract_detection_candidates(outputs, conf_threshold=self.conf_threshold)
        face_dets = []
        face_crops = []
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            x1 = x1 * (iw / self.input_spec[1]) if x2 <= 1.5 else x1
            x2 = x2 * (iw / self.input_spec[1]) if x2 <= 1.5 else x2
            y1 = y1 * (ih / self.input_spec[0]) if y2 <= 1.5 else y1
            y2 = y2 * (ih / self.input_spec[0]) if y2 <= 1.5 else y2
            bbox = safe_bbox_xyxy([x1, y1, x2, y2], iw, ih)
            crop = image_bgr[bbox[1] : bbox[3], bbox[0] : bbox[2]]
            if crop.size == 0:
                continue
            face_dets.append({"bbox": bbox, "conf": d["conf"]})
            face_crops.append(crop)
        summary = Summary(present=len(face_dets) > 0, count=len(face_dets), extras={"n_FD": len(face_dets)})
        return RunnerOutput(
            summary=summary,
            raw={
                "detections": face_dets,
                "crops": face_crops,
                "output_shapes": [list(o.shape) for o in outputs],
            },
        )

