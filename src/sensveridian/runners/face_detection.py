from __future__ import annotations

import cv2
import numpy as np
from .base import RunnerOutput, Summary
from .common import load_sensai_h5_model, preprocess_for_model, as_list_of_arrays, safe_bbox_xyxy
from ..postprocessors import interpret as interpret_outputs


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
        # Decode via the face-detection interpreter (2-anchor head + landmarks).
        # Boxes come back normalized; scale to the original image for cropping.
        dets = interpret_outputs("fd", outputs, self.input_spec[1], self.input_spec[0],
                                 self.conf_threshold)
        face_dets = []
        face_crops = []
        for d in dets:
            nx1, ny1, nx2, ny2 = d["bbox"]  # normalized xyxy
            bbox = safe_bbox_xyxy([nx1 * iw, ny1 * ih, nx2 * iw, ny2 * ih], iw, ih)
            crop = image_bgr[bbox[1] : bbox[3], bbox[0] : bbox[2]]
            if crop.size == 0:
                continue
            face_dets.append({"bbox": bbox, "conf": d["conf"], "landmarks": d.get("landmarks")})
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

