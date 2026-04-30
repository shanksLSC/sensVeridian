from __future__ import annotations

import cv2
import numpy as np
from .base import RunnerOutput, Summary
from .common import load_sensai_h5_model, preprocess_for_model, as_list_of_arrays, extract_detection_candidates


class QRCodeRunner:
    model_id = "qrcode"
    display_name = "QRCodeDetection"
    version = "final"
    depends_on: tuple[str, ...] = ()

    def __init__(self, weights_path: str, conf_threshold: float = 0.3):
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.model = None
        self.input_spec = (320, 320, 1)
        self.cv_qr = cv2.QRCodeDetector()

    def load(self) -> None:
        self.model = load_sensai_h5_model(self.weights_path)
        shape = self.model.input_shape
        self.input_spec = (int(shape[1]), int(shape[2]), int(shape[3]))

    def predict(self, image_bgr: np.ndarray, deps: dict[str, RunnerOutput]) -> RunnerOutput:
        if self.model is None:
            self.load()
        x = preprocess_for_model(image_bgr, self.input_spec)
        pred = self.model.predict(x, verbose=0)
        outputs = as_list_of_arrays(pred)
        dets = extract_detection_candidates(outputs, conf_threshold=self.conf_threshold)
        ok, decoded_info, points, _ = self.cv_qr.detectAndDecodeMulti(image_bgr)
        decoded = decoded_info if ok and decoded_info is not None else []
        summary = Summary(
            present=len(dets) > 0 or len(decoded) > 0,
            count=max(len(dets), len(decoded)),
            extras={"decoded_count": len([d for d in decoded if d])},
        )
        return RunnerOutput(
            summary=summary,
            raw={
                "detections": dets,
                "decoded_texts": [d for d in decoded if d],
                "points": points.tolist() if points is not None else [],
                "output_shapes": [list(o.shape) for o in outputs],
            },
        )

