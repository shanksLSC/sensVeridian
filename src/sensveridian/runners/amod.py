from __future__ import annotations

from collections import Counter
import numpy as np
from .base import RunnerOutput, Summary
from .common import load_sensai_h5_model, preprocess_for_model, as_list_of_arrays, extract_detection_candidates


class AMODRunner:
    model_id = "amod"
    display_name = "AutomotiveMultiObjectDetection"
    version = "8.2.0"
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
        x = preprocess_for_model(image_bgr, self.input_spec)
        pred = self.model.predict(x, verbose=0)
        outputs = as_list_of_arrays(pred)
        dets = extract_detection_candidates(outputs, conf_threshold=self.conf_threshold)
        class_hist = Counter(d["class_id"] for d in dets)
        summary = Summary(
            present=len(dets) > 0,
            count=len(dets),
            extras={"class_histogram": {str(k): v for k, v in class_hist.items()}},
        )
        return RunnerOutput(summary=summary, raw={"detections": dets, "output_shapes": [list(o.shape) for o in outputs]})

