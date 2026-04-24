from __future__ import annotations

import numpy as np
import cv2
from .base import RunnerOutput, Summary
from .common import load_sensai_h5_model
from ..store.faces_registry import FaceRegistry


class FaceRecognitionRunner:
    model_id = "fr"
    display_name = "FaceRecognition"
    version = "8.1.1"
    depends_on: tuple[str, ...] = ("fd",)

    def __init__(self, weights_path: str, registry: FaceRegistry, threshold: float = 0.5):
        self.weights_path = weights_path
        self.registry = registry
        self.threshold = threshold
        self.model = None
        self.input_spec = (112, 112, 3)
        self.embedding_dim = 128

    def load(self) -> None:
        self.model = load_sensai_h5_model(self.weights_path)
        shape = self.model.input_shape
        self.input_spec = (int(shape[1]), int(shape[2]), int(shape[3]))
        out_shape = self.model.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        self.embedding_dim = int(out_shape[-1])

    def _embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        x = cv2.resize(crop_bgr, (self.input_spec[1], self.input_spec[0]), interpolation=cv2.INTER_LINEAR)
        if self.input_spec[2] == 1:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            x = np.expand_dims(x, axis=-1)
        else:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=0)
        y = self.model.predict(x, verbose=0)
        emb = np.asarray(y).reshape(-1).astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb

    def predict(self, image_bgr: np.ndarray, deps: dict[str, RunnerOutput]) -> RunnerOutput:
        if self.model is None:
            self.load()
        fd = deps.get("fd")
        if not fd:
            return RunnerOutput(summary=Summary(False, 0, {"n_FID": 0}), raw={"recognized": []})
        detections = fd.raw.get("detections", [])
        crops = fd.raw.get("crops", [])
        recognized = []
        n_id = 0
        for idx, crop in enumerate(crops):
            emb = self._embed(crop)
            pid, score = self.registry.match(emb, threshold=self.threshold)
            hit = pid is not None
            n_id += int(hit)
            bbox = detections[idx]["bbox"] if idx < len(detections) else None
            recognized.append(
                {
                    "bbox": bbox,
                    "matched_person_id": pid,
                    "score": score,
                    "embedding": emb.tolist(),
                }
            )
        summary = Summary(present=n_id > 0, count=n_id, extras={"n_FID": n_id, "face_candidates": len(crops)})
        return RunnerOutput(summary=summary, raw={"recognized": recognized})

