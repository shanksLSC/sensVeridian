from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json
import cv2
import numpy as np

from .config import MODELS, SETTINGS
from .hashing import hash_decoded_image, hash_file
from .store.duck import DuckStore
from .store.faces_registry import FaceRegistry
from .runners.base import RunnerOutput
from .runners.amod import AMODRunner
from .runners.qrcode import QRCodeRunner
from .runners.face_detection import FaceDetectionRunner
from .runners.face_recognition import FaceRecognitionRunner


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class IngestResult:
    images_seen: int = 0
    images_ingested: int = 0
    predictions_written: int = 0


def _to_json_safe(value):
    if isinstance(value, dict):
        return {k: _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


class Orchestrator:
    def __init__(self, store: DuckStore, registry: FaceRegistry):
        self.store = store
        self.registry = registry
        self.runners = {
            "amod": AMODRunner(str(MODELS.amod)),
            "qrcode": QRCodeRunner(str(MODELS.qrcode)),
            "fd": FaceDetectionRunner(str(MODELS.fd)),
            "fr": FaceRecognitionRunner(str(MODELS.fr), registry=registry, threshold=SETTINGS.face_match_threshold),
        }

    def _ordered_models(self, selected: set[str]) -> list[str]:
        # Fixed topological order for current graph.
        order = ["amod", "qrcode", "fd", "fr"]
        return [m for m in order if m in selected]

    def _iter_images(self, path: Path) -> Iterable[Path]:
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            yield path
            return
        for p in sorted(path.rglob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                yield p

    def ingest(self, image_root: Path, run_id: str, selected_models: set[str], skip_existing: bool = True) -> IngestResult:
        result = IngestResult()
        self.store.ensure_run(run_id=run_id)
        for mid in selected_models:
            runner = self.runners[mid]
            self.store.upsert_model(
                model_id=runner.model_id,
                display_name=runner.display_name,
                version=runner.version,
                weights_path=runner.weights_path,
                weights_sha=hash_file(Path(runner.weights_path)) if Path(runner.weights_path).exists() else "",
            )
        ordered = self._ordered_models(selected_models)
        for img_path in self._iter_images(image_root):
            result.images_seen += 1
            image_id, w, h = hash_decoded_image(img_path)
            if skip_existing:
                exists = self.store.query_df(
                    f"select count(*) as c from predictions_summary where image_id='{image_id}' and run_id='{run_id}'"
                )["c"].iloc[0]
                if exists >= len(ordered):
                    continue
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            self.store.upsert_image(image_id=image_id, path=str(img_path), width=w, height=h)
            deps: dict[str, RunnerOutput] = {}
            for mid in ordered:
                runner = self.runners[mid]
                out = runner.predict(image, deps)
                deps[mid] = out
                self.store.upsert_summary(image_id, run_id, mid, out.summary)
                raw_payload = dict(out.raw)
                if mid == "fd" and "crops" in raw_payload:
                    # Crops are useful in-memory for FR but should not be persisted as huge arrays.
                    raw_payload["crop_count"] = len(raw_payload["crops"])
                    raw_payload.pop("crops")
                self.store.upsert_raw(image_id, run_id, mid, _to_json_safe(raw_payload))
                result.predictions_written += 1
            result.images_ingested += 1
        return result

