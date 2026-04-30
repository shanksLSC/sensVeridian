from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import json
import cv2
import numpy as np
from tqdm.auto import tqdm

from .config import MODELS, SETTINGS
from .hashing import hash_decoded_image, hash_file
from .store.duck import DuckStore
from .store.faces_registry import FaceRegistry
from .runners.base import RunnerOutput, set_conf_threshold
from .runners.amod import AMODRunner
from .runners.qrcode import QRCodeRunner
from .runners.face_detection import FaceDetectionRunner
from .runners.face_recognition import FaceRecognitionRunner


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FT_TO_M = 0.3048


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
    def __init__(self, store: DuckStore, registry: FaceRegistry, conf_threshold: Optional[float] = None):
        self.store = store
        self.registry = registry
        self.conf_threshold = conf_threshold
        self.runners = {
            "amod": AMODRunner(str(MODELS.amod)),
            "qrcode": QRCodeRunner(str(MODELS.qrcode)),
            "fd": FaceDetectionRunner(str(MODELS.fd)),
            "fr": FaceRecognitionRunner(str(MODELS.fr), registry=registry, threshold=SETTINGS.face_match_threshold),
        }
        if conf_threshold is not None:
            for runner in self.runners.values():
                set_conf_threshold(runner, conf_threshold)

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

    @staticmethod
    def _loads_json(value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return {}
        if isinstance(value, dict):
            return value
        return {}

    def _refresh_image_metadata(self, image_id: str) -> None:
        img_df = self.store.query_df(f"SELECT image_id, path, width, height FROM images WHERE image_id='{image_id}'")
        if img_df.empty:
            return
        img_row = img_df.iloc[0]

        aug_df = self.store.query_df(
            f"""
            SELECT parent_image_id, delta_ft
            FROM augmentations
            WHERE augmented_image_id='{image_id}'
            LIMIT 1
            """
        )
        is_augmented = not aug_df.empty
        parent_image_id = None
        augmented_distance_m = None
        if is_augmented:
            parent_image_id = aug_df.iloc[0]["parent_image_id"]
            delta_ft = float(aug_df.iloc[0]["delta_ft"])
            augmented_distance_m = round(delta_ft * FT_TO_M, 6)

        pred_df = self.store.query_df(
            f"""
            SELECT
              s.run_id,
              s.model_id,
              s.present,
              s.count,
              s.extras,
              r.payload,
              m.version,
              m.weights_path,
              m.weights_sha
            FROM predictions_summary s
            LEFT JOIN predictions_raw r
              ON r.image_id=s.image_id AND r.run_id=s.run_id AND r.model_id=s.model_id
            LEFT JOIN models m
              ON m.model_id=s.model_id
            WHERE s.image_id='{image_id}'
            ORDER BY s.run_id, s.model_id
            """
        )

        model_runs = []
        for _, row in pred_df.iterrows():
            model_id = row["model_id"]
            extras = self._loads_json(row["extras"])
            payload = self._loads_json(row["payload"])
            runner = self.runners.get(model_id)
            raw_conf = getattr(runner, "conf_threshold", None) if runner is not None else None
            # Only carry numeric conf thresholds; keeps the JSON serializable when
            # runners are mocked or expose non-numeric placeholders.
            conf = float(raw_conf) if isinstance(raw_conf, (int, float)) and not isinstance(raw_conf, bool) else None
            model_runs.append(
                {
                    "model_id": model_id,
                    "version": row["version"],
                    "weights_path": row["weights_path"],
                    "weights_sha": row["weights_sha"],
                    "run_id": row["run_id"],
                    "conf_threshold": conf,
                    "present": bool(row["present"]),
                    "n_detections": int(row["count"]) if row["count"] is not None else 0,
                    "class_histogram": extras.get("class_histogram", {}),
                    "detections": payload.get("detections", []),
                }
            )

        metadata = {
            "image_id": img_row["image_id"],
            "path": img_row["path"],
            "width": int(img_row["width"]),
            "height": int(img_row["height"]),
            "augmented_flag": is_augmented,
            "augmented_distance_m": augmented_distance_m,
            "parent_image_id": parent_image_id,
            "model_runs": model_runs,
        }
        self.store.upsert_image_metadata(image_id=image_id, metadata=_to_json_safe(metadata))

    def refresh_metadata(
        self,
        run_id: Optional[str] = None,
        image_id: Optional[str] = None,
        progress: bool = True,
    ) -> int:
        if image_id is not None:
            self._refresh_image_metadata(image_id=image_id)
            return 1

        where = []
        if run_id is not None:
            run_id_esc = run_id.replace("'", "''")
            where.append(f"run_id='{run_id_esc}'")
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        df = self.store.query_df(f"SELECT DISTINCT image_id FROM predictions_summary {where_sql}")
        image_ids = df["image_id"].tolist()
        desc = f"refresh-metadata[{run_id}]" if run_id else "refresh-metadata"
        iterator = tqdm(image_ids, desc=desc, unit="img", disable=not progress, leave=True)
        for iid in iterator:
            self._refresh_image_metadata(image_id=iid)
        return len(image_ids)

    def ingest(
        self,
        image_root: Path,
        run_id: str,
        selected_models: set[str],
        skip_existing: bool = True,
        progress: bool = True,
        progress_leave: bool = True,
    ) -> IngestResult:
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
        # Materialize the image list so tqdm can show a total and ETA.
        image_paths = list(self._iter_images(image_root))
        bar = tqdm(
            image_paths,
            desc=f"ingest[{run_id}]",
            unit="img",
            disable=not progress,
            leave=progress_leave,
        )
        for img_path in bar:
            result.images_seen += 1
            image_id, w, h = hash_decoded_image(img_path)
            if skip_existing:
                exists = self.store.query_df(
                    f"select count(*) as c from predictions_summary where image_id='{image_id}' and run_id='{run_id}'"
                )["c"].iloc[0]
                if exists >= len(ordered):
                    bar.set_postfix(skipped=result.images_seen - result.images_ingested, refresh=False)
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
            self._refresh_image_metadata(image_id=image_id)
            result.images_ingested += 1
            bar.set_postfix(ingested=result.images_ingested, writes=result.predictions_written, refresh=False)
        return result

