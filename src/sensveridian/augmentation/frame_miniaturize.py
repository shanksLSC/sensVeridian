from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

from ..config import SETTINGS
from ..hashing import hash_decoded_image
from ..store.duck import DuckStore
from ..orchestrator import Orchestrator
from .depth import ZoeDepthEstimator, median_depth_in_bbox
from .camera import CameraProfile
from .calibration import CalibratedDistanceEstimator
from .manual_distance import DistanceOverrides

SUPPORTED_PAD_MODES = {"black", "replicate", "reflect"}


@dataclass
class DetectionDistance:
    model_id: str
    detection_idx: int
    bbox: list[int]
    depth_ft: float
    source: str  # manual | calib | zoe


def scale_for_distance_shift(d0_ft: float, delta_ft: float) -> float:
    if d0_ft <= 0:
        raise ValueError("d0_ft must be > 0")
    if delta_ft < 0:
        raise ValueError("delta_ft must be >= 0")
    return float(d0_ft / (d0_ft + delta_ft))


def miniaturize_frame(image_bgr: np.ndarray, scale: float, pad_mode: str = "black") -> np.ndarray:
    if scale <= 0:
        raise ValueError("scale must be > 0")
    mode = pad_mode.strip().lower()
    if mode not in SUPPORTED_PAD_MODES:
        raise ValueError(f"Unsupported pad_mode '{pad_mode}'. Use one of {sorted(SUPPORTED_PAD_MODES)}")

    h, w = image_bgr.shape[:2]
    if abs(scale - 1.0) < 1e-9:
        return image_bgr.copy()

    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image_bgr, (nw, nh), interpolation=interp)

    left = max(0, (w - nw) // 2)
    right = max(0, w - nw - left)
    top = max(0, (h - nh) // 2)
    bottom = max(0, h - nh - top)

    if mode == "black":
        canvas = np.zeros_like(image_bgr)
        canvas[top : top + nh, left : left + nw] = resized
        return canvas
    if mode == "replicate":
        return cv2.copyMakeBorder(resized, top, bottom, left, right, borderType=cv2.BORDER_REPLICATE)
    return cv2.copyMakeBorder(resized, top, bottom, left, right, borderType=cv2.BORDER_REFLECT_101)


def _extract_detections(raw_payload: dict) -> list[tuple[list[int], int | None]]:
    out: list[tuple[list[int], int | None]] = []
    for d in raw_payload.get("detections", []):
        bbox = d.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        class_id = d.get("class_id")
        if class_id is not None:
            try:
                class_id = int(class_id)
            except (TypeError, ValueError):
                class_id = None
        out.append(([int(v) for v in bbox], class_id))
    return out


class FrameMiniaturizer:
    def __init__(
        self,
        store: DuckStore,
        orchestrator: Orchestrator,
        device: str = SETTINGS.device,
        camera_profile: CameraProfile | None = None,
    ):
        self.store = store
        self.orchestrator = orchestrator
        self.camera_profile = camera_profile
        self.depth = ZoeDepthEstimator(device=device) if camera_profile is None else None
        self.calibrated_depth = (
            CalibratedDistanceEstimator(camera_profile) if camera_profile is not None else None
        )

    def _estimate_detection_distances(
        self,
        *,
        image_bgr: np.ndarray,
        image_path: Path,
        image_id: str,
        source_models: list[str],
        overrides: DistanceOverrides,
    ) -> list[DetectionDistance]:
        h, w = image_bgr.shape[:2]
        df = self.store.query_df(
            f"""
            SELECT model_id, payload
            FROM predictions_raw
            WHERE image_id = '{image_id}'
              AND model_id IN ({",".join([f"'{m}'" for m in source_models])})
            """
        )

        refs: list[tuple[str, int, list[int], int | None]] = []
        for _, row in df.iterrows():
            model_id = row["model_id"]
            payload = row["payload"]
            if isinstance(payload, str):
                import json

                payload = json.loads(payload)
            detections = _extract_detections(payload)
            for idx, (bbox, class_id) in enumerate(detections):
                x1, y1, x2, y2 = bbox
                x1 = max(0, min(w - 1, x1))
                x2 = max(1, min(w, x2))
                y1 = max(0, min(h - 1, y1))
                y2 = max(1, min(h, y2))
                refs.append((model_id, idx, [x1, y1, x2, y2], class_id))

        if not refs:
            return []

        detection_refs = [(m, idx) for (m, idx, _, _) in refs]
        needs_zoe = (
            self.calibrated_depth is None
            and not overrides.covers_all(
                image_path=image_path,
                image_id=image_id,
                detection_refs=detection_refs,
            )
            and self.depth is not None
        )
        depth_ft_map = self.depth.estimate_depth_ft(image_bgr) if needs_zoe else None

        out: list[DetectionDistance] = []
        for model_id, detection_idx, bbox, class_id in refs:
            manual = overrides.lookup(
                image_path=image_path,
                image_id=image_id,
                model_id=model_id,
                detection_idx=detection_idx,
            )
            if manual is not None:
                d_ft = float(manual)
                source = "manual"
            elif self.calibrated_depth is not None:
                real_size_m = overrides.real_size_lookup(
                    image_path=image_path,
                    image_id=image_id,
                    model_id=model_id,
                    detection_idx=detection_idx,
                )
                try:
                    d_ft = self.calibrated_depth.distance_ft(
                        image_w=w,
                        image_h=h,
                        bbox_xyxy=bbox,
                        model_id=model_id,
                        class_id=class_id,
                        real_size_m=real_size_m,
                    )
                    source = "calib"
                except ValueError:
                    continue
            elif depth_ft_map is not None:
                d_ft = median_depth_in_bbox(depth_ft_map, bbox)
                source = "zoe"
            else:
                continue

            self.store.upsert_depth_stat(
                image_id=image_id,
                model_id=model_id,
                detection_idx=detection_idx,
                bbox_xyxy=bbox,
                d_initial_ft=d_ft,
                source=source,
            )
            out.append(
                DetectionDistance(
                    model_id=model_id,
                    detection_idx=detection_idx,
                    bbox=bbox,
                    depth_ft=float(d_ft),
                    source=source,
                )
            )
        return out

    def augment_image(
        self,
        image_path: Path,
        run_id: str,
        d_max_ft: Optional[float],
        step_ft: float,
        source_models: list[str],
        out_dir: Path,
        auto_run_oracle: bool = False,
        overrides: DistanceOverrides | None = None,
        pad_mode: str = "black",
        n_steps: Optional[int] = None,
        rerun_models: set[str] | None = None,
    ) -> int:
        overrides = overrides or DistanceOverrides.empty()
        image_id, _, _ = hash_decoded_image(image_path)
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            return 0

        distances = self._estimate_detection_distances(
            image_bgr=image_bgr,
            image_path=image_path,
            image_id=image_id,
            source_models=source_models,
            overrides=overrides,
        )
        if distances:
            d0_ft = float(np.median([d.depth_ft for d in distances]))
        elif overrides.global_ft is not None:
            d0_ft = float(overrides.global_ft)
        else:
            return 0

        n_manual = sum(1 for d in distances if d.source == "manual")
        n_calib = sum(1 for d in distances if d.source == "calib")
        n_zoe = sum(1 for d in distances if d.source == "zoe")

        out_dir.mkdir(parents=True, exist_ok=True)
        steps_written = 0
        delta = float(step_ft)
        step_index = 0
        while True:
            if n_steps is not None and step_index >= n_steps:
                break
            if n_steps is None:
                if d_max_ft is None:
                    raise ValueError("d_max_ft is required when n_steps is not set")
                if d0_ft + delta > d_max_ft + 1e-6:
                    break
            scale = scale_for_distance_shift(d0_ft=d0_ft, delta_ft=delta)
            canvas = miniaturize_frame(image_bgr, scale=scale, pad_mode=pad_mode)
            tag = f"{image_path.stem}_mini_dist_{delta:.2f}ft".replace(".", "p")
            out_path = out_dir / f"{tag}{image_path.suffix.lower()}"
            cv2.imwrite(str(out_path), canvas)

            aug_image_id, _, _ = hash_decoded_image(out_path)
            self.store.insert_augmentation(
                augmented_image_id=aug_image_id,
                parent_image_id=image_id,
                method="frame_miniaturize",
                step_index=int(round(delta / step_ft)),
                delta_ft=delta,
                params={
                    "d0_ft": d0_ft,
                    "scale": scale,
                    "pad_mode": pad_mode,
                    "source_models": source_models,
                    "n_distance_refs": len(distances),
                    "n_manual_distance": n_manual,
                    "n_calib_distance": n_calib,
                    "n_zoe_distance": n_zoe,
                },
            )
            steps_written += 1
            delta += step_ft
            step_index += 1

        if auto_run_oracle and steps_written > 0:
            self.orchestrator.ingest(
                image_root=out_dir,
                run_id=run_id,
                selected_models=rerun_models or {"amod", "qrcode", "fd", "fr"},
                skip_existing=False,
                progress_leave=False,
            )
        return steps_written
