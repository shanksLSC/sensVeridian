from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import uuid
import cv2
import numpy as np

from ..config import SETTINGS
from ..hashing import hash_decoded_image
from ..store.duck import DuckStore
from ..orchestrator import Orchestrator
from .depth import ZoeDepthEstimator, median_depth_in_bbox
from .camera import CameraProfile
from .calibration import CalibratedDistanceEstimator
from .segment import SAMSegmenter, union_masks
from .inpaint import LaMAInpainter
from .manual_distance import DistanceOverrides
from .geometry import (
    scale_for_delta,
    depth_sort_indices,
    extract_rgba_from_mask,
    scaled_object_rgba,
    paste_rgba_center,
)
from .effects import dof_blur, atmospheric_haze


@dataclass
class DetectionObject:
    model_id: str
    detection_idx: int
    bbox: list[int]
    center: tuple[int, int]
    depth_ft: float
    depth_source: str  # 'zoe', 'manual', or 'calib'
    rgba_obj: np.ndarray


def _extract_detections(raw_payload: dict) -> list[tuple[list[int], int | None]]:
    out = []
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


class DistanceAugmentor:
    def __init__(
        self,
        store: DuckStore,
        orchestrator: Orchestrator,
        sam_checkpoint: str,
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
        self.segmenter = SAMSegmenter(checkpoint_path=sam_checkpoint, model_type="vit_b", device=device)
        self.inpainter = LaMAInpainter(device=device, model_id="lama")

    def _build_objects(
        self,
        image_bgr: np.ndarray,
        image_path: Path,
        image_id: str,
        source_models: list[str],
        overrides: DistanceOverrides,
    ) -> tuple[list[DetectionObject], np.ndarray]:
        df = self.store.query_df(
            f"""
            SELECT model_id, payload
            FROM predictions_raw
            WHERE image_id = '{image_id}'
              AND model_id IN ({",".join([f"'{m}'" for m in source_models])})
            """
        )
        h, w = image_bgr.shape[:2]
        all_boxes: list[list[int]] = []
        boxed_refs: list[tuple[str, int, list[int], int | None]] = []
        for _, row in df.iterrows():
            model_id = row["model_id"]
            payload = row["payload"]
            if isinstance(payload, str):
                import json

                payload = json.loads(payload)
            detections = _extract_detections(payload)
            for idx, (bb, class_id) in enumerate(detections):
                x1, y1, x2, y2 = bb
                x1 = max(0, min(w - 1, x1))
                x2 = max(1, min(w, x2))
                y1 = max(0, min(h - 1, y1))
                y2 = max(1, min(h, y2))
                bb2 = [x1, y1, x2, y2]
                all_boxes.append(bb2)
                boxed_refs.append((model_id, idx, bb2, class_id))
        if not all_boxes:
            return [], np.zeros((h, w), dtype=np.uint8)

        masks = self.segmenter.segment(image_bgr, all_boxes)
        union = union_masks(masks, (h, w))

        # Manual overrides take precedence; ZoeDepth is only loaded if at
        # least one detection has no override.
        detection_refs = [(m, i) for (m, i, _, _) in boxed_refs]
        needs_zoe = not overrides.covers_all(
            image_path=image_path, image_id=image_id, detection_refs=detection_refs
        )
        use_calibration = self.calibrated_depth is not None
        depth_ft_map = (
            self.depth.estimate_depth_ft(image_bgr)
            if (needs_zoe and not use_calibration and self.depth is not None)
            else None
        )

        objects: list[DetectionObject] = []
        for i, (model_id, d_idx, bbox, class_id) in enumerate(boxed_refs):
            x1, y1, x2, y2 = bbox
            m = masks[i].astype(np.uint8)
            obj_full_rgba = extract_rgba_from_mask(image_bgr, m)
            obj_crop = obj_full_rgba[y1:y2, x1:x2]
            if obj_crop.size == 0:
                continue
            manual = overrides.lookup(
                image_path=image_path,
                image_id=image_id,
                model_id=model_id,
                detection_idx=d_idx,
            )
            if manual is not None:
                d_ft = float(manual)
                src = "manual"
            elif use_calibration and self.calibrated_depth is not None:
                real_size_m = overrides.real_size_lookup(
                    image_path=image_path,
                    image_id=image_id,
                    model_id=model_id,
                    detection_idx=d_idx,
                )
                d_ft = self.calibrated_depth.distance_ft(
                    image_w=w,
                    image_h=h,
                    bbox_xyxy=bbox,
                    model_id=model_id,
                    class_id=class_id,
                    real_size_m=real_size_m,
                )
                src = "calib"
            else:
                # depth_ft_map is guaranteed non-None here because covers_all
                # returned False for at least this detection.
                d_ft = median_depth_in_bbox(depth_ft_map, bbox)
                src = "zoe"
            self.store.upsert_depth_stat(
                image_id=image_id,
                model_id=model_id,
                detection_idx=d_idx,
                bbox_xyxy=bbox,
                d_initial_ft=d_ft,
                source=src,
            )
            objects.append(
                DetectionObject(
                    model_id=model_id,
                    detection_idx=d_idx,
                    bbox=bbox,
                    center=((x1 + x2) // 2, (y1 + y2) // 2),
                    depth_ft=d_ft,
                    depth_source=src,
                    rgba_obj=obj_crop,
                )
            )
        return objects, union

    def augment_image(
        self,
        image_path: Path,
        run_id: str,
        d_max_ft: float,
        step_ft: float,
        source_models: list[str],
        out_dir: Path,
        auto_run_oracle: bool = False,
        apply_effects: bool = True,
        overrides: DistanceOverrides | None = None,
    ) -> int:
        overrides = overrides or DistanceOverrides.empty()
        image_id, _, _ = hash_decoded_image(image_path)
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            return 0
        objects, union = self._build_objects(
            image_bgr,
            image_path=image_path,
            image_id=image_id,
            source_models=source_models,
            overrides=overrides,
        )
        if not objects:
            return 0
        plate, plate_path, msha = self.inpainter.get_or_create_plate(image_id=image_id, image_bgr=image_bgr, mask_u8=union, plate_dir=SETTINGS.bg_plate_dir)
        self.store.upsert_bg_plate(image_id=image_id, plate_path=plate_path, mask_sha=msha, inpainter="lama")
        out_dir.mkdir(parents=True, exist_ok=True)

        steps_written = 0
        delta = step_ft
        max_d0 = max(o.depth_ft for o in objects)
        n_manual = sum(1 for o in objects if o.depth_source == "manual")
        n_calib = sum(1 for o in objects if o.depth_source == "calib")
        n_zoe = len(objects) - n_manual - n_calib
        while max_d0 + delta <= d_max_ft + 1e-6:
            canvas = plate.copy()
            sort_idx = depth_sort_indices([o.depth_ft for o in objects], delta)
            for idx in sort_idx:
                obj = objects[idx]
                s = scale_for_delta(obj.depth_ft, delta)
                obj_scaled = scaled_object_rgba(obj.rgba_obj, s)
                canvas = paste_rgba_center(canvas, obj_scaled, obj.center)
            if apply_effects:
                canvas = dof_blur(canvas, strength=delta * 0.6)
                canvas = atmospheric_haze(canvas, strength=delta * 0.03)
            tag = f"{image_path.stem}_dist_{delta:.2f}ft".replace(".", "p")
            out_path = out_dir / f"{tag}{image_path.suffix.lower()}"
            cv2.imwrite(str(out_path), canvas)
            aug_image_id, _, _ = hash_decoded_image(out_path)
            self.store.insert_augmentation(
                augmented_image_id=aug_image_id,
                parent_image_id=image_id,
                step_index=int(round(delta / step_ft)),
                delta_ft=delta,
                params={
                    "d_max_ft": d_max_ft,
                    "step_ft": step_ft,
                    "source_models": source_models,
                    "inpainter": "lama",
                    "n_objects": len(objects),
                    "n_manual_distance": n_manual,
                    "n_calib_distance": n_calib,
                    "n_zoe_distance": n_zoe,
                },
            )
            steps_written += 1
            delta += step_ft

        if auto_run_oracle and steps_written > 0:
            self.orchestrator.ingest(image_root=out_dir, run_id=run_id, selected_models={"amod", "qrcode", "fd", "fr"}, skip_existing=False)
        return steps_written

