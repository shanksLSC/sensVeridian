from __future__ import annotations

from typing import Optional

from .camera import CameraProfile

M_TO_FT = 3.28084

# (model_id, class_id_int) -> (real_height_m, real_width_m)
CLASS_SIZE_PRIORS: dict[tuple[str, int], tuple[float, float]] = {
    ("fd", 0): (0.22, 0.16),
    ("qrcode", 0): (0.10, 0.10),
    ("amod", 0): (1.70, 0.45),
    ("amod", 1): (1.45, 1.80),
    ("amod", 2): (1.10, 0.75),
    ("amod", 3): (1.20, 0.50),
}


class CalibratedDistanceEstimator:
    def __init__(
        self,
        profile: CameraProfile,
        priors: Optional[dict[tuple[str, int], tuple[float, float]]] = None,
    ):
        self.profile = profile
        self.priors = priors or CLASS_SIZE_PRIORS

    def distance_ft(
        self,
        *,
        image_w: int,
        image_h: int,
        bbox_xyxy: list[int],
        model_id: str,
        class_id: Optional[int],
        real_size_m: Optional[tuple[float, float]] = None,
    ) -> float:
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        bbox_w_px = max(1, x2 - x1)
        bbox_h_px = max(1, y2 - y1)

        if real_size_m is None:
            key = (str(model_id), int(class_id) if class_id is not None else 0)
            real_size_m = self.priors.get(key)
        if real_size_m is None:
            cid = "None" if class_id is None else str(class_id)
            raise ValueError(f"No size prior available for model={model_id} class_id={cid}")

        real_h_m, real_w_m = float(real_size_m[0]), float(real_size_m[1])
        fx, fy = self.profile.fx_fy_at(image_w=image_w, image_h=image_h)

        if bbox_h_px > 0 and real_h_m > 0:
            distance_m = (real_h_m * fy) / float(bbox_h_px)
        elif bbox_w_px > 0 and real_w_m > 0:
            distance_m = (real_w_m * fx) / float(bbox_w_px)
        else:
            distance_m = 0.0

        distance_ft = max(distance_m * M_TO_FT, 1e-6)
        return float(distance_ft)
