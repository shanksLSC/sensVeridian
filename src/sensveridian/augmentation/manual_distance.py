"""Manual distance overrides for the distance-sweep augmentor.

ZoeDepth gives us a monocular estimate of ``d_initial_ft`` for every detection.
That estimate is generic but never exact. When the operator knows the real
distance at capture time (calibration rig, measured ground truth, etc.), they
should be able to pin that value per-detection, per-image, or globally, and
have ZoeDepth kick in only as a fallback.

Precedence, high -> low:

1. Per-detection override (``<model_id>:<detection_idx>``)
2. Per-image default (``images[<key>].default``)
3. Global ``--d0-ft`` value
4. ZoeDepth median depth inside the bbox

JSON schema accepted by :func:`DistanceOverrides.from_json`::

    {
      "global_ft": 5.0,                       # optional
      "images": {
        "door_01.jpg":                  {"default": 5.2},
        "/abs/path/to/door_02.jpg":     {"default": 4.8,
                                         "detections": {"amod:0": 4.5}},
        "<sha256-image_id>":            {"default": 6.0}
      }
    }

The image key can be any of: the full absolute path, the basename, the
basename stem (no extension), or the ``image_id`` (sha256 of decoded pixels).
Whichever the operator finds most convenient works.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json


@dataclass
class ImageOverride:
    default: Optional[float] = None
    detections: dict[str, float] = field(default_factory=dict)


@dataclass
class DistanceOverrides:
    global_ft: Optional[float] = None
    images: dict[str, ImageOverride] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "DistanceOverrides":
        return cls()

    @classmethod
    def from_json(cls, path: Path) -> "DistanceOverrides":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        global_ft = data.get("global_ft")
        images_raw = data.get("images", {}) or {}
        images: dict[str, ImageOverride] = {}
        for key, payload in images_raw.items():
            if not isinstance(payload, dict):
                continue
            default = payload.get("default")
            dets = payload.get("detections", {}) or {}
            images[str(key)] = ImageOverride(
                default=float(default) if default is not None else None,
                detections={str(k): float(v) for k, v in dets.items()},
            )
        return cls(
            global_ft=float(global_ft) if global_ft is not None else None,
            images=images,
        )

    def _image_keys(self, image_path: Path, image_id: str) -> list[str]:
        p = Path(image_path)
        return [str(p), p.name, p.stem, image_id]

    def _image_override(self, image_path: Path, image_id: str) -> Optional[ImageOverride]:
        for k in self._image_keys(image_path, image_id):
            if k in self.images:
                return self.images[k]
        return None

    def lookup(
        self,
        *,
        image_path: Path,
        image_id: str,
        model_id: str,
        detection_idx: int,
    ) -> Optional[float]:
        """Return the manual distance in feet, or ``None`` if no override applies."""
        img_over = self._image_override(image_path, image_id)
        if img_over is not None:
            key = f"{model_id}:{detection_idx}"
            if key in img_over.detections:
                return img_over.detections[key]
            if img_over.default is not None:
                return img_over.default
        if self.global_ft is not None:
            return self.global_ft
        return None

    def covers_all(
        self,
        *,
        image_path: Path,
        image_id: str,
        detection_refs: list[tuple[str, int]],
    ) -> bool:
        """True iff every ``(model_id, detection_idx)`` has a manual override.

        Used by the augmentor to skip loading ZoeDepth entirely when nothing
        will fall back to it.
        """
        if not detection_refs:
            return True
        for model_id, idx in detection_refs:
            if self.lookup(
                image_path=image_path,
                image_id=image_id,
                model_id=model_id,
                detection_idx=idx,
            ) is None:
                return False
        return True
