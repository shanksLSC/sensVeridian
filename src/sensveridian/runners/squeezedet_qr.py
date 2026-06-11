"""SqueezeDet QR runner — reuses the qr-detection-bboxes project's own decode.

The QR detectors the team ships are SqueezeDet models (config-driven anchors,
grayscale or RGB input, landmark/angle heads) decoded by that project's
``interpret_squeezedet_output``. Rather than reimplement that decode (and risk
divergence), this runner imports the project's ``model`` module at load time
(its ``sources`` dir is added to ``sys.path`` via ``QR_DETECTION_SOURCES``) and
calls it directly. The preprocess / NMS / rescale logic is ported from the
project's ``scripts/run_test_predictions.py`` so this runner has no dependency on
that script's heavyweight ``testing.py`` (matplotlib / pycocotools).

Registered with ``runner_kind = "squeezedet_qr"``; two model rows use it
(``qr_gray`` grayscale 4:3, ``qr_rgb`` RGB 4:3). Output detections are absolute
pixel ``[x1, y1, x2, y2]`` boxes which fusion normalizes (BOX_SPACE auto).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from .base import RunnerOutput, Summary

DEFAULT_SOURCES = "/data3/ssharma8/projects/qr-detection-bboxes/sources"


def _sources_dir() -> str:
    return os.getenv("QR_DETECTION_SOURCES", DEFAULT_SOURCES)


def _iou_xyxy(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    uni = ua + ub - inter
    return inter / uni if uni > 0 else 0.0


class SqueezeDetQRRunner:
    depends_on: tuple[str, ...] = ()

    def __init__(
        self,
        model_id: str,
        weights_path: str,
        config_path: str,
        display_name: str = "QRCodeDetection",
        version: str = "8.2",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.25,
        max_boxes: int = 20,
        sources_dir: Optional[str] = None,
    ):
        self.model_id = model_id
        self.weights_path = weights_path
        self.config_path = config_path
        self.display_name = display_name
        self.version = version
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_boxes = max_boxes
        self.sources_dir = sources_dir or _sources_dir()
        self.model = None
        self.cfg: dict = {}
        # filled from config at load(): (h, w, channels)
        self.input_spec = (192, 256, 1)
        self._interpret = None

    # ---- load ---------------------------------------------------------------
    def load(self) -> None:
        from .common import load_sensai_h5_model

        self.cfg = yaml.safe_load(Path(self.config_path).read_text(encoding="utf-8"))
        image_size = list(self.cfg.get("image_size", [192, 256]))
        h, w = int(image_size[0]), int(image_size[1])
        channels = int(image_size[2]) if len(image_size) >= 3 else 1
        self.input_spec = (h, w, channels)

        # Bring the project's decode onto the path and import it.
        if self.sources_dir not in sys.path:
            sys.path.insert(0, self.sources_dir)
        from model import interpret_squeezedet_output  # type: ignore

        self._interpret = interpret_squeezedet_output
        # load_sensai_h5_model registers the lscquant custom objects
        # (sensAI>QuantizeConv2D etc.) so the quantized .h5 deserializes.
        self.model = load_sensai_h5_model(self.weights_path)
        # honor the model's actual input shape if it disagrees with config
        try:
            shp = self.model.input_shape
            self.input_spec = (int(shp[1]), int(shp[2]), int(shp[3]))
        except Exception:
            pass

    # ---- preprocess (letterbox + /128 normalize), per channels --------------
    def _preprocess(self, image_bgr: np.ndarray):
        h, w, channels = self.input_spec
        oh, ow = image_bgr.shape[:2]
        if channels == 1:
            img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
        else:
            img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)

        scale = min(w / ow, h / oh)
        new_w, new_h = max(1, int(ow * scale)), max(1, int(oh * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if channels == 1:
            padded = np.zeros((h, w), dtype=np.uint8)
        else:
            padded = np.zeros((h, w, channels), dtype=np.uint8)
        dy, dx = (h - new_h) // 2, (w - new_w) // 2
        padded[dy:dy + new_h, dx:dx + new_w] = resized

        norm = padded.astype(np.float32) / 128.0
        if channels == 1:
            batch = np.expand_dims(np.expand_dims(norm, axis=-1), axis=0)
        else:
            batch = np.expand_dims(norm, axis=0)
        return batch, {"dx": dx, "dy": dy, "scale": scale, "oh": oh, "ow": ow}

    def _dummy_targets(self, tf, num_anchors: int, num_classes: int, num_landmarks: int):
        z = tf.zeros
        return (
            z((1, num_anchors), dtype=tf.float32),                       # anchors_mask
            z((1, num_anchors, 4), dtype=tf.float32),                    # bboxes_delta
            z((1, num_anchors, 4), dtype=tf.float32),                    # bboxes_input
            z((1, num_anchors, max(1, num_classes)), dtype=tf.float32),  # labels
            z((1, num_anchors, num_landmarks * 2), dtype=tf.float32),    # landmarks_delta
            z((1, num_anchors, num_landmarks * 2), dtype=tf.float32),    # landmarks_input
            z((1, num_anchors, 3), dtype=tf.float32),                    # angles_input
            z((1, num_anchors), dtype=tf.float32),                       # angles_mask
            z((1, num_anchors, 1), dtype=tf.float32),                    # true_boxes_mask
            z((1, num_anchors, num_landmarks * 2), dtype=tf.float32),    # landmarks_mask
        )

    def _nms(self, boxes_yxyx, probs, classes):
        """Greedy NMS. boxes are [y1,x1,y2,x2] in padded model space."""
        keep = []
        order = list(np.argsort(-np.asarray(probs))[: self.max_boxes])
        xyxy = [[b[1], b[0], b[3], b[2]] for b in boxes_yxyx]
        while order:
            i = order.pop(0)
            keep.append(i)
            order = [j for j in order if _iou_xyxy(xyxy[i], xyxy[j]) <= self.iou_threshold]
        return keep

    # ---- predict ------------------------------------------------------------
    def predict(self, image_bgr: np.ndarray, deps: dict[str, RunnerOutput]) -> RunnerOutput:
        import tensorflow as tf

        if self.model is None:
            self.load()

        batch, pad = self._preprocess(image_bgr)
        raw = self.model.predict(batch, verbose=0)
        if isinstance(raw, (list, tuple)):
            raw = raw[0]

        num_classes = int(self.cfg["num_classes"])
        image_size = tuple(self.cfg["image_size"][:2])
        anchor_grid_size = tuple(self.cfg["anchor_grid_size"])
        anchor_sizes = self.cfg["anchor_sizes"]
        num_landmarks = int(self.cfg.get("num_landmarks", 0))
        num_anchors = anchor_grid_size[0] * anchor_grid_size[1] * int(self.cfg["anchor_per_grid"])

        targets = self._dummy_targets(tf, num_anchors, num_classes, num_landmarks)
        _, _, predictions, _ = self._interpret(
            inputs=tf.constant(batch),
            targets=targets,
            predictions=tf.constant(raw),
            metadata={},
            num_classes=num_classes,
            image_size=image_size,
            anchor_grid_size=anchor_grid_size,
            anchor_sizes=anchor_sizes,
            num_landmarks=num_landmarks,
        )
        det_probs = np.array(predictions[5][0])
        det_boxes = np.array(predictions[7][0])   # [y1, x1, y2, x2] padded space
        det_classes = np.array(predictions[6][0])

        valid = det_probs >= self.conf_threshold
        det_probs, det_boxes, det_classes = det_probs[valid], det_boxes[valid], det_classes[valid]

        dets: list[dict] = []
        if len(det_boxes):
            keep = self._nms(det_boxes.tolist(), det_probs.tolist(), det_classes.tolist())
            scale, dx, dy = pad["scale"], pad["dx"], pad["dy"]
            oh, ow = pad["oh"], pad["ow"]
            for i in keep:
                y1, x1, y2, x2 = det_boxes[i]
                # undo letterbox -> original pixels, as [x1, y1, x2, y2]
                ox1 = float(np.clip((x1 - dx) / scale, 0, ow - 1))
                oy1 = float(np.clip((y1 - dy) / scale, 0, oh - 1))
                ox2 = float(np.clip((x2 - dx) / scale, 0, ow - 1))
                oy2 = float(np.clip((y2 - dy) / scale, 0, oh - 1))
                dets.append({
                    "bbox": [min(ox1, ox2), min(oy1, oy2), max(ox1, ox2), max(oy1, oy2)],
                    "conf": float(det_probs[i]),
                    "class_id": int(det_classes[i]),
                })

        summary = Summary(present=len(dets) > 0, count=len(dets),
                          extras={"output_shape": list(np.asarray(raw).shape)})
        return RunnerOutput(summary=summary, raw={"detections": dets})
