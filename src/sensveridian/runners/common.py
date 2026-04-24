from __future__ import annotations

from typing import Any
import importlib.util
from pathlib import Path
import numpy as np
import cv2


def lazy_load_tf():
    import tensorflow as tf  # type: ignore

    return tf


def _load_sensai_binary_ops_module():
    mod_path = Path("/data3/ssharma8/projects/lattice-internal/sensai-sdk/python/sensai/frontend/keras_binary_ops.py")
    if not mod_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("sensveridian_sensai_binary_ops", str(mod_path))
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


def load_sensai_h5_model(weights_path: str):
    tf = lazy_load_tf()
    custom_objects: dict[str, Any] = {}
    try:
        import lscquant.layers as lq_layers  # type: ignore

        for name in dir(lq_layers):
            obj = getattr(lq_layers, name)
            if isinstance(name, str) and name and name[0].isupper():
                custom_objects[name] = obj
    except Exception:
        pass

    bo = _load_sensai_binary_ops_module()
    if bo is not None:
        custom_objects.update(
            {
                "bo": bo,
                "binary_ops": bo,
                "lin_8b_quant": getattr(bo, "lin_8b_quant", None),
                "FixedDropout": getattr(bo, "FixedDropout", None),
                "MyInitializer": getattr(bo, "MyInitializer", None),
                "MyRegularizer": getattr(bo, "MyRegularizer", None),
                "MyConstraints": getattr(bo, "MyConstraints", None),
                "CastToFloat32": getattr(bo, "CastToFloat32", None),
                "lin_nb_quant": getattr(bo, "lin_nb_quant", None),
            }
        )
    custom_objects = {k: v for k, v in custom_objects.items() if v is not None}
    if not custom_objects:
        return tf.keras.models.load_model(weights_path, compile=False)
    return tf.keras.models.load_model(weights_path, compile=False, custom_objects=custom_objects)


def preprocess_for_model(image_bgr: np.ndarray, input_shape: tuple[int, int, int]) -> np.ndarray:
    h, w, c = input_shape
    resized = cv2.resize(image_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    if c == 1:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        x = gray.astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=-1)
    else:
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255.0
    return np.expand_dims(x, 0)


def as_list_of_arrays(pred: Any) -> list[np.ndarray]:
    if isinstance(pred, list):
        return [np.asarray(p) for p in pred]
    if isinstance(pred, tuple):
        return [np.asarray(p) for p in pred]
    return [np.asarray(pred)]


def extract_detection_candidates(outputs: list[np.ndarray], conf_threshold: float = 0.3) -> list[dict]:
    dets: list[dict] = []
    for arr in outputs:
        a = np.asarray(arr)
        if a.ndim >= 3:
            a = a.reshape(-1, a.shape[-1])
        elif a.ndim == 2:
            pass
        else:
            continue
        if a.shape[-1] < 5:
            continue
        for row in a:
            vals = row.astype(float).tolist()
            conf = float(max(vals[4], 0.0))
            if conf < conf_threshold:
                continue
            x1, y1, x2, y2 = vals[:4]
            class_id = int(np.argmax(vals[5:])) if len(vals) > 5 else 0
            dets.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "conf": conf,
                    "class_id": class_id,
                }
            )
    return dets


def safe_bbox_xyxy(bbox: list[float], w: int, h: int) -> list[int]:
    x1, y1, x2, y2 = bbox
    x1 = int(max(0, min(w - 1, x1)))
    x2 = int(max(0, min(w - 1, x2)))
    y1 = int(max(0, min(h - 1, y1)))
    y2 = int(max(0, min(h - 1, y2)))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return [x1, y1, x2, y2]

