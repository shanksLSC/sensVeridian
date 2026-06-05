"""Per-model output interpreters (post-processors).

Each model invoked for ingestion has a corresponding interpreter that turns its
raw head tensors into understandable outputs (decoded, NMS'd detections, or a
normalized embedding). These are Python ports of the reference C post-processors
at ``/data3/ssharma8/projects/lattice-internal/postptocessors_MLHILS/src``.

Convention: **every new detection model added for ingestion must register a
detection interpreter here** (model_id -> callable). A detection interpreter has
the signature ``interpret(outputs, img_w, img_h, conf_threshold=None) ->
list[dict]`` where each dict is ``{"bbox": [x1, y1, x2, y2] normalized 0..1,
"conf": float, "class_id": int, ...}``. Embedding models (e.g. ``fr``) use the
embedding interpreter instead.
"""
from __future__ import annotations

from . import detection, embedding, multiobject, qrcode

# model_id -> detection interpreter
DETECTION_INTERPRETERS = {
    "amod": multiobject.interpret,
    "qrcode": qrcode.interpret,
    "fd": detection.interpret_face,
}


def has_interpreter(model_id: str) -> bool:
    return model_id in DETECTION_INTERPRETERS


def interpret(model_id: str, outputs, img_w: int, img_h: int,
              conf_threshold: float | None = None) -> list[dict]:
    """Decode a detection model's raw outputs into normalized detections."""
    fn = DETECTION_INTERPRETERS.get(model_id)
    if fn is None:
        raise KeyError(
            f"no detection interpreter registered for model {model_id!r}; "
            f"add one under sensveridian.postprocessors and register it in "
            f"DETECTION_INTERPRETERS"
        )
    return fn(outputs, img_w, img_h, conf_threshold)


__all__ = [
    "DETECTION_INTERPRETERS",
    "has_interpreter",
    "interpret",
    "detection",
    "embedding",
    "multiobject",
    "qrcode",
]
