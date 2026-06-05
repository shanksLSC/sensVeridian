# Model interpreters (post-processors)

Every model invoked for ingestion has a corresponding **interpreter** that turns
its raw output tensors into understandable results — decoded, NMS'd detection
boxes (with class + confidence) or, for face recognition, a normalized
embedding. These are Python ports of the reference C post-processors at
`/data3/ssharma8/projects/lattice-internal/postptocessors_MLHILS/src` and live in
`sensveridian.postprocessors`.

## Why

The raw model heads emit anchor/grid candidates (objectness logits, class
logits, box deltas) — not usable boxes. The interpreter applies the model's
decode (anchor or FCOS), sigmoid/softmax gating, and non-max suppression, then
returns normalized detections. Without it the canvas would show hundreds of
degenerate boxes; with it, boxes land on the objects.

## Registry and contract

`sensveridian/postprocessors/__init__.py` maps `model_id -> interpreter`:

```python
DETECTION_INTERPRETERS = {
    "amod":   multiobject.interpret,      # FCOS, 6 tensors, 8 classes
    "qrcode": qrcode.interpret,           # 4 anchors, single class
    "fd":     detection.interpret_face,   # 2 anchors + 5 landmarks
}
```

A **detection interpreter** has the signature:

```python
interpret(outputs, img_w, img_h, conf_threshold=None) -> list[dict]
# each dict: {"bbox": [x1, y1, x2, y2] normalized 0..1, "conf": float, "class_id": int, ...}
```

Boxes are normalized against the model input size, so they overlay directly on
the displayed image regardless of the original resolution. Face recognition
(`fr`) uses `embedding.interpret_embedding` (unit vector) + cosine matching via
the `FaceRegistry` instead of a box interpreter.

## Adding a new model

When a new detection model is added for ingestion:

1. Add a module under `sensveridian/postprocessors/` (port its post-processor).
2. Put its constants (anchors, thresholds, class names) in `constants.py`.
3. Register it in `DETECTION_INTERPRETERS`.
4. Add a class map in `sensveridian/api/classmaps.py` if it introduces classes.

An unregistered model raises a clear `KeyError` at decode time, so the
requirement is explicit.

## Calibration note

The reference C headers (thresholds, `MOD_MAX_FPGA_OUTPUT`, exact class-name
order) were not shipped with the `.h5` weights. The values in `constants.py` are
taken from the decoder `.c` logic and **calibrated against real model outputs**
(boxes verified to land on objects). Confirm them against the MLHILS headers
when available — the decode structure is exact; only a few scalar constants are
calibrated.
