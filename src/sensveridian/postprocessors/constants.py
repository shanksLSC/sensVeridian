"""Per-model post-processing constants — mirror of the MLHILS reference headers.

The reference C headers (``multiobject.h``, ``qrcode.h``, ``detection.h``, …)
are not shipped with the weights, so the values here are taken from the decoder
``.c`` logic and calibrated against real model outputs. Where a value could not
be recovered exactly it is marked CALIBRATED and chosen so decoded boxes land on
objects; confirm against the MLHILS headers when available.

Each model's grids/strides are also derived at runtime from the actual output
tensor shapes, so a re-exported model with a different input size still decodes.
"""
from __future__ import annotations

# ---- AMOD (multiobject, FCOS, 6 output tensors) ----------------------------
# Reference frame the FCOS deltas were trained against (multiobject.c: 288x384).
MOD_REF_IMG_H = 288
MOD_REF_IMG_W = 384
MOD_NUM_CLASSES = 8                 # logits channels = 1 objectness + 8 classes
MOD_CONF_THRESHOLD = 0.30
MOD_NMS_IOU_THRESHOLD = 0.45
MOD_MAX_DETECTIONS = 64
# multiobject.c: x = cx ± delta * img_w / MAX_FPGA_OUTPUT. CALIBRATED (header
# value MOD_MAX_FPGA_OUTPUT not on disk) against AMOD outputs (delta range ~0-18).
MOD_MAX_FPGA_OUTPUT = 32.0
# FCOS scales: (grid_h, grid_w, stride). Matched to outputs by grid size.
MOD_SCALES = [(36, 48, 8), (18, 24, 16), (9, 12, 32)]
# Best-effort class names (header MOD_CLASS_NAMES not on disk). Index 0 = person
# is confirmed from real outputs; the remaining order is a best-effort automotive
# set — confirm against the MLHILS header when available.
MOD_CLASS_NAMES = [
    "person", "car", "truck", "bicycle", "motorcycle", "bus", "traffic_light", "stop_sign",
]

# ---- QR (qrcode, 4 anchors per cell) ---------------------------------------
# Channel layout (36 ch): 0-3 conf, 4-19 bbox(4 anchors x dx,dy,dw,dh),
# 20-31 angle (unused), 32-35 class (single class).
QR_ANCHOR_SIZES = [70.0, 89.0, 106.0, 129.0]
QR_ANCHOR_PER_GRID = 4
QR_CONF_THRESHOLD = 0.50
QR_NMS_IOU_THRESHOLD = 0.45
QR_MAX_DETECTIONS = 32
QR_CLASS_NAMES = ["qr"]

# ---- FD (face detection, 2 anchors per cell, + landmarks) ------------------
# Channel layout (38 ch): 0-1 conf, 2-9 bbox, 10-29 landmarks(5x2x2),
# 30-35 head pose (unused), 36-37 class (single).
FD_ANCHOR_W = [80.0, 30.0]
FD_ANCHOR_H = [80.0, 30.0]
FD_ANCHOR_PER_GRID = 2
FD_NUM_LANDMARKS = 5
FD_CONF_THRESHOLD = 0.50
FD_NMS_IOU_THRESHOLD = 0.45
FD_MAX_DETECTIONS = 32
FD_STRIDE = 16
FD_CLASS_NAMES = ["face"]

# ---- FR (face recognition, embedding) --------------------------------------
FR_EMBEDDING_SIZE = 128
FR_MATCH_THRESHOLD = 0.5  # cosine similarity
