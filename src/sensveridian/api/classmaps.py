"""Model / class metadata — the bridge between sensVeridian runner outputs and
the labels the Veridian Studio UI renders.

Runners emit integer ``class_id`` values (AMOD) or modality-implicit classes
(``fd``/``fr`` -> face, ``qrcode`` -> qr). The UI works in human labels, so this
module maps ``model_id`` + ``class_id`` -> label and supplies the model cards
used to seed the ``models`` / ``model_versions`` tables.

The AMOD class list is taken from the design reference (``design_reference/app/
data.js`` CLASSES) and matches the contract's ``"classes": 6``. Confirm against
the deployed weights' training labels before going live (TODO: load from a
sidecar labels file shipped with the .h5).

Pure module: stdlib only, so it is unit-testable without the API stack.
"""
from __future__ import annotations

from typing import Optional

# AMOD: automotive classes (order = model output class index). The deployed
# head emits 8 classes; this mirrors postprocessors.constants.MOD_CLASS_NAMES.
AMOD_CLASSES: list[str] = [
    "person", "car", "truck", "bicycle", "motorcycle", "bus", "traffic_light", "stop_sign",
]

# AED: acoustic event vocabulary (mirrors design_reference AUDIO_LABELS).
AED_LABELS: list[str] = ["speech", "music", "siren", "alarm", "keyword", "noise", "silence"]

# Single-class detectors map their detections to one label.
_SINGLE_CLASS = {"qrcode": "qr", "fd": "face", "fr": "face"}


def class_name(model_id: str, class_id: int = 0) -> str:
    """Map a runner's (model_id, class_id) to the UI label."""
    if model_id == "amod":
        if 0 <= class_id < len(AMOD_CLASSES):
            return AMOD_CLASSES[class_id]
        return f"class_{class_id}"
    if model_id == "aed":
        if 0 <= class_id < len(AED_LABELS):
            return AED_LABELS[class_id]
        return f"event_{class_id}"
    return _SINGLE_CLASS.get(model_id, f"class_{class_id}")


def n_classes(model_id: str) -> int:
    return {
        "amod": len(AMOD_CLASSES),
        "qrcode": 1,
        "fd": 1,
        "fr": 1,
        "aed": len(AED_LABELS),
    }.get(model_id, 0)


# Static model cards (display metadata + the runner versions, mirrored here so
# seeding the models table needs no TensorFlow import). Versions match the
# sensveridian.runners.* class attributes.
MODEL_CARDS: dict[str, dict] = {
    "amod": {
        "display_name": "AutomotiveMultiObjectDetection",
        "short": "AMOD",
        "input": "320×320×3",
        "version": "8.2.0",
        "depends_on": None,
    },
    "qrcode": {
        "display_name": "QRCodeDetection",
        "short": "QR",
        "input": "320×320×1",
        "version": "final",
        "depends_on": None,
    },
    "fd": {
        "display_name": "FaceDetection",
        "short": "FD",
        "input": "320×320×3",
        "version": "8.1.0",
        "depends_on": None,
    },
    "fr": {
        "display_name": "FaceRecognition",
        "short": "FR",
        "input": "112×112×3",
        "version": "8.1.1",
        "depends_on": "fd",
    },
    "aed": {
        "display_name": "AcousticEventDetection",
        "short": "AED",
        "input": "16kHz mono",
        "version": "0.1.0",
        "depends_on": None,
    },
}


def model_card(model_id: str) -> dict:
    """Return display metadata for a model id, with sensible fallbacks."""
    card = dict(MODEL_CARDS.get(model_id, {}))
    card.setdefault("display_name", model_id)
    card.setdefault("short", model_id.upper())
    card.setdefault("input", "")
    card.setdefault("version", "0")
    card.setdefault("depends_on", None)
    card["classes"] = n_classes(model_id)
    return card


def short_name(model_id: str) -> str:
    return MODEL_CARDS.get(model_id, {}).get("short", model_id.upper())


def input_spec(model_id: str) -> str:
    return MODEL_CARDS.get(model_id, {}).get("input", "")


def depends_on(model_id: str) -> Optional[str]:
    return MODEL_CARDS.get(model_id, {}).get("depends_on")
