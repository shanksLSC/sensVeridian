from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from sensveridian.runners.amod import AMODRunner
from sensveridian.runners.base import RunnerOutput, Summary
from sensveridian.runners.face_detection import FaceDetectionRunner
from sensveridian.runners.face_recognition import FaceRecognitionRunner
from sensveridian.runners.qrcode import QRCodeRunner


def test_amod_predict_has_positive_and_negative_paths(tiny_image_bgr: np.ndarray) -> None:
    runner = AMODRunner(weights_path="/tmp/amod.h5", conf_threshold=0.3)
    model = MagicMock()
    model.predict.return_value = np.array(
        [
            [10, 20, 30, 40, 0.9, 0.9, 0.1],
            [1, 2, 3, 4, 0.1, 0.2, 0.8],
        ],
        dtype=np.float32,
    )
    runner.model = model
    runner.input_spec = (64, 64, 3)
    out = runner.predict(tiny_image_bgr, deps={})
    assert out.summary.present is True
    assert out.summary.count == 1
    assert len(out.raw["detections"]) == 1

    model.predict.return_value = np.array([[1, 2, 3, 4, 0.2, 0.8, 0.2]], dtype=np.float32)
    out2 = runner.predict(tiny_image_bgr, deps={})
    assert out2.summary.present is False
    assert out2.summary.count == 0


def test_amod_load_updates_input_spec(monkeypatch) -> None:
    runner = AMODRunner(weights_path="/tmp/amod.h5")
    fake_model = MagicMock()
    fake_model.input_shape = (None, 320, 240, 3)

    monkeypatch.setattr("sensveridian.runners.amod.load_sensai_h5_model", lambda *_: fake_model)
    runner.load()
    assert runner.input_spec == (320, 240, 3)
    assert runner.model is fake_model


def test_qrcode_predict_uses_decode_multi(tiny_image_bgr: np.ndarray) -> None:
    runner = QRCodeRunner(weights_path="/tmp/qr.h5", conf_threshold=0.3)
    model = MagicMock()
    model.predict.return_value = np.array([[10, 20, 30, 40, 0.8, 0.7, 0.3]], dtype=np.float32)
    runner.model = model
    runner.input_spec = (64, 64, 1)

    runner.cv_qr = MagicMock()
    runner.cv_qr.detectAndDecodeMulti.return_value = (
        True,
        ["abc", ""],
        np.array([[[1, 2], [3, 4], [5, 6], [7, 8]]], dtype=np.float32),
        None,
    )
    out = runner.predict(tiny_image_bgr, deps={})
    assert out.summary.present is True
    assert out.summary.count == 2
    assert out.raw["decoded_texts"] == ["abc"]


def test_face_detection_scales_normalized_bboxes(tiny_image_bgr: np.ndarray) -> None:
    runner = FaceDetectionRunner(weights_path="/tmp/fd.h5", conf_threshold=0.3)
    model = MagicMock()
    # normalized values <= 1.5 trigger scaling path
    model.predict.return_value = np.array([[0.1, 0.2, 1.0, 1.2, 0.95, 0.8, 0.2]], dtype=np.float32)
    runner.model = model
    runner.input_spec = (64, 64, 3)
    out = runner.predict(tiny_image_bgr, deps={})
    assert out.summary.present is True
    assert out.summary.count == 1
    assert len(out.raw["crops"]) == len(out.raw["detections"]) == 1
    bbox = out.raw["detections"][0]["bbox"]
    assert bbox[2] > bbox[0]
    assert bbox[3] > bbox[1]


def test_face_recognition_empty_when_no_fd_dep(file_registry) -> None:
    runner = FaceRecognitionRunner(weights_path="/tmp/fr.h5", registry=file_registry, threshold=0.5)
    model = MagicMock()
    model.predict.return_value = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    runner.model = model
    runner.input_spec = (32, 32, 3)
    out = runner.predict(np.zeros((64, 64, 3), dtype=np.uint8), deps={})
    assert out.summary.present is False
    assert out.summary.count == 0
    assert out.raw["recognized"] == []


def test_face_recognition_matches_one_of_two(file_registry) -> None:
    # Register one known identity.
    file_registry.register("person_001", "Alice", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))

    runner = FaceRecognitionRunner(weights_path="/tmp/fr.h5", registry=file_registry, threshold=0.8)
    model = MagicMock()
    # first crop matches registered embedding, second does not
    model.predict.side_effect = [
        np.array([[0.99, 0.01, 0.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
    ]
    runner.model = model
    runner.input_spec = (32, 32, 3)

    crop1 = np.ones((20, 20, 3), dtype=np.uint8) * 255
    crop2 = np.zeros((20, 20, 3), dtype=np.uint8)
    fd_out = RunnerOutput(
        summary=Summary(True, 2, {}),
        raw={
            "detections": [{"bbox": [0, 0, 10, 10]}, {"bbox": [10, 10, 20, 20]}],
            "crops": [crop1, crop2],
        },
    )
    out = runner.predict(np.zeros((64, 64, 3), dtype=np.uint8), deps={"fd": fd_out})
    assert out.summary.count == 1
    assert out.summary.extras["face_candidates"] == 2
    assert out.raw["recognized"][0]["matched_person_id"] == "person_001"
    assert out.raw["recognized"][1]["matched_person_id"] is None
