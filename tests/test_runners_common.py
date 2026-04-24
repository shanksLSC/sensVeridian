from __future__ import annotations

import numpy as np

from sensveridian.runners.common import (
    as_list_of_arrays,
    extract_detection_candidates,
    preprocess_for_model,
    safe_bbox_xyxy,
)


def test_preprocess_for_model_rgb_shape_range(tiny_image_bgr: np.ndarray) -> None:
    x = preprocess_for_model(tiny_image_bgr, (32, 48, 3))
    assert x.shape == (1, 32, 48, 3)
    assert x.dtype == np.float32
    assert float(x.min()) >= 0.0
    assert float(x.max()) <= 1.0


def test_preprocess_for_model_grayscale_shape_range(tiny_image_bgr: np.ndarray) -> None:
    x = preprocess_for_model(tiny_image_bgr, (40, 24, 1))
    assert x.shape == (1, 40, 24, 1)
    assert x.dtype == np.float32
    assert float(x.min()) >= 0.0
    assert float(x.max()) <= 1.0


def test_as_list_of_arrays_accepts_list_tuple_and_single() -> None:
    arr = np.ones((2, 3), dtype=np.float32)
    assert len(as_list_of_arrays([arr, arr])) == 2
    assert len(as_list_of_arrays((arr,))) == 1
    out = as_list_of_arrays(arr)
    assert len(out) == 1
    assert np.array_equal(out[0], arr)


def test_extract_detection_candidates_filters_threshold_and_gets_class() -> None:
    # rows are [x1, y1, x2, y2, conf, cls0, cls1]
    pred = np.array(
        [
            [1, 2, 3, 4, 0.2, 0.1, 0.9],  # below threshold -> dropped
            [10, 20, 30, 40, 0.95, 0.8, 0.2],  # class 0
            [11, 21, 31, 41, 0.40, 0.1, 0.9],  # class 1
        ],
        dtype=np.float32,
    )
    dets = extract_detection_candidates([pred], conf_threshold=0.3)
    assert len(dets) == 2
    assert dets[0]["class_id"] == 0
    assert dets[1]["class_id"] == 1
    assert dets[0]["bbox"] == [10.0, 20.0, 30.0, 40.0]


def test_extract_detection_candidates_skips_invalid_shapes() -> None:
    bad = np.array([1, 2, 3], dtype=np.float32)
    short = np.array([[1, 2, 3, 4]], dtype=np.float32)  # only 4 cols
    assert extract_detection_candidates([bad, short], conf_threshold=0.1) == []


def test_safe_bbox_xyxy_clamps_and_repairs_degenerate() -> None:
    bbox = safe_bbox_xyxy([-10, -20, -5, -1], w=100, h=60)
    x1, y1, x2, y2 = bbox
    assert 0 <= x1 < 100
    assert 0 <= x2 < 100
    assert 0 <= y1 < 60
    assert 0 <= y2 < 60
    assert x2 > x1
    assert y2 > y1
