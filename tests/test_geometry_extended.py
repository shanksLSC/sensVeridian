from __future__ import annotations

import numpy as np

from sensveridian.augmentation.geometry import (
    extract_rgba_from_mask,
    paste_rgba_center,
    scaled_object_rgba,
)


def test_extract_rgba_from_mask_sets_alpha() -> None:
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[:] = (10, 20, 30)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    rgba = extract_rgba_from_mask(img, mask)
    assert rgba.shape == (4, 4, 4)
    assert rgba[0, 0, 3] == 0
    assert rgba[1, 1, 3] == 255


def test_scaled_object_rgba_resizes() -> None:
    obj = np.zeros((10, 20, 4), dtype=np.uint8)
    half = scaled_object_rgba(obj, 0.5)
    same = scaled_object_rgba(obj, 1.0)
    assert half.shape[:2] == (5, 10)
    assert same.shape[:2] == (10, 20)


def test_paste_rgba_center_inside_and_outside() -> None:
    bg = np.zeros((20, 20, 3), dtype=np.uint8)
    obj = np.zeros((6, 6, 4), dtype=np.uint8)
    obj[..., :3] = (200, 0, 0)
    obj[..., 3] = 255  # fully opaque

    out = paste_rgba_center(bg, obj, center_xy=(10, 10))
    assert np.any(out[..., 0] == 200)

    # Entirely out of bounds -> unchanged.
    out2 = paste_rgba_center(bg, obj, center_xy=(-100, -100))
    assert np.array_equal(out2, bg)
