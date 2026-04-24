from __future__ import annotations

import numpy as np

from sensveridian.augmentation.effects import atmospheric_haze, dof_blur


def test_dof_blur_keeps_shape_and_dtype(tiny_image_bgr: np.ndarray) -> None:
    out = dof_blur(tiny_image_bgr, strength=0.1)
    assert out.shape == tiny_image_bgr.shape
    assert out.dtype == tiny_image_bgr.dtype


def test_atmospheric_haze_clamps_strength_and_moves_toward_haze_color(tiny_image_bgr: np.ndarray) -> None:
    out = atmospheric_haze(tiny_image_bgr, strength=1.0, haze_color=(220, 220, 220))
    assert out.shape == tiny_image_bgr.shape
    assert out.dtype == tiny_image_bgr.dtype
    # clamp at 0.35 means blend is substantial; mean should shift upward for dark image.
    assert float(np.mean(out)) > float(np.mean(tiny_image_bgr))
