from __future__ import annotations

import numpy as np

from sensveridian.augmentation.depth import median_depth_in_bbox


def test_median_depth_in_bbox_normal_case() -> None:
    depth = np.arange(100, dtype=np.float32).reshape(10, 10)
    d = median_depth_in_bbox(depth, [2, 2, 6, 6])
    expected = float(np.median(depth[2:6, 2:6]))
    assert d == expected


def test_median_depth_in_bbox_out_of_bounds_is_clamped() -> None:
    depth = np.arange(100, dtype=np.float32).reshape(10, 10)
    d = median_depth_in_bbox(depth, [-10, -20, 100, 200])
    expected = float(np.median(depth))
    assert d == expected


def test_median_depth_in_bbox_zero_area_falls_back_to_global() -> None:
    depth = np.arange(25, dtype=np.float32).reshape(5, 5)
    d = median_depth_in_bbox(depth, [3, 3, 3, 3])  # empty roi
    assert d == float(np.median(depth))
