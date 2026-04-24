from __future__ import annotations

import pytest

from sensveridian.augmentation.calibration import CalibratedDistanceEstimator
from sensveridian.augmentation.camera import IMX219_PROFILE


def test_calibration_distance_closed_form() -> None:
    est = CalibratedDistanceEstimator(IMX219_PROFILE)
    d_ft = est.distance_ft(
        image_w=1920,
        image_h=1080,
        bbox_xyxy=[100, 200, 200, 400],  # h = 200 px
        model_id="fd",
        class_id=0,
    )
    # expected_m = (0.22 * fy) / 200; expected_ft = expected_m * 3.28084
    assert d_ft == pytest.approx(4.293, rel=1e-3)


def test_missing_class_prior_raises() -> None:
    est = CalibratedDistanceEstimator(IMX219_PROFILE, priors={})
    with pytest.raises(ValueError):
        est.distance_ft(
            image_w=1920,
            image_h=1080,
            bbox_xyxy=[100, 100, 200, 260],
            model_id="amod",
            class_id=99,
        )


def test_real_size_override_beats_prior() -> None:
    est = CalibratedDistanceEstimator(IMX219_PROFILE)
    from_prior = est.distance_ft(
        image_w=1920,
        image_h=1080,
        bbox_xyxy=[100, 100, 200, 300],  # h = 200
        model_id="fd",
        class_id=0,
    )
    from_override = est.distance_ft(
        image_w=1920,
        image_h=1080,
        bbox_xyxy=[100, 100, 200, 300],
        model_id="fd",
        class_id=0,
        real_size_m=(0.30, 0.20),
    )
    assert from_override > from_prior
