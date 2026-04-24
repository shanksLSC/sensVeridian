from __future__ import annotations

import pytest

from sensveridian.augmentation.camera import IMX219_PROFILE, get_camera_profile


def test_imx219_native_intrinsics() -> None:
    fx, fy = IMX219_PROFILE.fx_fy_native()
    assert fx == pytest.approx(2709.5652, rel=1e-4)
    assert fy == pytest.approx(2713.9710, rel=1e-4)


def test_imx219_resolution_scaling() -> None:
    fx, fy = IMX219_PROFILE.fx_fy_at(1640, 1232)
    assert fx == pytest.approx(1354.7826, rel=1e-4)
    assert fy == pytest.approx(1356.9855, rel=1e-4)


def test_registry_lookup() -> None:
    prof = get_camera_profile("imx219")
    assert prof.name == "imx219"
    assert prof.native_w_px == 3280
    assert prof.native_h_px == 2464
