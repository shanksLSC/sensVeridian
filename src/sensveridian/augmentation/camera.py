from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class CameraProfile:
    name: str
    focal_length_mm: float
    sensor_w_mm: float
    sensor_h_mm: float
    native_w_px: int
    native_h_px: int

    def fx_fy_native(self) -> tuple[float, float]:
        fx = self.focal_length_mm * self.native_w_px / self.sensor_w_mm
        fy = self.focal_length_mm * self.native_h_px / self.sensor_h_mm
        return float(fx), float(fy)

    def fx_fy_at(self, image_w: int, image_h: int) -> tuple[float, float]:
        fx_native, fy_native = self.fx_fy_native()
        sx = float(image_w) / float(self.native_w_px)
        sy = float(image_h) / float(self.native_h_px)
        return float(fx_native * sx), float(fy_native * sy)

    def with_native_resolution(self, native_w_px: int, native_h_px: int) -> "CameraProfile":
        return replace(self, native_w_px=int(native_w_px), native_h_px=int(native_h_px))


IMX219_PROFILE = CameraProfile(
    name="imx219",
    focal_length_mm=3.04,
    sensor_w_mm=3.68,
    sensor_h_mm=2.76,
    native_w_px=3280,
    native_h_px=2464,
)

CAMERA_REGISTRY: dict[str, CameraProfile] = {
    "imx219": IMX219_PROFILE,
}


def get_camera_profile(name: str) -> CameraProfile:
    key = str(name).strip().lower()
    if key not in CAMERA_REGISTRY:
        supported = ", ".join(sorted(CAMERA_REGISTRY.keys()))
        raise ValueError(f"Unknown camera profile '{name}'. Supported: {supported}")
    return CAMERA_REGISTRY[key]
