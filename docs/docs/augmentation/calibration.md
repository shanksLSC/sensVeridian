# Camera Calibration Distance Estimation

Camera calibration mode uses a pinhole-camera estimate instead of ZoeDepth for unresolved detections.
It is useful when you know the camera profile (for example Sony IMX219) and want deterministic, fast
distance initialization without loading a depth model.

## Formula

For a pinhole camera:

`distance_m = (real_size_m * f_px) / bbox_size_px`

- `real_size_m`: object real-world height/width in meters
- `f_px`: focal length in pixels at the current image resolution
- `bbox_size_px`: detection box size in pixels (height preferred, width fallback)

sensVeridian converts meters to feet with:

`distance_ft = distance_m * 3.28084`

## Built-in Sony IMX219 Profile

| Field | Value |
|---|---|
| Camera name | `imx219` |
| Focal length | `3.04 mm` |
| Sensor size | `3.68 x 2.76 mm` |
| Native resolution | `3280 x 2464` |
| Pixel pitch | `1.12 um` |
| Native `fx`, `fy` | `~2710.87 px` |

For different capture resolutions, `fx/fy` are scaled from native dimensions.

## CLI Usage

```bash
sv augment distance /path/to/images \
  --d-max-ft 15 --step-ft 1 \
  --camera imx219 \
  --sam-checkpoint /data3/ssharma8/model-cache/sam/sam_vit_b_01ec64.pth
```

Optional native mode override:

```bash
sv augment distance /path/to/images \
  --d-max-ft 15 --step-ft 1 \
  --camera imx219 \
  --camera-native 1640x1232 \
  --sam-checkpoint /data3/ssharma8/model-cache/sam/sam_vit_b_01ec64.pth
```

When `--camera` is set:

- Manual overrides still take precedence
- Calibration is used for unresolved detections
- ZoeDepth is not loaded

## Size Priors (Defaults)

Calibration needs real object size. sensVeridian supports:

1. Class priors (default)
2. Per-detection size override in `--d0-map`

Default class priors:

| Model/Class | Height (m) | Width (m) |
|---|---:|---:|
| `fd:0` (face) | 0.22 | 0.16 |
| `qrcode:0` | 0.10 | 0.10 |
| `amod:0` (person) | 1.70 | 0.45 |
| `amod:1` (car) | 1.45 | 1.80 |
| `amod:2` (motorcycle) | 1.10 | 0.75 |
| `amod:3` (bicycle) | 1.20 | 0.50 |

## Per-Detection Real Size Override

Add `real_sizes_m` in the same image override block used by `--d0-map`:

```json
{
  "global_ft": null,
  "images": {
    "door_01.jpg": {
      "default": 5.2,
      "detections": {"amod:0": 4.5},
      "real_sizes_m": {
        "amod:0": {"h": 1.78, "w": 0.48},
        "qrcode:0": {"h": 0.15, "w": 0.15}
      }
    }
  }
}
```

Precedence remains:

1. Manual per-detection distance (`detections`)
2. Manual per-image default (`default`)
3. Manual global distance (`global_ft` / `--d0-ft`)
4. Camera calibration (`--camera`)
5. ZoeDepth (only when camera calibration is not enabled)

## Adding a New Camera

Add a new `CameraProfile` entry in:

- `src/sensveridian/augmentation/camera.py`

Then register it in `CAMERA_REGISTRY` so `--camera <name>` can resolve it.
