# Frame Miniaturization Augmentation

Frame miniaturization simulates increased camera distance by uniformly shrinking the entire frame,
then placing it back into the original canvas with controlled padding.

This is a lightweight deterministic alternative to the inpaint+composite pipeline.

## Core Geometry

If the baseline apparent distance is `d0` and we simulate a farther view by `delta`, the scale is:

`s = d0 / (d0 + delta)`

Then:

- resize frame by `s` (same for width/height)
- center it in original resolution
- fill borders by pad mode

## Why Use It

- Very fast (OpenCV only)
- Deterministic and reproducible
- No SAM/LaMa/ZoeDepth dependency in pure manual mode
- Good for CI and broad robustness sweeps

## CLI

```bash
sv augment miniaturize /path/to/images \
  --d-max-ft 15 --step-ft 1 \
  --pad-mode black
```

With calibration fallback:

```bash
sv augment miniaturize /path/to/images \
  --d-max-ft 15 --step-ft 1 \
  --camera imx219 \
  --pad-mode replicate
```

With manual baseline distances:

```bash
sv augment miniaturize /path/to/images \
  --d-max-ft 15 --step-ft 1 \
  --d0-ft 5.0 \
  --pad-mode black
```

## Pad Modes

- `black` (default): zero-filled border
- `replicate`: edge replication
- `reflect`: reflected border

## Distance Source Precedence

Per detection:

1. Manual per-detection (`--d0-map` `detections`)
2. Manual image default (`--d0-map` `default`)
3. Manual global (`--d0-ft` / `global_ft`)
4. Camera calibration (`--camera`)
5. ZoeDepth fallback (only when no camera profile is used)

## Stored Metadata

Rows are written into `augmentations` with:

- `method = "frame_miniaturize"`
- `delta_ft`
- params including `d0_ft`, `scale`, `pad_mode`, and source counts
