# Distance Sweep Augmentation

The **distance sweep** augmentation generates photorealistic versions of each detected object at multiple distances, useful for training distance-robust models and stress-testing oracle performance.

## Problem & Solution

### The Problem

Objects vary in distance from camera. A model trained only on objects at 5 ft may fail at 15 ft. Creating a diverse training set at every possible distance is expensive (manual capture, setup, etc.).

### The Solution

Given one image with known detections, we:

1. **Estimate** metric depth of the scene (monocular, no calibration)
2. **Segment** each detected object (SAM, Segment Anything)
3. **Inpaint** a clean background plate
4. **Loop** over distances: scale each object (perspective), composite, add effects, save

Result: 6+ photorealistic variants per image at minimal cost.

## Pipeline Stages

### 1. Depth Estimation (ZoeDepth)

Monocular depth model that predicts a depth map in meters. sensVeridian converts to feet for US-based specs.

- **Model**: ZoeD_NK from [isl-org/ZoeDepth](https://github.com/isl-org/ZoeDepth)
- **Download**: ~800 MB, cached in `TORCH_HOME`
- **Metric**: Real-world meters, no calibration needed
- **Output**: Full per-pixel depth map

### 2. Object Segmentation (SAM)

Segment Anything Model (ViT-B) extracts precise object masks from detection bboxes.

- **Model**: `sam_vit_b_01ec64.pth` (~375 MB)
- **Download**: One-time, ~375 MB, recommended to pre-download
- **Output**: Binary masks for each detected object

### 3. Inpainting (LaMa)

Photorealistic background inpainting fills the areas where objects were removed.

- **Model**: `simple-lama-inpainting` package (auto-downloads)
- **Output**: Clean background plate image

### 4. Geometric Augmentation

For each distance step `d` from initial `d0` to `d_max`:

1. Compute scale factor: `s = d0 / d` (perspective scaling law)
2. Compute new object size: `new_h, new_w = scale(original_h, original_w, s)`
3. Position the scaled object to maintain image-space center (pinhole model)
4. Composite onto background plate in depth order
5. Clamp to image bounds

### 5. Effects (Optional)

- **Depth-of-Field (DoF)**: Simulate camera blur proportional to distance
- **Atmospheric Haze**: Add slight haze for far objects

### 6. Oracle Re-run (Optional)

Set `--auto-run-oracle` to re-run all oracle models on each augmented image. Creates a **distance vs. accuracy dataset** automatically.

## Specifying Initial Distance

### Automatic (ZoeDepth)

By default, ZoeDepth estimates the initial distance `d0` from the input image:

```bash
sv augment distance /path/to/image.jpg \
  --d-max-ft 10 --step-ft 1 \
  --sam-checkpoint /data3/ssharma8/model-cache/sam/sam_vit_b_01ec64.pth
```

### Manual: Global (`--d0-ft`)

If you know all objects are at the same distance:

```bash
sv augment distance /path/to/images \
  --d0-ft 5.0 \  # All objects start at 5 feet
  --d-max-ft 15 --step-ft 1 \
  --sam-checkpoint /data3/ssharma8/model-cache/sam/sam_vit_b_01ec64.pth
```

**Effect**: Skips ZoeDepth entirely, saving ~2 min/image on GPU.

### Manual: Per-image/detection (`--d0-map`)

Provide a JSON file for fine-grained control:

```bash
sv augment distance /path/to/images \
  --d0-map /path/to/d0_overrides.json \
  --d-max-ft 15 --step-ft 1 \
  --sam-checkpoint /data3/ssharma8/model-cache/sam/sam_vit_b_01ec64.pth
```

**JSON Schema** (`d0_overrides.json`):

```json
{
  "global_ft": null,
  "images": {
    "image_id_sha256_1": {
      "default_ft": 5.0,
      "detections": {
        "0": 4.5,
        "2": 6.0
      }
    },
    "image_id_sha256_2": {
      "default_ft": 10.0,
      "detections": {}
    }
  }
}
```

**Precedence**:

1. Detection-specific override (e.g., detection 0: 4.5 ft)
2. Image default override (e.g., all detections: 5 ft)
3. Global override (e.g., all images: 6 ft)
4. ZoeDepth estimate (fallback)

**Example**: If detecting a QR code at exact camera distance (e.g., 3 ft by ruler), set `detections["0"]: 3.0` to force accuracy; other detections use image default, etc.

## Output Structure

```
./cache/augmentations/
├── {parent_image_id}_{distance_ft}_{seed}.png  # Augmented image
├── ...
./cache/bg_plates/
├── {parent_image_id}_{inpainter_name}.png      # Cached background plate
├── ...
./cache/depth_maps/
├── {parent_image_id}_depth.npy                 # Metric depth map (m)
├── ...
```

Metadata is stored in DuckDB tables (`augmentations`, `image_depth_stats`, `image_bg_plates`).

## Performance

On GPU (V100 or better):

| Component | Time/image |
|-----------|-----------|
| ZoeDepth | ~2 min |
| SAM (multi-object) | ~3 sec/object |
| LaMa inpainting | ~10 sec |
| Distance loop (6 steps) | ~2 sec |
| **Total (6 augmentations)** | ~3 min |

With `--d0-ft` (no ZoeDepth): ~30 sec/image.

## Troubleshooting

### "CUDA out of memory"

Reduce `--step-ft` (fewer augmentations) or use `--d0-ft` to skip ZoeDepth:

```bash
sv augment distance /path --d0-ft 5.0 --d-max-ft 10 --step-ft 2 \
  --sam-checkpoint /data3/ssharma8/model-cache/sam/sam_vit_b_01ec64.pth
```

### "ZoeDepth checkpoint not found"

Pre-download ZoeDepth into `TORCH_HOME`:

```bash
export TORCH_HOME=/data3/ssharma8/torch-cache
python -c "from zoedepth.models import build_model; model = build_model('zoedepth_nk')"
```

### "Missing SAM checkpoint"

```bash
mkdir -p /data3/ssharma8/model-cache/sam
wget -O /data3/ssharma8/model-cache/sam/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```
