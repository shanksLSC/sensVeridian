# Manual Distance Overrides

The distance sweep augmentation can accept manual distance specifications, allowing you to bypass ZoeDepth estimation and inject known distances for improved realism or faster processing.

## Use Cases

1. **Benchmark captures at known distance**: "These images were taken exactly 5 ft from the camera (by ruler)."
2. **Mixed studio/field data**: Use ZoeDepth for field shots, but known values for controlled studio captures.
3. **GPU memory constraints**: Skip ZoeDepth (2 min/image) to save ~120 GB VRAM and 2× speedup.
4. **Evaluation reproducibility**: Fix initial distances so augmentation is deterministic.

## Three Levels of Specification

### 1. Global (Applies to all images, all detections)

```bash
sv augment distance /path/to/images \
  --d0-ft 5.0 \
  --d-max-ft 10 --step-ft 1 \
  --sam-checkpoint /data3/ssharma8/model-cache/sam/sam_vit_b_01ec64.pth
```

**Effect**: All detected objects start at 5 ft, regardless of model. ZoeDepth skipped entirely.

**Time savings**: ~2 min/image on GPU.

### 2. Per-image (Different defaults per image)

Use a JSON file (`--d0-map`) to specify per-image distances:

```json
{
  "global_ft": null,
  "images": {
    "abc123def456...": {
      "default_ft": 5.0,
      "detections": {}
    },
    "xyz789uvw012...": {
      "default_ft": 8.0,
      "detections": {}
    }
  }
}
```

Then:

```bash
sv augment distance /path/to/images \
  --d0-map /path/to/overrides.json \
  --d-max-ft 10 --step-ft 1 \
  --sam-checkpoint /data3/ssharma8/model-cache/sam/sam_vit_b_01ec64.pth
```

### 3. Per-detection (Fine-grained control)

Specify exact distances for individual detected objects:

```json
{
  "global_ft": null,
  "images": {
    "abc123def456...": {
      "default_ft": 5.0,
      "detections": {
        "0": 4.5,
        "2": 6.0
      }
    }
  }
}
```

**Interpretation**:
- Detection index 0 → 4.5 ft (exact override)
- Detection index 1 → 5.0 ft (image default)
- Detection index 2 → 6.0 ft (exact override)
- Any others → 5.0 ft (image default)
- Or ZoeDepth if no image default set

## Precedence Order

When augmenting, distance lookup follows this priority:

1. **Detection-specific override** (if present in `detections.<index>`)
2. **Image-level default** (if present in `images.<image_id>.default_ft`)
3. **Global override** (if `--d0-ft` flag provided)
4. **ZoeDepth fallback** (if no manual override found)

## Building the Override JSON

### Option A: Generate from metadata

If you have a CSV with image IDs and distances:

```python
import csv
import json

overrides = {"global_ft": None, "images": {}}

with open('distances.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        image_id = row['image_id']
        distance_ft = float(row['distance_ft'])
        overrides['images'][image_id] = {
            "default_ft": distance_ft,
            "detections": {}
        }

with open('d0_overrides.json', 'w') as f:
    json.dump(overrides, f, indent=2)
```

### Option B: Partial overrides with defaults

Specify global default, then override specific images:

```json
{
  "global_ft": 5.0,
  "images": {
    "studio_shot_1": {
      "default_ft": 3.0,
      "detections": {}
    },
    "studio_shot_2": {
      "default_ft": 3.0,
      "detections": {
        "0": 2.9,
        "1": 3.1
      }
    }
  }
}
```

**Effect**:
- studio_shot_1, studio_shot_2 use image-level distances
- All others use global 5.0 ft
- studio_shot_2 detection 0 uses exact 2.9 ft

## Example: Mixed Benchmark

Suppose you have:
- **Field images** (5 images): Use ZoeDepth (unknown distances)
- **Studio images** (10 images): Known exactly 3 ft

```json
{
  "global_ft": null,
  "images": {
    "studio_00001": {"default_ft": 3.0, "detections": {}},
    "studio_00002": {"default_ft": 3.0, "detections": {}},
    "studio_00003": {"default_ft": 3.0, "detections": {}},
    ...
    "studio_00010": {"default_ft": 3.0, "detections": {}}
  }
}
```

Then:

```bash
sv augment distance /benchmark/images \
  --d0-map studio_overrides.json \
  --d-max-ft 15 --step-ft 1 \
  --sam-checkpoint /data3/ssharma8/model-cache/sam/sam_vit_b_01ec64.pth
```

Result:
- Studio images: skip ZoeDepth, use 3 ft → 15 ft in 1 ft steps (13 images each)
- Field images: run ZoeDepth, use estimated depth (varies per image)

## Validation

After generating the JSON, validate syntax:

```bash
python3 -c "import json; json.load(open('d0_overrides.json'))" && echo "Valid JSON"
```

Then check that it matches your image set:

```bash
sv query "SELECT COUNT(DISTINCT image_id) FROM images" | \
  python3 -c "import sys, json; data = json.load(open('d0_overrides.json')); \
    print(f'Images in DB: {sys.stdin.read()}')"
```

## Output Metadata

After augmentation with overrides, inspect depth stats:

```bash
sv query "
  SELECT image_id, source, n_manual_distance, n_zoe_distance, median_depth_ft
  FROM image_depth_stats
  ORDER BY source
"
```

Expected:
- `source = 'manual'`: Used your override (n_zoe_distance = 0)
- `source = 'zoe'`: Used ZoeDepth (n_manual_distance = 0)
- `source = 'mixed'`: Some manual, some ZoeDepth
