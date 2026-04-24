# Features

## Core Pipeline

1. **Image Ingestion**: Recursive folder scan with format detection (.jpg, .png, .bmp, .webp)
2. **Oracle Models**: Run selected subset of 4 configurable models
3. **Result Storage**: Summary + raw JSON tiers in DuckDB
4. **Face Matching**: Redis-backed or file-fallback registry with cosine similarity
5. **Query API**: SQL against DuckDB; export to Parquet for analytics

## Oracle Models (in scope)

| Model | Version | Input | Output | Notes |
|-------|---------|-------|--------|-------|
| AMOD | 8.2.0 | RGB image | Bboxes + class IDs + confidence | Automotive multi-object detection |
| QRCode | final | Grayscale | Bboxes + decoded text | Text extraction + barcode points |
| FD | 8.1.0 | RGB image | Bboxes + face crops | Dependency: provides crops for FR |
| FR | 8.1.1 | Face crop | 128-dim embedding | Matches against registered faces |

## Storage

- **DuckDB** — Queryable ground-truth cache (single-file, embedded, no server)
- **Redis** — Registered-faces lookup (with file-backed fallback if unavailable)
- **Filesystem** — Augmented images, inpainted backgrounds, depth maps

## Distance-Sweep Augmentation

1. **Monocular Depth Estimation** — ZoeDepth for metric depth (no calibration needed)
2. **Object Segmentation** — SAM (Segment Anything) for precise masks
3. **Photorealistic Inpainting** — LaMa for clean background plates
4. **Geometric Augmentation** — Pinhole camera scaling + depth-sorted compositing
5. **Effects** — Optional Depth-of-Field blur and atmospheric haze
6. **Manual Overrides** — Specify known distances per-image or per-detection (ZoeDepth fallback)

## Optional: Auto-Oracle Re-run

After generating augmented images, optionally re-run oracle models:

```bash
sv augment distance /path/to/imgs \
  --d-max-ft 10 --step-ft 1 \
  --auto-run-oracle --run-id distance_eval
```

Creates a **distance vs. accuracy evaluation dataset** automatically.
