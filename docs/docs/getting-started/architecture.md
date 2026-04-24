# Architecture

## Data Flow

```
Images (folder)
    ↓
Orchestrator.ingest()
    ├─ Compute image_id (SHA256 of pixels)
    ├─ Register models & run
    └─ For each image:
        ├─ AMOD → detections
        ├─ QRCode → detections + decoded text
        ├─ FaceDetection → detections + crops
        └─ FaceRecognition (depends on FD):
            └─ Match crops against FaceRegistry (Redis/file)
    ↓
DuckDB Store
    ├─ predictions_summary (present, count, extras)
    ├─ predictions_raw (JSON payloads)
    └─ v_image_summary_wide (pivoted view)
```

## Storage Tiers

| Tier | Storage | Purpose | Query |
|------|---------|---------|-------|
| **Summary** | DuckDB `predictions_summary` | Fast aggregates (present, count) | `SELECT model_id, present, count FROM predictions_summary` |
| **Raw** | DuckDB `predictions_raw` | Full detections, embeddings, decoded text | `SELECT payload FROM predictions_raw WHERE model_id='fr'` |
| **Registry** | Redis (+ file fallback) | Face embeddings for matching | `FaceRegistry.match(embedding, threshold=0.5)` |

## Distance Augmentation Pipeline

```
Image + Detections (from oracle run)
    ↓
1. ZoeDepth Estimation
   └─ Monocular metric depth (meters → feet)
    ↓
2. SAM Segmentation
   └─ Precise object masks from bboxes
    ↓
3. LaMa Inpainting
   └─ Photorealistic background plate
    ↓
4. Distance Sweep Loop
   ├─ For each distance step d ∈ [d₀, d_max]:
   │   ├─ Scale objects by d₀/d (perspective)
   │   ├─ Depth-sort and composite onto plate
   │   ├─ Apply DoF + atmospheric effects
   │   └─ Save augmented image
   ↓
5. Optional Oracle Re-run
   └─ Run models on augmented images
    ↓
Database: augmentations, image_depth_stats, image_bg_plates
```

## Dependencies & Topological Order

```
┌─────────┐
│  AMOD   │ (no deps)
│ QRCode  │ (no deps)
└────┬────┘
     │
     └──→ ┌──────────┐
          │    FD    │ (no deps, but provides crops)
          └────┬─────┘
               │
               └──→ ┌──────────┐
                    │    FR    │ (depends on FD for crops)
                    └──────────┘
```

## Redis Modes

- **Redis mode**: Connected to `SV_REDIS_URL` (default: `redis://localhost:6379/0`)
  - Per-project server: `./scripts/redis_start.sh`
  - Key schema: `face:registered:index`, `face:registered:<person_id>`
  
- **File fallback mode**: If Redis unreachable, uses `cache/faces_registry.json`
  - Automatic fallback; no configuration needed
  - Use `./scripts/migrate_faces_to_redis.py` to promote entries to Redis
