# sensVeridian — Complete Usage Guide

A ground-truth cache and distance-aware augmentation pipeline for vision model evaluation.

sensVeridian ingests images, runs a configurable pipeline of ML oracle models on each, stores results in a queryable DuckDB cache, and can synthesize photorealistic distance-swept augmentations for robustness testing.

---

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Redis Server](#redis-server)
6. [Environment Variables](#environment-variables)
7. [CLI Reference](#cli-reference)
8. [Data Model / Schema](#data-model--schema)
9. [Sample Queries](#sample-queries)
10. [Python API](#python-api)
11. [Operational Tips](#operational-tips)
12. [Troubleshooting](#troubleshooting)

---

## Quick Reference

```bash
# One-time setup
source .venv/bin/activate
sv faces seed --n 8 --clear-first

# Ingest images through all oracle models
sv ingest /path/to/images --run-id baseline

# See what's in the cache
sv stats
sv query "select * from v_image_summary_wide limit 10"

# Generate distance-swept augmentations (GPU recommended)
sv augment distance /path/to/image.png \
    --d-max-ft 10 --step-ft 1 \
    --sam-checkpoint /data3/ssharma8/model-cache/sam/sam_vit_b_01ec64.pth \
    --auto-run-oracle

# Export ground truth as parquet
sv export --to /tmp/ground_truth.parquet
```

---

## Features

### Core Pipeline
- **Oracle ingestion**: Run multiple ML models on each image and cache structured outputs.
- **Content-addressed storage**: Every image keyed by SHA-256 of decoded pixels (dedup-friendly).
- **Tiered outputs**: Fast summary tier for filtering + raw JSON tier for per-detection metadata.
- **Versioning**: Every ingest tagged with `run_id` — keep many runs side-by-side.

### Oracle Models (in scope)
| ID        | Name                             | Weights                                                                      |
|-----------|----------------------------------|------------------------------------------------------------------------------|
| `amod`    | AutomotiveMultiObjectDetection   | `/data3/ssharma8/all-models/AutomotiveMultiObjectDetection/amod-cpnx-8.2.0.h5` |
| `qrcode`  | QRCodeDetection                  | `/data3/ssharma8/all-models/QRCode/qr-code-detection-final.h5`                |
| `fd`      | FaceDetection                    | `/data3/ssharma8/all-models/FaceDetection/fd_lnd_hp-fpga-8.1.0.h5`            |
| `fr`      | FaceRecognition                  | `/data3/ssharma8/all-models/FaceRecognition/fr-fpga-8.1.1.h5`                 |

- FR depends on FD crops; orchestrator runs FD first.
- Pluggable: add a new runner in `src/sensveridian/runners/` and register in `orchestrator.py`.

### Storage
- **DuckDB** as analytical store: one-file embedded DB with SQL, JSON columns, parquet export.
- **Redis** as registered-faces lookup for FR matching (with file-backed fallback if no Redis).

### Distance-Sweep Augmentation
- **ZoeDepth** for metric monocular depth (meters → feet).
- **SAM (Segment Anything, ViT-B)** for tight object masks.
- **LaMa** for photorealistic background inpainting (no HF auth required).
- Pinhole camera scale model: `scale = d_initial / (d_initial + Δ)`.
- Depth-sorted compositing ensures correct occlusion.
- Optional: DoF blur + atmospheric haze that scale with distance.
- Output images are first-class: they get an `image_id` and live in the cache with a link to parent.

---

## Architecture

```
CLI (sv) → Orchestrator → {AMOD, QRCode, FD, FR} → DuckDB
                                  ↓
                         FR matches faces against Redis/file registry

CLI (sv augment) → DistanceAugmentor → {ZoeDepth, SAM, LaMa} → Cache
                                  ↓
                         optional oracle-rerun on outputs
```

---

## Installation

Assumes Python 3.10 from `/data3/ssharma8/py310`.

```bash
cd /data3/ssharma8/projects/lattice-internal/sensVeridian

# Use existing interpreter directly
/data3/ssharma8/py310/bin/python -m pip install -e .

# Or set a project venv
/data3/ssharma8/py310/bin/python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### External Assets
- **SAM checkpoint** (~375 MB, one-time download):
  ```bash
  mkdir -p /data3/ssharma8/model-cache/sam
  wget -O /data3/ssharma8/model-cache/sam/sam_vit_b_01ec64.pth \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
  ```
- **ZoeDepth & LaMa** are pulled on first run into `TORCH_HOME`.

---

## Redis Server

The registered-faces DB used by FR is a Redis hash + set index. The project
ships with a self-contained, per-project Redis that lives entirely under
`/data3/ssharma8` — no system package, no `sudo`, no writes to `/home/ssharma8`.

**Binary**: `/data3/ssharma8/py310/bin/redis-server` (installed via
`conda install -p /data3/ssharma8/py310 -c conda-forge --override-channels -y redis-server`).

**Config**: `cache/redis/redis.conf`
- Binds `127.0.0.1:6379` only (protected-mode on, no network exposure).
- Data dir: `cache/redis/data/` (RDB snapshots + AOF for durability).
- Log: `cache/redis/redis.log`, PID: `cache/redis/redis.pid`.
- `maxmemory 256mb` with `noeviction` (faces must never be silently dropped).

### Start / stop / status

```bash
cd /data3/ssharma8/projects/lattice-internal/sensVeridian

# Start (idempotent; returns 0 if already running, incl. an external instance)
./scripts/redis_start.sh

# Status (exit 0 if running)
./scripts/redis_status.sh

# Stop (clean SHUTDOWN via redis-cli, SIGTERM/SIGKILL as fallback)
./scripts/redis_stop.sh
```

Override host/port for a non-default setup:

```bash
SV_REDIS_HOST=127.0.0.1 SV_REDIS_PORT=6380 ./scripts/redis_start.sh
export SV_REDIS_URL=redis://127.0.0.1:6380/0
```

### One-time migration from the file fallback

If you used sensVeridian before Redis was running, dummy entries were persisted
to `cache/faces_registry.json`. After starting Redis, migrate them with:

```bash
/data3/ssharma8/py310/bin/python scripts/migrate_faces_to_redis.py
```

The script re-registers each entry in Redis and archives the JSON as
`cache/faces_registry.json.migrated-<timestamp>` so the file fallback can’t
silently shadow the live Redis data on later runs.

### Verifying Redis is being used

```bash
# Ping the server
/data3/ssharma8/py310/bin/redis-cli PING             # -> PONG

# Inspect registered-faces keys
/data3/ssharma8/py310/bin/redis-cli --scan --pattern 'face:registered:*'
/data3/ssharma8/py310/bin/redis-cli SMEMBERS face:registered:index

# Confirm FaceRegistry mode
PYTHONPATH=src /data3/ssharma8/py310/bin/python -c "
from sensveridian.store.faces_registry import FaceRegistry
from sensveridian.config import SETTINGS
r = FaceRegistry(redis_url=SETTINGS.redis_url)
print('mode:', r.mode, 'entries:', len(r.list_entries()))
"
# Expected: mode: redis  entries: N
```

### Key schema used in Redis

| Key                                | Type | Fields / members                             |
|-----------------------------------|------|----------------------------------------------|
| `face:registered:index`           | set  | all registered `person_id`s                  |
| `face:registered:<person_id>`     | hash | `name` (utf-8), `embedding` (raw float32 bytes) |

### File fallback (still supported)

If Redis is unreachable at `SV_REDIS_URL`, `FaceRegistry` transparently falls
back to `cache/faces_registry.json` so local development still works. Once
Redis is up, run the migration script above to promote those entries and
archive the JSON.

---

## Environment Variables

Keep everything under `/data3/ssharma8` to avoid writing to `/home`.

| Variable                     | Purpose                              | Suggested Value                      |
|-----------------------------|--------------------------------------|--------------------------------------|
| `SV_REDIS_URL`               | Redis URL for face registry          | `redis://localhost:6379/0` (default) |
| `SV_REDIS_HOST`              | Host used by `scripts/redis_*.sh`    | `127.0.0.1` (default)                |
| `SV_REDIS_PORT`              | Port used by `scripts/redis_*.sh`    | `6379` (default)                     |
| `SV_FACE_MATCH_THRESHOLD`    | Cosine threshold for FR match        | `0.5` (default)                      |
| `SV_DEVICE`                  | Device for torch models              | `cuda` or `cpu`                      |
| `CUDA_VISIBLE_DEVICES`       | Pin to a GPU                         | `1`                                  |
| `HOME`                       | Where Torch/HF caches default        | `/data3/ssharma8/runtime-home`       |
| `HF_HOME`                    | HuggingFace cache                    | `/data3/ssharma8/hf-cache`           |
| `HUGGINGFACE_HUB_CACHE`      | Same as above                        | `/data3/ssharma8/hf-cache`           |
| `TRANSFORMERS_CACHE`         | transformers model cache             | `/data3/ssharma8/hf-cache`           |
| `TORCH_HOME`                 | torch.hub / ZoeDepth / LaMa cache    | `/data3/ssharma8/torch-cache`        |
| `XDG_CACHE_HOME`             | Misc                                 | `/data3/ssharma8/xdg-cache`          |
| `PYTORCH_ALLOC_CONF`         | Memory tuning                        | `expandable_segments:True`           |

Convenience one-liner:

```bash
export CUDA_VISIBLE_DEVICES=1 SV_DEVICE=cuda \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  HOME=/data3/ssharma8/runtime-home \
  HF_HOME=/data3/ssharma8/hf-cache \
  HUGGINGFACE_HUB_CACHE=/data3/ssharma8/hf-cache \
  TRANSFORMERS_CACHE=/data3/ssharma8/hf-cache \
  TORCH_HOME=/data3/ssharma8/torch-cache \
  XDG_CACHE_HOME=/data3/ssharma8/xdg-cache \
  PYTHONPATH=/data3/ssharma8/projects/lattice-internal/sensVeridian/src
```

---

## CLI Reference

Entrypoint: `sv` (installed via `pyproject.toml`) or `python -m sensveridian.cli`.

### Top-level commands

| Command          | What it does                                             |
|------------------|----------------------------------------------------------|
| `sv ingest`      | Run oracles on images and write to cache                 |
| `sv query`       | Execute arbitrary SQL against DuckDB                     |
| `sv export`      | Export a SQL result to parquet                           |
| `sv stats`       | Show row counts for all tables                           |
| `sv faces ...`   | Face registry subcommands                                |
| `sv augment ...` | Augmentation subcommands                                 |

### `sv ingest`

Run the oracle pipeline on a folder (recursive) or a single file.

```
sv ingest <image_root>
  [--run-id STR]           Run identifier (default: "baseline")
  [--models STR]           Comma list of model IDs (default: "amod,qrcode,fd,fr")
  [--skip-existing BOOL]   Skip (image_id, run_id) already processed (default: True)
```

Examples:

```bash
# Baseline ingest, all four models
sv ingest /data3/ssharma8/all-models/Images/qr_code --run-id baseline

# Only AMOD + QR on a folder, force reprocess
sv ingest /data3/ssharma8/all-models/Images --models amod,qrcode --skip-existing False

# Custom run tag for an experiment
sv ingest /data3/ssharma8/all-models/Images --run-id exp_2026_04_24_cpu
```

### `sv query`

Run any SQL against the live DuckDB file.

```
sv query "<SQL>"
```

Examples:

```bash
sv query "select * from v_image_summary_wide limit 5"
sv query "select model_id, count(*) from predictions_summary group by model_id"
sv query "select image_id, present, count from predictions_summary where model_id='qrcode'"
```

### `sv export`

Export a SQL result to parquet for pandas/analytics pipelines.

```
sv export --to <path.parquet> [--sql "<SQL>"]
```

Defaults to exporting `SELECT * FROM v_image_summary_wide` if no SQL is given.

Examples:

```bash
# Full summary view
sv export --to /tmp/ground_truth_wide.parquet

# Only QR-positive images
sv export --to /tmp/qr_positive.parquet \
  --sql "SELECT image_id, path, n_qrc FROM v_image_summary_wide WHERE qrc_present"

# Raw detections for a specific run
sv export --to /tmp/raw_baseline.parquet \
  --sql "SELECT * FROM predictions_raw WHERE run_id='baseline'"
```

### `sv stats`

Quick count of rows across all tables.

```bash
sv stats
# images: 4
# runs: 1
# predictions_summary: 16
# predictions_raw: 16
# augmentations: 5
```

### `sv faces`

Manage the registered-faces lookup used by FR.

```
sv faces seed [--n INT] [--embedding-dim INT] [--clear-first]
sv faces list
sv faces clear
```

- Redis-first; falls back to a JSON file under `cache/faces_registry.json` if Redis is down. See [Redis Server](#redis-server) for starting the per-project instance and migrating fallback entries.
- Dummy embeddings are random unit vectors. Safe to seed any number; real FR embeddings will dominate as they are added.

Examples:

```bash
# Seed 8 dummy identities (clears first)
sv faces seed --n 8 --clear-first

# Inspect registered identities
sv faces list

# Seed with a custom embedding dim (e.g., 512)
sv faces seed --n 16 --embedding-dim 512 --clear-first
```

### `sv augment distance`

Generate distance-swept versions of each detected object on an image.

```
sv augment distance <image_or_folder>
  --d-max-ft FLOAT            Upper distance threshold in feet (required)
  --step-ft FLOAT             Distance step in feet (required)
  --sam-checkpoint PATH       SAM .pth checkpoint path (required)
  [--source-models STR]       Detection sources (default: "amod,fd,qrcode")
  [--run-id STR]              Run tag for auto oracle rerun (default: "augmented")
  [--auto-run-oracle]         Re-run oracles on generated images to populate GT
  [--d0-ft FLOAT]             Manual initial distance (feet) applied to every detection
  [--d0-map PATH]             JSON file with per-image / per-detection overrides
```

**Initial distance** (`d_initial_ft`) is resolved with this precedence, high → low:

1. Per-detection override from `--d0-map` (`images.<key>.detections["<model_id>:<idx>"]`)
2. Per-image default from `--d0-map` (`images.<key>.default`)
3. Global `--d0-ft`
4. **ZoeDepth fallback** (median of the per-pixel metric depth map inside the detection's bbox)

The source is recorded per-detection in `image_depth_stats.source` (`'manual'` or `'zoe'`). When every detection is covered by a manual value, ZoeDepth is **not** loaded at all — no GPU memory, no download, no inference.

The `--d0-map` JSON keys for images can be any of: absolute path, basename, basename stem, or the `image_id` (sha256 of decoded pixels).

```json
{
  "global_ft": 5.0,
  "images": {
    "gard_00000.png":               {"default": 4.5},
    "/data3/…/Images/qr/qr_01.png": {"default": 3.2, "detections": {"amod:0": 3.0, "qrcode:0": 3.1}},
    "329acdb6e9850f346b06d04fd73aec48f28d793e5c8…": {"default": 6.0}
  }
}
```

Output images go under `cache/augmentations/<run_id>_<8-hex>/`.

Each output gets an `augmentations` row linking back to `parent_image_id`; `params` also records `n_objects`, `n_manual_distance`, and `n_zoe_distance` so you can tell per-step how many detections used ZoeDepth vs. a manual value.

Examples:

```bash
# One image, 5ft -> 10ft at 1ft steps
sv augment distance /data3/ssharma8/all-models/Images/qr_code/gard_00000.png \
  --d-max-ft 10 --step-ft 1 \
  --sam-checkpoint /data3/ssharma8/model-cache/sam/sam_vit_b_01ec64.pth

# Whole folder, full ground truth populated on outputs too
sv augment distance /data3/ssharma8/all-models/Images \
  --d-max-ft 15 --step-ft 2 \
  --sam-checkpoint /data3/ssharma8/model-cache/sam/sam_vit_b_01ec64.pth \
  --auto-run-oracle --run-id distance_eval_2026_04

# Only use AMOD as the source of detections (skip face/QR boxes)
sv augment distance /path/to/imgs \
  --d-max-ft 20 --step-ft 1 --source-models amod \
  --sam-checkpoint /data3/ssharma8/model-cache/sam/sam_vit_b_01ec64.pth

# Known capture distance: bench tests shot at exactly 5 ft; skip ZoeDepth entirely
sv augment distance /data3/ssharma8/all-models/Images/qr_code \
  --d-max-ft 10 --step-ft 1 --d0-ft 5.0 \
  --sam-checkpoint /data3/ssharma8/model-cache/sam/sam_vit_b_01ec64.pth

# Mixed: precise per-image (and per-detection) distances from a JSON file
sv augment distance /data3/ssharma8/all-models/Images \
  --d-max-ft 15 --step-ft 1 \
  --d0-map /data3/ssharma8/projects/lattice-internal/sensVeridian/cache/d0_overrides.json \
  --sam-checkpoint /data3/ssharma8/model-cache/sam/sam_vit_b_01ec64.pth
```

### `sv augment list`

List augmentations for a given parent image.

```
sv augment list <parent_image_id>
```

Example:

```bash
sv augment list 329acdb6e9850f346b06d04fd73aec48f28d793e5c8...
```

---

## Data Model / Schema

All tables live in `sensveridian.duckdb` at the project root.

| Table                  | Purpose                                                    |
|-----------------------|------------------------------------------------------------|
| `images`              | Registry of ingested images (image_id = sha256 of pixels). |
| `models`              | Metadata for each oracle model and weights.                |
| `runs`                | Named runs for versioning the cache.                       |
| `predictions_summary` | Fast tier: `present, count, extras` per (image, run, model). |
| `predictions_raw`     | Full tier: JSON payloads with bboxes, keypoints, embeddings. |
| `augmentations`       | Links augmented images to their parents.                   |
| `image_depth_stats`   | Cached per-detection initial depth in feet; `source` = `'manual'` or `'zoe'`. |
| `image_bg_plates`     | On-disk inpainted background plates.                       |
| `v_image_summary_wide`| View that pivots summary into one row per image.           |

Full schema is at `src/sensveridian/store/schema.sql`.

---

## Sample Queries

### Basic inspection

```sql
-- Count everything
SELECT
  (SELECT count(*) FROM images) AS images,
  (SELECT count(*) FROM runs) AS runs,
  (SELECT count(*) FROM predictions_summary) AS summaries,
  (SELECT count(*) FROM augmentations) AS augs;

-- The user-shaped "dict" view (one row per image)
SELECT image_id, amod_present, n_amod, qrc_present, n_qrc, fd_present, n_fd, fid_present, n_fid
FROM v_image_summary_wide;
```

### Filtering by model outputs

```sql
-- Images with QR codes but no faces
SELECT path FROM v_image_summary_wide WHERE qrc_present AND NOT fd_present;

-- Images where at least one registered face was recognized
SELECT image_id, n_fid FROM v_image_summary_wide WHERE fid_present;

-- Images with more than 50 face boxes (candidate crowd scenes)
SELECT path, n_fd FROM v_image_summary_wide WHERE n_fd > 50 ORDER BY n_fd DESC;
```

### Run / model metadata

```sql
-- List known runs
SELECT * FROM runs;

-- Weights provenance for oracles
SELECT model_id, version, weights_path, weights_sha FROM models;

-- Counts per run per model
SELECT run_id, model_id, count(*) AS n
FROM predictions_summary GROUP BY run_id, model_id ORDER BY run_id, model_id;
```

### Raw payload drill-down (JSON operators)

```sql
-- Pull bboxes out of raw AMOD detections
SELECT image_id,
       payload->>'$.detections' AS detections_json
FROM predictions_raw
WHERE model_id = 'amod'
LIMIT 5;

-- Explode QR decoded text counts
SELECT image_id,
       json_array_length(payload->'$.decoded_texts') AS decoded_count
FROM predictions_raw
WHERE model_id = 'qrcode';
```

### Augmentations

```sql
-- All augmentations grouped by parent
SELECT parent_image_id,
       count(*) AS n_aug,
       min(delta_ft) AS min_delta,
       max(delta_ft) AS max_delta
FROM augmentations GROUP BY parent_image_id;

-- Show just the parameters used per augmentation
SELECT augmented_image_id, step_index, delta_ft, params
FROM augmentations ORDER BY parent_image_id, step_index;

-- Chain: augmented images' oracle outputs (requires --auto-run-oracle at time of augment)
SELECT a.parent_image_id, a.delta_ft, s.model_id, s.present, s.count
FROM augmentations a
JOIN predictions_summary s ON a.augmented_image_id = s.image_id
WHERE s.run_id = 'augmented'
ORDER BY a.parent_image_id, a.delta_ft, s.model_id;
```

### Depth cache

```sql
-- Per-detection initial depth (feet) with provenance
SELECT image_id, model_id, detection_idx, d_initial_ft, source, bbox_xyxy
FROM image_depth_stats
ORDER BY image_id, model_id, detection_idx;

-- How many detections used the ZoeDepth fallback vs. a manual value
SELECT source, count(*) AS dets,
       avg(d_initial_ft) AS avg_ft,
       min(d_initial_ft) AS min_ft,
       max(d_initial_ft) AS max_ft
FROM image_depth_stats
GROUP BY source;

-- Only the detections where ZoeDepth is carrying the estimate
SELECT image_id, model_id, detection_idx, d_initial_ft
FROM image_depth_stats
WHERE source = 'zoe'
ORDER BY d_initial_ft DESC;
```

> `image_depth_stats` has a primary key on `(image_id, model_id, detection_idx)`, so it only stores the most recent write per detection. Re-running `sv augment distance` with `--d0-ft` on the same image will overwrite a previously ZoeDepth-derived row and flip `source` to `'manual'`.

### Export patterns

```bash
# Export distance-vs-accuracy eval set to parquet
sv export --to /tmp/distance_eval.parquet --sql "
  SELECT a.parent_image_id, a.delta_ft, a.augmented_image_id,
         s.model_id, s.present, s.count
  FROM augmentations a
  JOIN predictions_summary s ON a.augmented_image_id = s.image_id
  WHERE s.run_id = 'augmented'
"
```

---

## Python API

For programmatic use outside the CLI.

```python
from pathlib import Path
from sensveridian.config import SETTINGS
from sensveridian.store.duck import DuckStore
from sensveridian.store.faces_registry import FaceRegistry
from sensveridian.orchestrator import Orchestrator
from sensveridian.augmentation.distance_sweep import DistanceAugmentor

schema = Path(__file__).parent / "src/sensveridian/store/schema.sql"
store = DuckStore(db_path=SETTINGS.db_path, schema_path=schema)
store.migrate()

registry = FaceRegistry(redis_url=SETTINGS.redis_url)
orch = Orchestrator(store=store, registry=registry)

res = orch.ingest(
    image_root=Path("/data3/ssharma8/all-models/Images/qr_code"),
    run_id="baseline",
    selected_models={"amod", "qrcode", "fd", "fr"},
    skip_existing=True,
)
print(res)

aug = DistanceAugmentor(
    store=store,
    orchestrator=orch,
    sam_checkpoint="/data3/ssharma8/model-cache/sam/sam_vit_b_01ec64.pth",
    device="cuda",
)
generated = aug.augment_image(
    image_path=Path("/data3/ssharma8/all-models/Images/qr_code/gard_00000.png"),
    run_id="augmented",
    d_max_ft=10.0,
    step_ft=1.0,
    source_models=["amod", "fd", "qrcode"],
    out_dir=SETTINGS.cache_dir / "augmentations" / "manual",
    auto_run_oracle=True,
)
print("Generated:", generated)
```

---

## Operational Tips

- **Use a specific GPU**: `export CUDA_VISIBLE_DEVICES=1` before augmentation commands. Ingest fits on CPU but SAM+LaMa+ZoeDepth is much faster on GPU.
- **Keep caches off `/home`**: set `HOME`, `HF_HOME`, `TORCH_HOME`, `XDG_CACHE_HOME` under `/data3/ssharma8/...`.
- **Re-run safely**: `sv ingest` is idempotent per `(image_id, run_id, model_id)`; use `--skip-existing False` to force overwrite.
- **Compare runs**: ingest the same folder with two `run_id` values, then diff `predictions_summary` across runs.
- **Parquet export is king** for downstream notebooks — DuckDB + `pandas.read_parquet` is the cleanest path.
- **Registered-faces store**: prefer the per-project Redis — start it with `./scripts/redis_start.sh`. If Redis is down, FR transparently falls back to `cache/faces_registry.json`; run `scripts/migrate_faces_to_redis.py` to promote those entries once Redis is up.
- **Adding a new oracle**:
  1. Create `src/sensveridian/runners/<name>.py` implementing the `Runner` protocol from `runners/base.py`.
  2. Register it in `src/sensveridian/orchestrator.py`.
  3. Add a pivot column in `v_image_summary_wide` (optional).

---

## Troubleshooting

| Symptom                                                      | Fix                                                                       |
|-------------------------------------------------------------|---------------------------------------------------------------------------|
| `ModuleNotFoundError: duckdb / typer / redis ...`           | `pip install -e .` in the project root.                                   |
| `Unknown layer: sensAI>QuantizeConv2D`                      | Already handled — loader uses `lscquant.layers` custom objects.           |
| `CUDA out of memory`                                         | Pin `CUDA_VISIBLE_DEVICES` to a free GPU; or set `SV_DEVICE=cpu`.         |
| `RuntimeError: Error(s) in loading state_dict for ZoeDepthNK` | Pin `timm==0.6.12` (ZoeDepth is incompatible with newer timm checkpoints). |
| `ConnectionRefusedError: ... 6379`                          | Start the per-project Redis: `./scripts/redis_start.sh`. Registry auto-falls back to `cache/faces_registry.json` if unreachable. |
| Redis started but `sv faces list` is empty                   | Previous entries live in `cache/faces_registry.json`. Run `python scripts/migrate_faces_to_redis.py` to promote them. |
| `401 Unauthorized` from HuggingFace                          | Not an issue anymore — we use LaMa, no HF auth required.                  |
| Very slow first augmentation                                 | ZoeDepth (1.35 GB) and LaMa (196 MB) are downloaded once to `TORCH_HOME`. |

---

## File Layout

```
sensVeridian/
├── pyproject.toml
├── README.md
├── USAGE.md                           <-- this file
├── sensveridian.duckdb                <-- created on first run
├── cache/
│   ├── augmentations/<run_id>_<hex>/  <-- generated images
│   ├── bg_plates/                     <-- inpainted backgrounds
│   ├── faces_registry.json            <-- file fallback if no Redis
│   └── redis/
│       ├── redis.conf                 <-- per-project Redis config
│       ├── redis.log / redis.pid      <-- runtime files
│       └── data/                      <-- RDB snapshots + AOF
├── scripts/
│   ├── redis_start.sh                 <-- start per-project Redis
│   ├── redis_stop.sh                  <-- stop per-project Redis
│   ├── redis_status.sh                <-- status / key count
│   └── migrate_faces_to_redis.py      <-- promote file fallback -> Redis
├── src/sensveridian/
│   ├── cli.py                         <-- all commands
│   ├── orchestrator.py                <-- pipeline executor
│   ├── config.py                      <-- paths, env, thresholds
│   ├── hashing.py                     <-- sha256 helpers
│   ├── runners/                       <-- oracle runners
│   ├── augmentation/                  <-- depth/segment/inpaint/distance_sweep
│   └── store/                         <-- DuckDB + Redis
└── tests/
```
