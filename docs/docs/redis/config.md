# Configuration

Reference guide for sensVeridian configuration options, both environment variables and hardcoded defaults.

## File: `src/sensveridian/config.py`

### ModelPaths

Paths to the four oracle models (`.h5` files).

```python
class ModelPaths:
    amod = "/data3/ssharma8/all-models/AutomotiveMultiObjectDetection/amod-cpnx-8.2.0.h5"
    qrcode = "/data3/ssharma8/all-models/QRCode/qr-code-detection-final.h5"
    fd = "/data3/ssharma8/all-models/FaceDetection/fd_lnd_hp-fpga-8.1.0.h5"
    fr = "/data3/ssharma8/all-models/FaceRecognition/fr-fpga-8.1.1.h5"
```

To override at runtime, edit `src/sensveridian/config.py` or patch via Python:

```python
from sensveridian import config
config.ModelPaths.amod = "/custom/path/to/amod.h5"
```

### Settings

Global configuration dataclass.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `db_path` | str | `./sensveridian.duckdb` | DuckDB file path |
| `redis_url` | str | `redis://localhost:6379/0` | Redis connection string |
| `cache_dir` | str | `./cache` | Cache directory (augmentations, depth maps, etc.) |
| `face_registry_fallback` | str | `./cache/faces_registry.json` | File fallback if Redis down |
| `depth_estimation_default` | float | `5.0` | Default distance in feet if no ZoeDepth/override |
| `max_workers` | int | `4` | Thread pool size for parallel ops |

**Override via environment**:

```bash
export SV_DB_PATH=/mnt/data/sensveridian.duckdb
export SV_REDIS_URL=redis://localhost:6380/0
sv stats  # Uses new paths
```

**Or in Python**:

```python
from sensveridian.config import SETTINGS
SETTINGS.db_path = "/custom/db.duckdb"
SETTINGS.redis_url = "redis://remote:6379/0"
```

## Distance Augmentation Tuning

In `src/sensveridian/augmentation/distance_sweep.py`:

```python
# Defaults for geometric scaling
DEFAULT_DOF_BLUR_STRENGTH = 0.5  # Blur amount (0-1)
DEFAULT_HAZE_STRENGTH = 0.3      # Atmospheric haze (0-1)

# Object sizing constraints
MIN_OBJECT_SIZE_PX = 10          # Don't scale below this
MAX_OBJECT_SIZE_PX = 2000        # Don't scale above this
```

To customize, edit the file or patch at runtime:

```python
import sensveridian.augmentation.distance_sweep as ds
ds.DEFAULT_DOF_BLUR_STRENGTH = 0.8
```

## Detection Thresholds

Oracle model confidence thresholds (in respective runners):

| Model | Threshold | File |
|-------|-----------|------|
| AMOD | 0.4 | `src/sensveridian/runners/amod.py` |
| QRCode | (N/A, decoded) | `src/sensveridian/runners/qrcode.py` |
| FD | 0.5 | `src/sensveridian/runners/face_detection.py` |
| FR | 0.6 (cosine) | `src/sensveridian/runners/face_recognition.py` |

Adjust thresholds to trade recall for precision:

```python
# In amod.py
CONFIDENCE_THRESHOLD = 0.5  # Default 0.4; raise to 0.5 for fewer false positives
```

## Face Registry Settings

In `src/sensveridian/store/faces_registry.py`:

| Setting | Default | Purpose |
|---------|---------|---------|
| `EMBEDDING_NORM` | `L2` | Normalization for face embeddings |
| `MATCH_THRESHOLD` | `0.6` | Cosine similarity threshold for match |
| `REDIS_KEY_PREFIX` | `face:` | Redis key namespace |

## CLI Defaults

In `src/sensveridian/cli.py`:

```python
@app.command()
def ingest(
    image_root: str,
    run_id: str = "baseline",
    models: str = "amod,qrcode,fd,fr",  # <- Default models
    skip_existing: bool = True,          # <- Default behavior
):
    ...

@app.command()
def augment_distance(
    image_root: str,
    d_max_ft: float = typer.Option(...),
    step_ft: float = typer.Option(...),
    source_models: str = "amod,fd,qrcode",  # <- Default sources
    run_id: str = "augmented",               # <- Default run tag
    auto_run_oracle: bool = False,           # <- Default: no re-run
):
    ...
```

To change defaults, edit `cli.py` or pass explicit flags:

```bash
sv ingest /path --models amod --skip-existing False
```

## Logging

Set `LOGLEVEL` to control verbosity:

```bash
export LOGLEVEL=DEBUG
sv ingest /path --run-id test
```

| Level | Used for |
|-------|----------|
| `DEBUG` | Detailed traces (model loading, inpainting steps) |
| `INFO` | Status messages (images processed, rows inserted) |
| `WARNING` | Recoverable issues (model fallback, Redis down) |
| `ERROR` | Critical failures (early exit) |

## Testing Configuration

For running `pytest`:

```bash
# Use in-memory DuckDB + file fallback for faces
export SV_DB_PATH=:memory:
export SV_REDIS_URL=redis://localhost:9999/0  # Non-existent; triggers file fallback

pytest tests/ -v
```

All test fixtures respect these environment variables.
