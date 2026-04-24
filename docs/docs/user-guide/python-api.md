# Python API

Programmatic access to sensVeridian via Python modules.

## Quick Start

```python
from sensveridian.orchestrator import Orchestrator
from sensveridian.store.duck import DuckStore
from sensveridian.store.faces_registry import FaceRegistry

# Initialize
store = DuckStore()
registry = FaceRegistry()
orch = Orchestrator(store, registry)

# Ingest images
orch.ingest(image_root="/data3/ssharma8/all-models/Images", run_id="baseline")

# Query
results_df = store.query_df("SELECT * FROM v_image_summary_wide")
print(results_df)

# Export
store.export_parquet("/tmp/results.parquet")
```

## Core Classes

### DuckStore

Database abstraction layer for DuckDB.

```python
from sensveridian.store.duck import DuckStore

store = DuckStore(db_path="./sensveridian.duckdb")

# Query
df = store.query_df("SELECT * FROM predictions_summary WHERE model_id='amod'")

# Insert/upsert
store.upsert_image(image_id="abc123", path="/img.jpg", width=640, height=480, file_size_bytes=150000)
store.upsert_prediction_summary(
    image_id="abc123",
    model_id="amod",
    run_id="baseline",
    present=True,
    count=3,
    extras={"class_distribution": {"car": 2, "person": 1}}
)

# Export
store.export_parquet("/tmp/results.parquet", sql="SELECT * FROM images")
```

### FaceRegistry

Manages registered face embeddings (Redis or file fallback).

```python
from sensveridian.store.faces_registry import FaceRegistry
import numpy as np

registry = FaceRegistry(redis_url="redis://localhost:6379/0")

# Register a face
person_id = "john_doe_001"
embedding = np.random.randn(128).astype(np.float32)  # 128-dim vector
registry.register(person_id, "John Doe", embedding)

# List registered
entries = registry.list_entries()
print(entries)  # [{"person_id": "john_doe_001", "name": "John Doe", ...}, ...]

# Match against new embedding
test_embedding = np.random.randn(128).astype(np.float32)
matches = registry.match(test_embedding, threshold=0.6)
print(matches)  # List of matched person_ids with distances

# Clear all
registry.clear()
```

### Orchestrator

High-level orchestration of the ingest pipeline.

```python
from sensveridian.orchestrator import Orchestrator
from sensveridian.store.duck import DuckStore
from sensveridian.store.faces_registry import FaceRegistry

store = DuckStore()
registry = FaceRegistry()
orch = Orchestrator(store, registry)

# Ingest with all four models
orch.ingest(
    image_root="/data3/ssharma8/all-models/Images",
    run_id="baseline",
    models=["amod", "qrcode", "fd", "fr"]
)

# Ingest with specific models
orch.ingest(
    image_root="/data3/ssharma8/all-models/Images/qr_code",
    run_id="qr_only",
    models=["qrcode"],
    skip_existing=False  # Force reprocess
)
```

## Model Runners

Direct access to oracle models.

```python
from sensveridian.runners.amod import AMODRunner
from sensveridian.runners.qrcode import QRCodeRunner
from sensveridian.runners.face_detection import FaceDetectionRunner
from sensveridian.runners.face_recognition import FaceRecognitionRunner
import cv2

# Load runner
runner = AMODRunner()
model = runner.load()

# Predict
image = cv2.imread("/path/to/image.jpg")
result = runner.predict(image, {"model": model})

print(result["detections"])  # List of {bbox, class_id, confidence, ...}
print(result["summary"])     # {present: bool, count: int}
```

## Distance Augmentation

```python
from sensveridian.augmentation.distance_sweep import DistanceAugmentor
from sensveridian.augmentation.manual_distance import DistanceOverrides
import json

# Initialize with components
augmentor = DistanceAugmentor(
    segmenter=None,  # Auto-creates SAMSegmenter
    depth_estimator=None,  # Auto-creates ZoeDepthEstimator
    inpainter=None,  # Auto-creates LaMAInpainter
)

# Optional: manual distance overrides
with open("d0_overrides.json") as f:
    overrides_dict = json.load(f)
overrides = DistanceOverrides(**overrides_dict)

# Augment
parent_image_path = "/path/to/image.jpg"
detections = [
    {"bbox": [100, 100, 200, 200], "class_id": 1, ...},
    {"bbox": [300, 300, 400, 400], "class_id": 2, ...},
]

augmented_images = augmentor.augment_image(
    parent_image_path,
    detections,
    d_max_ft=10.0,
    step_ft=1.0,
    overrides=overrides,
)

for dist_ft, aug_img_path in augmented_images:
    print(f"Distance {dist_ft} ft: {aug_img_path}")
```

## Utility Modules

### Hash

```python
from sensveridian.hashing import hash_file, hash_decoded_image
import cv2

# Hash a file
image_id = hash_file("/path/to/image.jpg")

# Hash a decoded image
image = cv2.imread("/path/to/image.jpg")
image_id, width, height = hash_decoded_image(image)
```

### Config

```python
from sensveridian.config import SETTINGS, ModelPaths

print(SETTINGS.db_path)      # ./sensveridian.duckdb
print(SETTINGS.redis_url)    # redis://localhost:6379/0
print(ModelPaths.amod)       # /data3/.../amod.h5
```

## Full Example: Custom Pipeline

```python
import os
from pathlib import Path
from sensveridian.orchestrator import Orchestrator
from sensveridian.store.duck import DuckStore
from sensveridian.store.faces_registry import FaceRegistry
from sensveridian.augmentation.distance_sweep import DistanceAugmentor
from sensveridian.augmentation.manual_distance import DistanceOverrides

# Setup
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
store = DuckStore()
registry = FaceRegistry()
orch = Orchestrator(store, registry)

# Ingest baseline
orch.ingest(
    image_root="/data3/ssharma8/all-models/Images",
    run_id="baseline",
    skip_existing=False
)

# Query results
df = store.query_df("SELECT * FROM v_image_summary_wide")
print(df)

# Augment with manual distances
overrides = DistanceOverrides(
    global_ft=5.0,
    images={}
)
augmentor = DistanceAugmentor()

for image_path in Path("/data3/ssharma8/all-models/Images").glob("**/*.jpg"):
    # Get detections from database
    detections_raw = store.query_df(
        f"SELECT payload FROM predictions_raw WHERE model_id='amod' AND image_path='{image_path}'"
    )
    if len(detections_raw) == 0:
        continue
    
    payload = detections_raw.iloc[0]["payload"]
    detections = payload.get("detections", [])
    
    # Augment
    augmented = augmentor.augment_image(
        str(image_path),
        detections,
        d_max_ft=10.0,
        step_ft=1.0,
        overrides=overrides,
    )
    print(f"Generated {len(augmented)} augmentations for {image_path.name}")

# Export for analytics
store.export_parquet("/tmp/ground_truth.parquet")
print("Exported to /tmp/ground_truth.parquet")
```
