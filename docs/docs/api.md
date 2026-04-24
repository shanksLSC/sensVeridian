# API Documentation

Auto-generated API reference for sensVeridian modules (generated with Sphinx/pdoc on each build).

For now, consult the Python docstrings directly:

```bash
cd /data3/ssharma8/projects/lattice-internal/sensVeridian
python3 -c "from sensveridian.orchestrator import Orchestrator; help(Orchestrator.ingest)"
```

## Module Overview

| Module | Purpose |
|--------|---------|
| `sensveridian.orchestrator` | High-level ingest orchestration |
| `sensveridian.store.duck` | DuckDB interface |
| `sensveridian.store.faces_registry` | Face registry (Redis/file) |
| `sensveridian.runners.*` | Oracle model runners (AMOD, QRCode, FD, FR) |
| `sensveridian.augmentation.distance_sweep` | Distance-sweep main pipeline |
| `sensveridian.augmentation.depth` | ZoeDepth wrapper |
| `sensveridian.augmentation.segment` | SAM segmentation wrapper |
| `sensveridian.augmentation.inpaint` | LaMa inpainting backend |
| `sensveridian.augmentation.geometry` | Geometric transformations |
| `sensveridian.augmentation.effects` | DoF blur, atmospheric haze |
| `sensveridian.config` | Configuration (paths, thresholds, defaults) |

## Key Classes

### Orchestrator

```python
from sensveridian.orchestrator import Orchestrator

orch = Orchestrator(store, registry)
orch.ingest(
    image_root: str,
    run_id: str = "baseline",
    models: list[str] = ["amod", "qrcode", "fd", "fr"],
    skip_existing: bool = True
)
```

### DuckStore

```python
from sensveridian.store.duck import DuckStore

store = DuckStore(db_path="./sensveridian.duckdb")
store.query_df(sql: str) -> pd.DataFrame
store.export_parquet(path: str, sql: str = None)
```

### FaceRegistry

```python
from sensveridian.store.faces_registry import FaceRegistry

registry = FaceRegistry(redis_url="redis://localhost:6379/0")
registry.register(person_id: str, name: str, embedding: np.ndarray)
registry.match(embedding: np.ndarray, threshold: float = 0.6) -> list[str]
```

### DistanceAugmentor

```python
from sensveridian.augmentation.distance_sweep import DistanceAugmentor

augmentor = DistanceAugmentor()
augmented_paths = augmentor.augment_image(
    parent_image_path: str,
    detections: list[dict],
    d_max_ft: float,
    step_ft: float,
    overrides: DistanceOverrides = None
) -> list[tuple[float, str]]
```

## Full API Reference

For detailed API docs, generate with Sphinx:

```bash
cd /data3/ssharma8/projects/lattice-internal/sensVeridian
pip install sphinx sphinx-rtd-theme
sphinx-build -b html docs/source docs/build
open docs/build/index.html
```

Or use interactive help in Python:

```python
import sensveridian.orchestrator as orch
help(orch)
```
