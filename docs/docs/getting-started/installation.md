# Installation

Assumes Python 3.10 from `/data3/ssharma8/py310`.

## Setup

```bash
cd /data3/ssharma8/projects/lattice-internal/sensVeridian

# Use existing interpreter directly
/data3/ssharma8/py310/bin/python -m pip install -e .

# Or set up a project venv
/data3/ssharma8/py310/bin/python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## External Assets

### SAM Checkpoint (~375 MB, one-time download)

```bash
mkdir -p /data3/ssharma8/model-cache/sam
wget -O /data3/ssharma8/model-cache/sam/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### ZoeDepth & LaMa

These are automatically downloaded on first run into `TORCH_HOME`.

## Dependencies

The project requires:

- **Python 3.10+**
- **TensorFlow 2.12+** — Oracle model inference
- **PyTorch 2.2+** — ZoeDepth, SAM, LaMa
- **DuckDB** — Queryable ground-truth cache
- **Redis** — Face registry (per-project server included)
- **OpenCV** — Image I/O and transforms
- **NumPy, Pandas** — Data manipulation

Full dependency list in `pyproject.toml`.
