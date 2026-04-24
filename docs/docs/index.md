# sensVeridian

**Ground-truth cache and distance-aware augmentation pipeline for vision ML model evaluation.**

sensVeridian ingests images, runs a configurable pipeline of ML oracle models on each, stores results in a queryable DuckDB cache, and can synthesize photorealistic distance-swept augmentations for robustness testing.

## Features

- **Oracle Models**: Run AMOD, QRCode Detection, Face Detection, Face Recognition on images
- **Queryable Cache**: DuckDB-backed storage with SQL queries and Parquet export
- **Face Registry**: Redis-backed registered-faces lookup for FR matching
- **Distance Augmentation**: Photorealistic distance-swept augmentations using ZoeDepth + SAM + LaMa
- **Manual Overrides**: Specify known distances per-image or per-detection, ZoeDepth as fallback
- **CLI & Python API**: Full command-line and programmatic interfaces

## Quick Start

```bash
# Install
pip install -e .

# Start Redis (per-project instance)
./scripts/redis_start.sh

# Ingest images
sv ingest /path/to/images --run-id baseline

# Query results
sv query "SELECT * FROM v_image_summary_wide"

# Generate augmented images
sv augment distance /path/to/images --d-max-ft 10 --step-ft 1 \
  --sam-checkpoint /data3/ssharma8/model-cache/sam/sam_vit_b_01ec64.pth
```

## Documentation

- **[Installation](getting-started/installation.md)** — Setup and dependencies
- **[CLI Reference](user-guide/cli.md)** — All commands with examples
- **[Data Model](user-guide/data-model.md)** — Schema and tables
- **[Python API](user-guide/python-api.md)** — Programmatic usage
- **[Troubleshooting](operations/troubleshooting.md)** — Common issues and fixes

## GitHub

- **Repository**: [shanksLSC/sensVeridian](https://github.com/shanksLSC/sensVeridian)
- **Issues**: Report bugs or request features [here](https://github.com/shanksLSC/sensVeridian/issues)
