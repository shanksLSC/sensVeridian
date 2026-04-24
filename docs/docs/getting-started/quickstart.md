# Quick Start

## One-time setup

```bash
# Start the per-project Redis server
./scripts/redis_start.sh

# Seed 8 dummy registered faces
sv faces seed --n 8 --clear-first
```

## Ingest images through all oracle models

```bash
sv ingest /data3/ssharma8/all-models/Images --run-id baseline
```

## See what's in the cache

```bash
# Summary view (one row per image, all models pivoted)
sv query "SELECT * FROM v_image_summary_wide LIMIT 5"

# Full row counts
sv stats
```

## Generate distance-swept augmentations (GPU recommended)

```bash
sv augment distance /data3/ssharma8/all-models/Images \
  --d-max-ft 10 --step-ft 1 \
  --sam-checkpoint /data3/ssharma8/model-cache/sam/sam_vit_b_01ec64.pth \
  --auto-run-oracle --run-id distance_eval
```

## Export ground truth as parquet

```bash
sv export --to /tmp/ground_truth.parquet
```

## Stop Redis when done

```bash
./scripts/redis_stop.sh
```
