# Operational Tips

Best practices for running sensVeridian in production and development.

## Pre-ingest Checklist

- [ ] Redis running: `./scripts/redis_status.sh`
- [ ] Face registry seeded: `sv faces list` shows ≥1 entry
- [ ] Models accessible: `ls /data3/ssharma8/all-models/*/..h5`
- [ ] SAM checkpoint available: `ls /data3/ssharma8/model-cache/sam/sam_vit_b_01ec64.pth`
- [ ] Disk space: `df -h` (DuckDB, augmentations, depth maps need 10–100 GB depending on image count)

## Ingest Optimization

### Fast path (summary only, no augmentation)

```bash
# Ingest all images quickly
time sv ingest /data3/ssharma8/all-models/Images --run-id baseline

# Query results
sv query "SELECT COUNT(*) FROM v_image_summary_wide"
```

Expected: ~5–10 min for 10 images on GPU (model loading overhead dominates for small sets).

### Parallel ingests (different model subsets)

Run multiple ingest commands in parallel to speed up if you have many images:

```bash
# Terminal 1
sv ingest /data3/ssharma8/all-models/Images --models amod,qrcode --run-id baseline_part1 &

# Terminal 2
sv ingest /data3/ssharma8/all-models/Images --models fd,fr --run-id baseline_part2 &

wait
```

Then merge results by querying with `run_id IN (...)`.

## Augmentation Optimization

### GPU memory management

If running out of memory during distance sweep:

1. **Skip ZoeDepth**: `--d0-ft 5.0` saves ~2 min/image and 4GB VRAM
2. **Reduce steps**: `--step-ft 2` instead of `1` (fewer augmentations)
3. **Use multiple GPUs**: Split images across GPUs

```bash
# Split dataset
ls /data3/ssharma8/all-models/Images | head -5 | xargs -I {} cp {} /tmp/batch1/
ls /data3/ssharma8/all-models/Images | tail -n +6 | head -5 | xargs -I {} cp {} /tmp/batch2/

# Run in parallel
export CUDA_VISIBLE_DEVICES=0
sv augment distance /tmp/batch1 --d-max-ft 10 --step-ft 1 --d0-ft 5.0 \
  --sam-checkpoint /data3/ssharma8/model-cache/sam/sam_vit_b_01ec64.pth &

export CUDA_VISIBLE_DEVICES=1
sv augment distance /tmp/batch2 --d-max-ft 10 --step-ft 1 --d0-ft 5.0 \
  --sam-checkpoint /data3/ssharma8/model-cache/sam/sam_vit_b_01ec64.pth &

wait
```

### Caching inpainted plates

LaMa inpainting is slow (~10 sec). sensVeridian caches background plates per image and inpainter:

```bash
# First run: slow (generates plate)
sv augment distance img1.jpg --d-max-ft 10 --step-ft 1 \
  --sam-checkpoint /data3/ssharma8/model-cache/sam/sam_vit_b_01ec64.pth
# Slow, creates cache/bg_plates/img1_lama.png

# Second run on same image: fast (reuses plate)
sv augment distance img1.jpg --d-max-ft 15 --step-ft 1 \
  --sam-checkpoint /data3/ssharma8/model-cache/sam/sam_vit_b_01ec64.pth
# Faster, reuses cache/bg_plates/img1_lama.png
```

## Database Maintenance

### Regular backups

```bash
# Backup before large operations
cp sensveridian.duckdb sensveridian.duckdb.backup.$(date +%s)

# Keep last 3 backups
ls -1t sensveridian.duckdb.backup.* | tail -n +4 | xargs rm
```

### Cleanup old runs

```bash
# Remove old runs to free space
sv query "DELETE FROM predictions_summary WHERE run_id='old_experiment'"
sv query "DELETE FROM predictions_raw WHERE run_id='old_experiment'"
```

### Index for large tables

After ingesting 100s of images, add indexes for common queries:

```bash
sv query "CREATE INDEX IF NOT EXISTS idx_image_model ON predictions_summary(image_id, model_id)"
sv query "CREATE INDEX IF NOT EXISTS idx_run ON predictions_summary(run_id)"
```

## Monitor Resources

Real-time monitoring during augmentation:

```bash
# Terminal 1: Watch GPU
watch -n 2 nvidia-smi

# Terminal 2: Watch disk
watch -n 5 'du -sh cache/ .'

# Terminal 3: Watch Redis
watch -n 2 '/data3/ssharma8/py310/bin/redis-cli INFO server'
```

## Reproducibility

Fix random seeds for deterministic augmentations:

```bash
# Environment
export PYTHONHASHSEED=0

# In Python code
import random, numpy as np, torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

sv augment distance /path --d-max-ft 10 --step-ft 1 \
  --sam-checkpoint /data3/ssharma8/model-cache/sam/sam_vit_b_01ec64.pth
```

Result: Same input images produce identical augmentations on re-run.

## Dataset Export for ML Training

Export augmented images + ground truth as a training dataset:

```bash
# 1. Export metadata to parquet
sv export --to /tmp/training_set.parquet \
  --sql "SELECT * FROM v_image_summary_wide WHERE run_id='distance_eval'"

# 2. Copy augmented images
cp -r cache/augmentations /mnt/training_dataset/

# 3. Load in training pipeline
import pandas as pd
df = pd.read_parquet('/tmp/training_set.parquet')
for idx, row in df.iterrows():
    image_path = f"/mnt/training_dataset/{row['image_id']}_*.png"
    labels = {...}  # Construct from row
```

## Troubleshooting Performance

### Ingest is slow

- **Check GPU**: `nvidia-smi` (should show high utilization)
- **Check model path**: Verify `.h5` files are local, not NFS
- **Check Redis**: `./scripts/redis_status.sh` (should respond quickly)

### Augmentation stalls

- **Check ZoeDepth**: If `--d0-ft` not set, it's loading (~2 min)
- **Check SAM**: If many objects, SAM can take 5+ sec per image
- **Check LaMa**: First image is slowest (model loading); subsequent faster

### Queries are slow

- **Check table size**: `sv query "SELECT COUNT(*) FROM predictions_raw"`
- **Add indexes**: See "Index for large tables" above
- **Export to Parquet**: For repeated analysis, use pandas

## Shutdown Cleanly

```bash
# Stop any running ingest/augment jobs (Ctrl+C)

# Stop Redis gracefully
./scripts/redis_stop.sh

# Backup database
cp sensveridian.duckdb sensveridian.duckdb.final.$(date +%s)

echo "sensVeridian stopped cleanly"
```
