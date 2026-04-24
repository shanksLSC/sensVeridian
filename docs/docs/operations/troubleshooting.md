# Troubleshooting

## Model & Data Issues

### "ValueError: Unknown layer: sensAI>QuantizeConv2D"

The oracle `.h5` models use custom Keras layers from the sensAI SDK. sensVeridian dynamically loads these at runtime.

**Fix**: Ensure sensai SDK modules are available. If models load successfully at least once, they're cached in `HF_HOME`.

### "Input shape mismatch" for QRCode model

QRCode model expects grayscale (1 channel), but preprocessing sometimes passes RGB (3 channels).

**Fix**: This is handled automatically by `preprocess_for_model`. If it persists, check the model's `input_spec` in the runner.

### "No module named 'redis'"

Redis client not installed.

**Fix**: 
```bash
/data3/ssharma8/py310/bin/pip install redis
```

### Images not found or skipped

By default, `sv ingest` skips images already in the database.

**Fix**: Either:
- Use `--skip-existing False` to force reprocess
- Check the database with `sv query "SELECT COUNT(*) FROM images"`

## Redis Issues

### "redis.exceptions.ConnectionError: Error 111 connecting to localhost:6379"

Redis server not running.

**Fix**: Start it:
```bash
./scripts/redis_start.sh
./scripts/redis_status.sh  # Verify
```

### Redis fell back to file mode (`cache/faces_registry.json`)

Redis unavailable; using local JSON fallback. This is intentional, but data is not persistent across instances.

**Fix**: Start Redis and promote entries:
```bash
./scripts/redis_start.sh
/data3/ssharma8/py310/bin/python scripts/migrate_faces_to_redis.py
```

### "maxmemory exceeded"

Redis hit 256 MB limit (configured in `cache/redis/redis.conf`).

**Fix**: Faces must not be dropped due to `noeviction` policy. Either:
- Reduce registered faces with `sv faces clear`
- Increase `maxmemory` in `cache/redis/redis.conf` and reload

## GPU / CUDA Issues

### "CUDA out of memory"

Distance augmentation uses significant GPU memory (ZoeDepth, SAM, LaMa).

**Fix**:
1. Use GPU with more memory: `export CUDA_VISIBLE_DEVICES=0` (or check `nvidia-smi`)
2. Skip ZoeDepth if possible: `sv augment distance --d0-ft 5.0` (manual distance)
3. Reduce batch size or distance steps: `--step-ft 2` instead of `1`
4. Use a smaller SAM checkpoint (if available)

### "CUDA device not found"

GPU not visible to PyTorch.

**Fix**:
```bash
# Check available GPUs
nvidia-smi

# Explicitly select GPU
export CUDA_VISIBLE_DEVICES=0
sv ingest /path  # Now uses GPU-0
```

### ZoeDepth loads but runs very slow

ZoeDepth is not on GPU; running on CPU instead.

**Fix**: Check `export CUDA_VISIBLE_DEVICES` and that CUDA is properly installed. If on correct GPU and still slow, it's likely a library issue; try:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Augmentation Issues

### "FileNotFoundError: SAM checkpoint not found"

SAM model checkpoint not downloaded.

**Fix**:
```bash
mkdir -p /data3/ssharma8/model-cache/sam
wget -O /data3/ssharma8/model-cache/sam/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

Then re-run:
```bash
sv augment distance /path --sam-checkpoint /data3/ssharma8/model-cache/sam/sam_vit_b_01ec64.pth ...
```

### "RuntimeError: Error(s) in loading state_dict for ZoeDepthNK"

ZoeDepth model version mismatch (usually `timm` version).

**Fix**: Ensure `timm==0.6.12` is pinned in `pyproject.toml`:
```bash
/data3/ssharma8/py310/bin/pip install timm==0.6.12
```

### Augmented images look wrong (scaled objects misaligned)

Depth estimation inaccurate, or manual distance override too far from reality.

**Fix**:
1. Inspect depth map: `cache/depth_maps/{image_id}_depth.npy`
2. Check manual overrides: `--d0-map` JSON is correct
3. Re-run without manual override to use ZoeDepth

### "HTTPStatusError: 401 Unauthorized" from Hugging Face

Authentication required for model downloads (unlikely for public models like SAM, ZoeDepth).

**Fix**: Set up Hugging Face credentials:
```bash
huggingface-cli login
# or set HF_TOKEN environment variable
```

## Database Issues

### "sensveridian.duckdb" corrupted or locked

Database file corrupted or in use.

**Fix**:
```bash
# Backup and reset
cp sensveridian.duckdb sensveridian.duckdb.bak
rm sensveridian.duckdb  # Will recreate on next ingest
sv query "SELECT COUNT(*) FROM images"  # Recreates schema
```

### Query runs very slow

Large dataset with expensive grouping/joins.

**Fix**:
1. Add indexes: `CREATE INDEX idx_model_id ON predictions_summary(model_id)`
2. Export to Parquet and use pandas for analytics
3. Filter by run_id or model_id to reduce rows

### "DuckDB memory error" on export

Full dataset too large for in-memory parquet.

**Fix**: Export in chunks with filtered SQL:
```bash
sv export --to /tmp/part1.parquet \
  --sql "SELECT * FROM predictions_summary WHERE run_id='baseline'"
```

## General Troubleshooting

### "Permission denied" on scripts

Scripts not executable.

**Fix**:
```bash
chmod +x scripts/redis_*.sh
```

### "ModuleNotFoundError: sensveridian" when running tests

Python path not set.

**Fix**:
```bash
cd /data3/ssharma8/projects/lattice-internal/sensVeridian
export PYTHONPATH=src:$PYTHONPATH
pytest tests/
```

### Tests pass locally but fail in CI

Environment differences (GPU, cache, paths).

**Fix**: Check `.github/workflows/docs.yml` and CI logs. Ensure all paths are absolute or relative to project root.

## Report an Issue

If you find a bug not listed here:

1. Check the [GitHub Issues](https://github.com/shanksLSC/sensVeridian/issues)
2. Provide:
   - Command that failed
   - Full error traceback
   - Output of `sv stats`
   - Environment (OS, Python version, GPU, etc.)

Then open an issue with details.
