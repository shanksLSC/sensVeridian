# Environment Variables

sensVeridian respects the following environment variables for caching, GPU, and service configuration:

## Model & Torch Cache

| Var | Default | Purpose |
|-----|---------|---------|
| `HF_HOME` | `~/.cache/huggingface` | Hugging Face model cache |
| `TORCH_HOME` | `~/.cache/torch` | PyTorch model cache (ZoeDepth, SAM) |
| `XDG_CACHE_HOME` | `~/.cache` | Generic cache (LaMa inpainting) |

**Recommended for `/data3` only operation**:

```bash
export HF_HOME=/data3/ssharma8/hf-cache
export TORCH_HOME=/data3/ssharma8/torch-cache
export XDG_CACHE_HOME=/data3/ssharma8/xdg-cache
```

## GPU

| Var | Default | Purpose |
|-----|---------|---------|
| `CUDA_VISIBLE_DEVICES` | (all) | Comma-separated GPU indices to use (e.g., "1" for GPU-1 only) |
| `PYTORCH_CUDA_ALLOC_CONF` | (default) | PyTorch CUDA memory config; `expandable_segments:True` for flexible allocation |

**Example** (use GPU-1 with flexible allocation):

```bash
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
sv augment distance /path --d-max-ft 10 --step-ft 1 --sam-checkpoint /data3/ssharma8/model-cache/sam/sam_vit_b_01ec64.pth
```

## Redis

| Var | Default | Purpose |
|-----|---------|---------|
| `SV_REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `SV_REDIS_HOST` | `localhost` | Redis host (used by start/stop scripts) |
| `SV_REDIS_PORT` | `6379` | Redis port (used by start/stop scripts) |

**Examples**:

```bash
# Connect to remote Redis
export SV_REDIS_URL=redis://redis.company.com:6379/1

# Custom local Redis (use scripts for management)
export SV_REDIS_HOST=127.0.0.1
export SV_REDIS_PORT=6380
./scripts/redis_start.sh
```

## Database

| Var | Default | Purpose |
|-----|---------|---------|
| `SV_DB_PATH` | `./sensveridian.duckdb` | DuckDB file path |

**Example** (remote or different location):

```bash
export SV_DB_PATH=/mnt/data/sensveridian_prod.duckdb
```

## Logging

sensVeridian uses Python's standard logging; configure with `LOGLEVEL`:

```bash
export LOGLEVEL=DEBUG
sv ingest /path --run-id baseline
```

## Full Example

Recommended setup for `/data3` only operation with GPU-1:

```bash
export HF_HOME=/data3/ssharma8/hf-cache
export TORCH_HOME=/data3/ssharma8/torch-cache
export XDG_CACHE_HOME=/data3/ssharma8/xdg-cache
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SV_REDIS_URL=redis://localhost:6379/0
export SV_DB_PATH=./sensveridian.duckdb

# Now run commands
cd /data3/ssharma8/projects/lattice-internal/sensVeridian
./scripts/redis_start.sh
sv ingest /data3/ssharma8/all-models/Images --run-id baseline
```
