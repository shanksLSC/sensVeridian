# Redis Setup

The registered-faces DB is a Redis hash + set index. The project ships with a self-contained, per-project Redis that lives entirely under `/data3/ssharma8` — no system package, no `sudo`, no writes to `/home/ssharma8`.

## Binary

`/data3/ssharma8/py310/bin/redis-server` (installed via `conda install -p /data3/ssharma8/py310 -c conda-forge --override-channels -y redis-server`).

## Configuration

**File**: `cache/redis/redis.conf`

- Binds `127.0.0.1:6379` only (protected-mode on, no network exposure)
- Data dir: `cache/redis/data/` (RDB snapshots + AOF for durability)
- Log: `cache/redis/redis.log`, PID: `cache/redis/redis.pid`
- `maxmemory 256mb` with `noeviction` (faces must never be silently dropped)

## Start / Stop / Status

```bash
cd /data3/ssharma8/projects/lattice-internal/sensVeridian

# Start (idempotent; returns 0 if already running, including external instance)
./scripts/redis_start.sh

# Status (exit 0 if running)
./scripts/redis_status.sh

# Stop (clean SHUTDOWN via redis-cli, SIGTERM/SIGKILL fallback)
./scripts/redis_stop.sh
```

### Override Host/Port

```bash
SV_REDIS_HOST=127.0.0.1 SV_REDIS_PORT=6380 ./scripts/redis_start.sh
export SV_REDIS_URL=redis://127.0.0.1:6380/0
```

## One-time Migration

If you used sensVeridian before Redis was running, dummy entries were persisted to `cache/faces_registry.json`. After starting Redis, promote them:

```bash
/data3/ssharma8/py310/bin/python scripts/migrate_faces_to_redis.py
```

The script re-registers each entry in Redis and archives the JSON as `cache/faces_registry.json.migrated-<timestamp>` so the file fallback can't silently shadow the live Redis data on later runs.

## Verify Redis

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

## Key Schema

| Key | Type | Fields / members |
|-----|------|------------------|
| `face:registered:index` | set | all registered `person_id`s |
| `face:registered:<person_id>` | hash | `name` (utf-8), `embedding` (raw float32 bytes) |

## File Fallback (Still Supported)

If Redis is unreachable at `SV_REDIS_URL`, `FaceRegistry` transparently falls back to `cache/faces_registry.json` so local development still works. Once Redis is up, run the migration script above to promote those entries and archive the JSON.
