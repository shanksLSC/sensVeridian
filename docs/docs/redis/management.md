# Server Management

Quick reference for Redis server operations specific to this project.

## Start

```bash
./scripts/redis_start.sh
```

- Checks if Redis already running (returns 0 if yes)
- Starts `/data3/ssharma8/py310/bin/redis-server` with config `cache/redis/redis.conf`
- Waits for server to respond
- Writes PID to `cache/redis/redis.pid`

## Status

```bash
./scripts/redis_status.sh
```

- Checks if process is running
- Pings the server
- Shows version, PID, uptime, key count

**Exit codes**: 0 = running, 1 = not running, 2 = error

## Stop

```bash
./scripts/redis_stop.sh
```

- Sends `SHUTDOWN NOSAVE` via redis-cli (graceful)
- Falls back to `SIGTERM` if no response
- Falls back to `SIGKILL` if still running after 5 sec

## Custom Port/Host

Override the default `127.0.0.1:6379`:

```bash
export SV_REDIS_HOST=127.0.0.1
export SV_REDIS_PORT=6380
./scripts/redis_start.sh
```

The scripts will use the specified host/port.

## Manual Commands

Use `redis-cli` directly (from conda env):

```bash
/data3/ssharma8/py310/bin/redis-cli -h 127.0.0.1 -p 6379

# Inside redis-cli:
> PING                    # -> PONG
> KEYS face:*             # List all face keys
> SMEMBERS face:registered:index  # List all registered person IDs
> INFO server             # Server info
> DBSIZE                  # Total keys
> SHUTDOWN NOSAVE         # Stop cleanly
```

## Verify FaceRegistry

Check that the Python side is connected to Redis (not file fallback):

```bash
cd /data3/ssharma8/projects/lattice-internal/sensVeridian
export PYTHONPATH=src:$PYTHONPATH
python3 -c "
from sensveridian.store.faces_registry import FaceRegistry
from sensveridian.config import SETTINGS
r = FaceRegistry(redis_url=SETTINGS.redis_url)
print(f'mode: {r.mode}, entries: {len(r.list_entries())}')
"
```

Expected output: `mode: redis, entries: N`

If it says `mode: file`, Redis is not reachable.

## Monitoring

Watch Redis in real time:

```bash
# CPU/memory usage
watch -n 1 'redis-cli INFO server | grep -E "used_memory|total_system_memory"'

# Key operations
redis-cli MONITOR
```

## Backup & Recovery

Persist faces to JSON (portable):

```bash
python3 -c "
import json
from sensveridian.store.faces_registry import FaceRegistry
r = FaceRegistry()
entries = r.list_entries()
with open('faces_backup.json', 'w') as f:
    json.dump({e['person_id']: e for e in entries}, f, default=str)
"
```

Restore from JSON:

```bash
python3 -c "
import json
from sensveridian.store.faces_registry import FaceRegistry
r = FaceRegistry()
with open('faces_backup.json', 'r') as f:
    data = json.load(f)
    for entry in data.values():
        r.register(entry['person_id'], entry.get('name', ''), entry['embedding'])
"
```
