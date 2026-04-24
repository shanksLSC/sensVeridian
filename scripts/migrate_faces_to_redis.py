"""Migrate registered faces from the file fallback into Redis.

The ``FaceRegistry`` falls back to a JSON file when Redis is unreachable.
Once a real Redis is running, this script re-registers every entry from the
fallback into Redis and archives the JSON so subsequent runs never
accidentally fall back again with stale data.

Usage:
    /data3/ssharma8/py310/bin/python \
        /data3/ssharma8/projects/lattice-internal/sensVeridian/scripts/migrate_faces_to_redis.py

Dummy Check-in for testing purposes.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from sensveridian.config import SETTINGS  # noqa: E402
from sensveridian.store.faces_registry import FaceRegistry  # noqa: E402


def main() -> int:
    fallback = PROJECT_ROOT / "cache" / "faces_registry.json"
    if not fallback.exists():
        print(f"no fallback file at {fallback}; nothing to migrate")
        return 0

    try:
        raw = json.loads(fallback.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"could not parse fallback file {fallback}: {exc}", file=sys.stderr)
        return 1

    if not raw:
        print("fallback file is empty; nothing to migrate")
        return 0

    redis_url = os.environ.get("SV_REDIS_URL", SETTINGS.redis_url)
    registry = FaceRegistry(redis_url=redis_url)
    if registry.mode != "redis":
        print(
            f"Redis is not reachable at {redis_url}; start it with "
            f"scripts/redis_start.sh before migrating",
            file=sys.stderr,
        )
        return 2

    migrated = 0
    for person_id, payload in sorted(raw.items()):
        name = str(payload.get("name", person_id))
        emb = np.asarray(payload.get("embedding", []), dtype=np.float32)
        if emb.size == 0:
            print(f"  skip {person_id}: empty embedding")
            continue
        registry.register(person_id=person_id, name=name, embedding=emb)
        migrated += 1
        print(f"  migrated {person_id} ({name}, dim={emb.size})")

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    archive = fallback.with_suffix(f".json.migrated-{ts}")
    shutil.move(str(fallback), str(archive))
    print(f"migrated {migrated} entries into Redis; archived fallback to {archive}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
