from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import redis


@dataclass
class FaceEntry:
    person_id: str
    name: str
    embedding: np.ndarray


class FaceRegistry:
    def __init__(self, redis_url: str, key_prefix: str = "face:registered", fallback_file: str | None = None):
        self.prefix = key_prefix
        self.index_key = f"{self.prefix}:index"
        self.mode = "redis"
        self.fallback_file = Path(
            fallback_file
            or "/data3/ssharma8/projects/lattice-internal/sensVeridian/cache/faces_registry.json"
        )
        self.r = redis.Redis.from_url(redis_url, socket_connect_timeout=0.3, socket_timeout=0.3)
        try:
            self.r.ping()
        except Exception:
            self.mode = "file"
            self.fallback_file.parent.mkdir(parents=True, exist_ok=True)
            if not self.fallback_file.exists():
                self.fallback_file.write_text("{}", encoding="utf-8")

    def _person_key(self, person_id: str) -> str:
        return f"{self.prefix}:{person_id}"

    def _read_file_store(self) -> dict:
        if not self.fallback_file.exists():
            return {}
        try:
            return json.loads(self.fallback_file.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_file_store(self, payload: dict) -> None:
        self.fallback_file.write_text(json.dumps(payload), encoding="utf-8")

    def register(self, person_id: str, name: str, embedding: np.ndarray) -> None:
        emb = np.asarray(embedding, dtype=np.float32).flatten()
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        if self.mode == "redis":
            self.r.hset(
                self._person_key(person_id),
                mapping={
                    "name": name,
                    "embedding": emb.tobytes(),
                },
            )
            self.r.sadd(self.index_key, person_id)
            return
        store = self._read_file_store()
        store[person_id] = {"name": name, "embedding": emb.tolist()}
        self._write_file_store(store)

    def clear(self) -> None:
        if self.mode == "file":
            self._write_file_store({})
            return
        ids = self.r.smembers(self.index_key)
        pipe = self.r.pipeline()
        for pid in ids:
            pipe.delete(self._person_key(pid.decode("utf-8")))
        pipe.delete(self.index_key)
        pipe.execute()

    def list_entries(self) -> list[FaceEntry]:
        if self.mode == "file":
            out = []
            for person_id, payload in sorted(self._read_file_store().items()):
                out.append(
                    FaceEntry(
                        person_id=person_id,
                        name=str(payload.get("name", "")),
                        embedding=np.asarray(payload.get("embedding", []), dtype=np.float32),
                    )
                )
            return out
        entries: list[FaceEntry] = []
        for bpid in sorted(self.r.smembers(self.index_key)):
            person_id = bpid.decode("utf-8")
            data = self.r.hgetall(self._person_key(person_id))
            if not data:
                continue
            name = data.get(b"name", b"").decode("utf-8")
            emb = np.frombuffer(data[b"embedding"], dtype=np.float32)
            entries.append(FaceEntry(person_id=person_id, name=name, embedding=emb))
        return entries

    def match(self, embedding: np.ndarray, threshold: float) -> tuple[str | None, float]:
        q = np.asarray(embedding, dtype=np.float32).flatten()
        q = q / (np.linalg.norm(q) + 1e-8)
        best_pid = None
        best_score = -1.0
        for entry in self.list_entries():
            score = float(np.dot(q, entry.embedding))
            if score > best_score:
                best_score = score
                best_pid = entry.person_id
        if best_score < threshold:
            return None, best_score
        return best_pid, best_score

    def seed_dummy(self, n: int, embedding_dim: int, seed: int = 7) -> None:
        rng = np.random.default_rng(seed)
        for i in range(n):
            vec = rng.normal(size=(embedding_dim,)).astype(np.float32)
            self.register(f"person_{i+1:03d}", f"Dummy Person {i+1}", vec)

