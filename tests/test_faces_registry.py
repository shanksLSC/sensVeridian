from __future__ import annotations

import numpy as np


def _exercise_registry_basic(registry) -> None:
    registry.clear()
    registry.register("person_001", "Alice", np.array([1.0, 0.0, 0.0], dtype=np.float32))
    registry.register("person_002", "Bob", np.array([0.0, 1.0, 0.0], dtype=np.float32))

    entries = registry.list_entries()
    assert [e.person_id for e in entries] == ["person_001", "person_002"]
    assert all(len(e.embedding) == 3 for e in entries)
    # Embeddings are normalized at register-time.
    assert np.isclose(np.linalg.norm(entries[0].embedding), 1.0, atol=1e-4)

    pid, score = registry.match(np.array([1.0, 0.0, 0.0], dtype=np.float32), threshold=0.5)
    assert pid == "person_001"
    assert score > 0.5

    pid2, score2 = registry.match(np.array([-1.0, 0.0, 0.0], dtype=np.float32), threshold=0.99)
    assert pid2 is None
    assert score2 < 0.99

    registry.clear()
    assert registry.list_entries() == []


def test_file_registry_mode_and_behavior(file_registry) -> None:
    assert file_registry.mode == "file"
    _exercise_registry_basic(file_registry)
    file_registry.seed_dummy(n=3, embedding_dim=5, seed=7)
    entries = file_registry.list_entries()
    assert len(entries) == 3
    assert all(len(e.embedding) == 5 for e in entries)


def test_redis_registry_mode_and_behavior(redis_registry) -> None:
    assert redis_registry.mode == "redis"
    _exercise_registry_basic(redis_registry)
    redis_registry.seed_dummy(n=2, embedding_dim=4, seed=7)
    entries = redis_registry.list_entries()
    assert len(entries) == 2
    assert all(len(e.embedding) == 4 for e in entries)

    keys = {k.decode("utf-8") for k in redis_registry.r.keys("*")}
    assert f"{redis_registry.prefix}:index" in keys
    assert f"{redis_registry.prefix}:person_001" in keys
