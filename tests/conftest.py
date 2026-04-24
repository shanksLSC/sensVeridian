from __future__ import annotations

from pathlib import Path

import cv2
import fakeredis
import numpy as np
import pytest

from sensveridian.store.duck import DuckStore
from sensveridian.store.faces_registry import FaceRegistry


@pytest.fixture
def schema_path() -> Path:
    return Path(__file__).resolve().parents[1] / "src" / "sensveridian" / "store" / "schema.sql"


@pytest.fixture
def duck_store(tmp_path: Path, schema_path: Path):
    db_path = tmp_path / "test.duckdb"
    store = DuckStore(db_path=db_path, schema_path=schema_path)
    store.migrate()
    try:
        yield store
    finally:
        store.close()


@pytest.fixture
def file_registry(tmp_path: Path) -> FaceRegistry:
    fallback_file = tmp_path / "faces_registry.json"
    reg = FaceRegistry(
        redis_url="redis://127.0.0.1:65001/0",
        fallback_file=str(fallback_file),
    )
    return reg


@pytest.fixture
def redis_registry(monkeypatch) -> FaceRegistry:
    import sensveridian.store.faces_registry as fr_mod

    fake = fakeredis.FakeRedis(decode_responses=False)

    def _fake_from_url(*_args, **_kwargs):
        return fake

    monkeypatch.setattr(fr_mod.redis.Redis, "from_url", _fake_from_url)
    return FaceRegistry(redis_url="redis://localhost:6379/0")


@pytest.fixture
def tiny_image_bgr() -> np.ndarray:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[8:24, 8:24] = (255, 255, 255)
    img[30:48, 30:60] = (0, 255, 0)
    return img


@pytest.fixture
def tiny_image_file(tmp_path: Path, tiny_image_bgr: np.ndarray) -> Path:
    path = tmp_path / "tiny.png"
    ok = cv2.imwrite(str(path), tiny_image_bgr)
    if not ok:
        raise RuntimeError(f"Failed to write fixture image to {path}")
    return path
