from __future__ import annotations

import os
from pathlib import Path

import cv2
import fakeredis
import numpy as np
import pytest

from sensveridian.config import SETTINGS
from sensveridian.store.faces_registry import FaceRegistry
from sensveridian.store.pg import PgStore

# Tests run inside a dedicated `sv_test` schema of the `sensveridian` database so
# they never touch the live `public` data (the veridian role lacks CREATEDB, so
# a separate schema gives the same isolation without a separate database). The
# schema is migrated once per session and truncated between tests.
TEST_SCHEMA = os.getenv("SENSVERIDIAN_TEST_SCHEMA", "sv_test")
_TABLES = (
    "images, models, runs, predictions_summary, predictions_raw, augmentations, "
    "image_depth_stats, image_bg_plates, datasets, model_versions, layers, "
    "gt_boxes, reviews, clips, segments, ingest_jobs, connections, exports, eval_metrics"
)


@pytest.fixture(scope="session")
def _test_schema_ready() -> str:
    try:
        store = PgStore(SETTINGS.database_url, schema=TEST_SCHEMA)
        store.migrate()
        store.close()
    except Exception as exc:  # pragma: no cover - infra-dependent
        pytest.skip(f"PostgreSQL test schema unavailable: {exc}")
    return TEST_SCHEMA


@pytest.fixture
def pg_store(_test_schema_ready: str):
    store = PgStore(SETTINGS.database_url, schema=TEST_SCHEMA)
    store.con.exec_driver_sql(f"TRUNCATE {_TABLES} RESTART IDENTITY CASCADE")
    # keep the materialized view consistent with the now-empty base tables
    store.con.exec_driver_sql("REFRESH MATERIALIZED VIEW v_image_summary_wide")
    try:
        yield store
    finally:
        store.close()


@pytest.fixture
def file_registry(tmp_path: Path) -> FaceRegistry:
    fallback_file = tmp_path / "faces_registry.json"
    return FaceRegistry(redis_url="redis://127.0.0.1:65001/0", fallback_file=str(fallback_file))


@pytest.fixture
def redis_registry(monkeypatch) -> FaceRegistry:
    import sensveridian.store.faces_registry as fr_mod

    fake = fakeredis.FakeRedis(decode_responses=False)
    monkeypatch.setattr(fr_mod.redis.Redis, "from_url", lambda *_a, **_k: fake)
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
    if not cv2.imwrite(str(path), tiny_image_bgr):
        raise RuntimeError(f"Failed to write fixture image to {path}")
    return path
