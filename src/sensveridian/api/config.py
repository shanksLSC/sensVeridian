"""API settings — reads env (.env) and reuses ``sensveridian.config`` for the
oracle model paths / device / face-match threshold so there is a single source
of truth. Caches and media stay under /data3 per repo convention.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from ..config import MODELS, SETTINGS


def _origins() -> list[str]:
    raw = os.getenv("VERIDIAN_CORS", "http://localhost:5173,http://localhost:8000")
    return [o.strip() for o in raw.split(",") if o.strip()]


@dataclass
class Settings:
    # PostgreSQL (DuckDB -> Postgres migration target). Async URL for the API;
    # the worker derives the sync URL via sensveridian.store.pg.sync_url().
    database_url: str = os.getenv(
        "DATABASE_URL", "postgresql+asyncpg://veridian:veridian@localhost:5432/sensveridian"
    )
    # Redis backs both the face registry and the arq ingest queue.
    redis_url: str = os.getenv("SV_REDIS_URL", SETTINGS.redis_url)
    cors_origins: list[str] = field(default_factory=_origins)

    # media + frame storage (keep off /home, per repo convention)
    media_root: str = os.getenv("VERIDIAN_MEDIA_ROOT", "/data3/ssharma8/veridian/media")
    frames_root: str = os.getenv("VERIDIAN_FRAMES_ROOT", "/data3/ssharma8/veridian/frames")

    # oracle weights / device — reuse sensveridian.config
    models_root: str = os.getenv("SV_MODELS_ROOT", str(MODELS.amod.parents[1]))
    device: str = os.getenv("SV_DEVICE", SETTINGS.device)
    face_match_threshold: float = float(
        os.getenv("SV_FACE_MATCH_THRESHOLD", str(SETTINGS.face_match_threshold))
    )

    # frame sampling / auto-label defaults. sample_fps is the target-fps the
    # OpenCV stride sampler aims for; dedup_stride applies the additional
    # select_every_nth near-duplicate dropper (1 = keep all decoded frames).
    source_fps: int = int(os.getenv("VERIDIAN_SOURCE_FPS", "30"))
    sample_fps: float = float(os.getenv("VERIDIAN_SAMPLE_FPS", "2"))
    dedup_stride: int = int(os.getenv("VERIDIAN_DEDUP_STRIDE", "1"))
    jpeg_quality: int = int(os.getenv("VERIDIAN_JPEG_QUALITY", "95"))

    # optional bearer token; when set, requests must present it (see deps.py)
    auth_token: str | None = os.getenv("VERIDIAN_AUTH_TOKEN") or None


settings = Settings()
