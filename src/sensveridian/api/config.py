"""API settings — reads env (.env) and reuses ``sensveridian.config`` for the
oracle model paths / device / face-match threshold so there is a single source
of truth. Caches and media stay under /data3 per repo convention.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from ..config import MODELS, SETTINGS

# Bundled no-build front-end served at /studio (sits next to this package).
_STUDIO_DIR = Path(__file__).resolve().parent / "studio"


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

    # raw data store: the local filesystem (no S3). Ingest browses + reads here.
    datasets_root: str = os.getenv("VERIDIAN_DATASETS_ROOT", "/data3/ssharma8/datasets")

    # media + frame storage (keep off /home, per repo convention)
    media_root: str = os.getenv("VERIDIAN_MEDIA_ROOT", "/data3/ssharma8/veridian/media")
    frames_root: str = os.getenv("VERIDIAN_FRAMES_ROOT", "/data3/ssharma8/veridian/frames")

    # bundled front-end (served at /studio)
    studio_dir: str = os.getenv("VERIDIAN_STUDIO_DIR", str(_STUDIO_DIR))

    # default per-run image cap for a local-folder ingest (UI-adjustable)
    max_ingest_images: int = int(os.getenv("VERIDIAN_MAX_INGEST_IMAGES", "200"))

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

    # the qr-detection-bboxes project's sources dir (its model.py decode is
    # imported at runtime by the SqueezeDet QR runner)
    qr_detection_sources: str = os.getenv(
        "QR_DETECTION_SOURCES", "/data3/ssharma8/projects/qr-detection-bboxes/sources"
    )


settings = Settings()

# Models seeded at API startup (see AsyncPgStore.seed_models). Per the QR-only
# configuration this is just the two SqueezeDet QR detectors; each carries a
# runner_kind + config_path so the orchestrator can build a weights-driven
# runner. Add more via POST /models ("Register weights").
_QR_CFG_ROOT = "/data3/ssharma8/projects/qr-detection-bboxes/configs"
SEED_MODELS: list[dict] = [
    {
        "model_id": "qr_gray",
        "display_name": "QRCodeDetection (grayscale 4:3)",
        "version": "8.2",
        "runner_kind": "squeezedet_qr",
        "weights_path": f"{_QR_CFG_ROOT}/real_world_for_tight_bbox_4:3/convert/qr-detection-model-4_3-grayscaled-v8.2.h5",
        "config_path": f"{_QR_CFG_ROOT}/real_world_for_tight_bbox_4:3.yaml",
        "input_spec": "256x192x1",   # W x H x C
        "n_classes": 1,
    },
    {
        "model_id": "qr_rgb",
        "display_name": "QRCodeDetection (RGB 4:3)",
        "version": "best",
        "runner_kind": "squeezedet_qr",
        "weights_path": f"{_QR_CFG_ROOT}/qr_code_4_3_RGB/convert/model-sensai-h5-best.h5",
        "config_path": f"{_QR_CFG_ROOT}/qr_code_4_3_RGB.yaml",
        "input_spec": "256x192x3",   # W x H x C
        "n_classes": 1,
    },
]
