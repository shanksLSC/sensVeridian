from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelPaths:
    amod: Path = Path("/data3/ssharma8/all-models/AutomotiveMultiObjectDetection/amod-cpnx-8.2.0.h5")
    qrcode: Path = Path("/data3/ssharma8/all-models/QRCode/qr-code-detection-final.h5")
    fd: Path = Path("/data3/ssharma8/all-models/FaceDetection/fd_lnd_hp-fpga-8.1.0.h5")
    fr: Path = Path("/data3/ssharma8/all-models/FaceRecognition/fr-fpga-8.1.1.h5")


@dataclass(frozen=True)
class Settings:
    db_path: Path = PROJECT_ROOT / "sensveridian.duckdb"
    redis_url: str = os.getenv("SV_REDIS_URL", "redis://localhost:6379/0")
    cache_dir: Path = PROJECT_ROOT / "cache"
    bg_plate_dir: Path = PROJECT_ROOT / "cache" / "bg_plates"
    face_match_threshold: float = float(os.getenv("SV_FACE_MATCH_THRESHOLD", "0.5"))
    device: str = os.getenv("SV_DEVICE", "cuda")


SETTINGS = Settings()
MODELS = ModelPaths()

