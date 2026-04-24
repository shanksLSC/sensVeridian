from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
import cv2


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_decoded_image(path: Path) -> tuple[str, int, int]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Unable to decode image: {path}")
    h, w = img.shape[:2]
    digest = hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest()
    return digest, w, h

