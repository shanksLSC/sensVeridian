"""Face-recognition interpreter — embedding L2-normalize + cosine similarity.

Python port of ``embedding.c`` ``face_recognition_post_process`` (float path).
Unlike the detectors this returns a unit embedding; identity matching against
the gallery is done by the runner via the FaceRegistry (cosine >= threshold).
"""
from __future__ import annotations

import numpy as np


def interpret_embedding(output) -> np.ndarray:
    """L2-normalize the raw model embedding to a unit vector."""
    v = np.asarray(output, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def cosine_similarity(a, b) -> float:
    a = interpret_embedding(a)
    b = interpret_embedding(b)
    return float(np.dot(a, b))
