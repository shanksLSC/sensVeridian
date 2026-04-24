from __future__ import annotations

from .store.faces_registry import FaceRegistry


def seed_dummy_faces(registry: FaceRegistry, n: int, embedding_dim: int = 128) -> None:
    registry.seed_dummy(n=n, embedding_dim=embedding_dim)

