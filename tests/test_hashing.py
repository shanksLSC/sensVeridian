from __future__ import annotations

from pathlib import Path

import pytest

from sensveridian.hashing import hash_decoded_image, hash_file


def test_hash_file_is_deterministic_and_changes(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc123")
    h1 = hash_file(p)
    h2 = hash_file(p)
    assert h1 == h2

    p.write_bytes(b"abc1234")
    h3 = hash_file(p)
    assert h3 != h1


def test_hash_decoded_image_shape_and_digest_stability(tiny_image_file: Path) -> None:
    d1, w1, h1 = hash_decoded_image(tiny_image_file)
    d2, w2, h2 = hash_decoded_image(tiny_image_file)
    assert (w1, h1) == (64, 64)
    assert (w1, h1) == (w2, h2)
    assert d1 == d2
    assert len(d1) == 64


def test_hash_decoded_image_raises_on_bad_file(tmp_path: Path) -> None:
    p = tmp_path / "not_image.txt"
    p.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError):
        hash_decoded_image(p)
