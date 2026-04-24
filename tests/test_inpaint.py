from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np

from sensveridian.augmentation.inpaint import LaMAInpainter, mask_sha


def test_mask_sha_deterministic_and_changes() -> None:
    m1 = np.zeros((8, 8), dtype=np.uint8)
    m2 = m1.copy()
    assert mask_sha(m1) == mask_sha(m2)
    m2[0, 0] = 1
    assert mask_sha(m2) != mask_sha(m1)


def test_inpaint_with_mock_pipe(tiny_image_bgr: np.ndarray) -> None:
    inp = LaMAInpainter(device="cpu")
    # Pipe returns PIL RGB-style array; keep identity-like behavior.
    fake_pipe = MagicMock()
    fake_pipe.return_value = cv2.cvtColor(tiny_image_bgr, cv2.COLOR_BGR2RGB)
    inp.pipe = fake_pipe
    mask = np.zeros(tiny_image_bgr.shape[:2], dtype=np.uint8)
    mask[10:20, 10:20] = 1

    out = inp.inpaint(tiny_image_bgr, mask)
    assert out.shape == tiny_image_bgr.shape
    assert out.dtype == tiny_image_bgr.dtype
    assert fake_pipe.call_count == 1


def test_get_or_create_plate_cache_miss_then_hit(tmp_path: Path, tiny_image_bgr: np.ndarray) -> None:
    plate_dir = tmp_path / "plates"
    mask = np.zeros(tiny_image_bgr.shape[:2], dtype=np.uint8)
    mask[5:12, 5:12] = 1

    inp = LaMAInpainter(device="cpu")
    # First call: force deterministic generated plate.
    generated = tiny_image_bgr.copy()
    generated[:] = (17, 33, 99)
    inp.inpaint = MagicMock(return_value=generated)

    plate1, plate_path1, msha1 = inp.get_or_create_plate(
        image_id="img_1", image_bgr=tiny_image_bgr, mask_u8=mask, plate_dir=plate_dir
    )
    assert plate1.shape == tiny_image_bgr.shape
    assert Path(plate_path1).exists()
    assert len(msha1) == 64
    assert inp.inpaint.call_count == 1

    # Second call: should hit cached file and not call inpaint again.
    inp.inpaint.reset_mock()
    plate2, plate_path2, msha2 = inp.get_or_create_plate(
        image_id="img_1", image_bgr=tiny_image_bgr, mask_u8=mask, plate_dir=plate_dir
    )
    assert plate_path2 == plate_path1
    assert msha2 == msha1
    assert inp.inpaint.call_count == 0
    # Read-back from disk can have exact equality for png.
    assert np.array_equal(plate2, cv2.imread(plate_path1, cv2.IMREAD_COLOR))
