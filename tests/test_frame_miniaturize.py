from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np

from sensveridian.augmentation.frame_miniaturize import (
    FrameMiniaturizer,
    miniaturize_frame,
    scale_for_distance_shift,
)
from sensveridian.augmentation.manual_distance import DistanceOverrides
from sensveridian.hashing import hash_decoded_image
from sensveridian.store.duck import SummaryRow


def _write_img(path: Path) -> None:
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    img[20:50, 20:50] = (255, 255, 255)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise RuntimeError(f"Failed to write {path}")


def _seed_raw_for_image(duck_store, image_id: str, run_id: str = "r1") -> None:
    duck_store.ensure_run(run_id)
    duck_store.upsert_image(image_id, "/tmp/x.png", 80, 80)
    duck_store.upsert_summary(image_id, run_id, "amod", SummaryRow(True, 1, {}))
    duck_store.upsert_raw(
        image_id=image_id,
        run_id=run_id,
        model_id="amod",
        payload={"detections": [{"bbox": [20, 20, 50, 50], "class_id": 0}]},
    )


def test_scale_for_distance_shift() -> None:
    assert scale_for_distance_shift(5.0, 0.0) == 1.0
    assert abs(scale_for_distance_shift(5.0, 5.0) - 0.5) < 1e-9


def test_miniaturize_frame_keeps_shape_and_shrinks() -> None:
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    img[10:30, 10:30] = 255
    out = miniaturize_frame(img, scale=0.5, pad_mode="black")
    assert out.shape == img.shape
    assert int(np.count_nonzero(out)) < int(np.count_nonzero(img))


def test_frame_miniaturizer_manual_mode_no_zoe(tmp_path: Path, duck_store, file_registry) -> None:
    img_path = tmp_path / "img.png"
    _write_img(img_path)
    image_id, _, _ = hash_decoded_image(img_path)
    _seed_raw_for_image(duck_store, image_id, run_id="r2")

    mini = FrameMiniaturizer(
        store=duck_store,
        orchestrator=MagicMock(),
        device="cpu",
    )
    mini.depth.estimate_depth_ft = MagicMock(return_value=np.ones((80, 80), dtype=np.float32) * 100.0)

    steps = mini.augment_image(
        image_path=img_path,
        run_id="mini_manual",
        d_max_ft=7.0,  # d0=5 -> 6,7 => two frames
        step_ft=1.0,
        source_models=["amod"],
        out_dir=tmp_path / "out",
        auto_run_oracle=False,
        overrides=DistanceOverrides(global_ft=5.0),
        pad_mode="black",
    )
    assert steps == 2
    assert mini.depth.estimate_depth_ft.call_count == 0

    aug = duck_store.query_df("select method, params from augmentations order by step_index")
    assert len(aug) == 2
    assert set(aug["method"].tolist()) == {"frame_miniaturize"}
