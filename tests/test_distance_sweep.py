from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np

from sensveridian.augmentation.distance_sweep import DistanceAugmentor
from sensveridian.augmentation.camera import IMX219_PROFILE
from sensveridian.augmentation.manual_distance import DistanceOverrides
from sensveridian.hashing import hash_decoded_image
from sensveridian.store.duck import SummaryRow


def _write_img(path: Path) -> None:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[10:30, 10:30] = (255, 255, 255)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise RuntimeError(f"Failed to write {path}")


def _seed_raw_for_image(duck_store, image_id: str, run_id: str = "r1") -> None:
    duck_store.ensure_run(run_id)
    duck_store.upsert_image(image_id, "/tmp/x.png", 64, 64)
    duck_store.upsert_summary(image_id, run_id, "amod", SummaryRow(True, 1, {}))
    duck_store.upsert_raw(
        image_id=image_id,
        run_id=run_id,
        model_id="amod",
        payload={"detections": [{"bbox": [10, 10, 30, 30], "conf": 0.9}]},
    )


def test_distance_augmentor_uses_manual_override_without_zoe(tmp_path: Path, duck_store, file_registry) -> None:
    img_path = tmp_path / "img.png"
    _write_img(img_path)
    image_id, _, _ = hash_decoded_image(img_path)
    _seed_raw_for_image(duck_store, image_id)

    fake_orch = MagicMock()
    aug = DistanceAugmentor(
        store=duck_store,
        orchestrator=fake_orch,
        sam_checkpoint="/tmp/sam.pth",
        device="cpu",
    )
    # Mock heavy components.
    aug.segmenter.segment = MagicMock(return_value=[np.ones((64, 64), dtype=np.uint8)])
    aug.depth.estimate_depth_ft = MagicMock(return_value=np.ones((64, 64), dtype=np.float32) * 100.0)
    plate = np.zeros((64, 64, 3), dtype=np.uint8)
    aug.inpainter.get_or_create_plate = MagicMock(return_value=(plate, str(tmp_path / "plate.png"), "masksha"))

    steps = aug.augment_image(
        image_path=img_path,
        run_id="aug_manual",
        d_max_ft=7.0,
        step_ft=1.0,
        source_models=["amod"],
        out_dir=tmp_path / "out_manual",
        auto_run_oracle=False,
        overrides=DistanceOverrides(global_ft=5.0),
    )
    assert steps == 2  # deltas: 1.0, 2.0
    assert aug.depth.estimate_depth_ft.call_count == 0

    ds = duck_store.query_df("select source, d_initial_ft from image_depth_stats")
    assert len(ds) == 1
    assert ds.iloc[0]["source"] == "manual"
    assert float(ds.iloc[0]["d_initial_ft"]) == 5.0


def test_distance_augmentor_zoe_fallback_and_auto_oracle(tmp_path: Path, duck_store, file_registry) -> None:
    img_path = tmp_path / "img2.png"
    _write_img(img_path)
    image_id, _, _ = hash_decoded_image(img_path)
    _seed_raw_for_image(duck_store, image_id, run_id="r2")

    fake_orch = MagicMock()
    aug = DistanceAugmentor(
        store=duck_store,
        orchestrator=fake_orch,
        sam_checkpoint="/tmp/sam.pth",
        device="cpu",
    )
    aug.segmenter.segment = MagicMock(return_value=[np.ones((64, 64), dtype=np.uint8)])
    aug.depth.estimate_depth_ft = MagicMock(return_value=np.ones((64, 64), dtype=np.float32) * 4.0)
    plate = np.zeros((64, 64, 3), dtype=np.uint8)
    aug.inpainter.get_or_create_plate = MagicMock(return_value=(plate, str(tmp_path / "plate2.png"), "masksha2"))

    steps = aug.augment_image(
        image_path=img_path,
        run_id="aug_zoe",
        d_max_ft=5.0,  # with d0=4 and step=1 -> exactly one step
        step_ft=1.0,
        source_models=["amod"],
        out_dir=tmp_path / "out_zoe",
        auto_run_oracle=True,
        overrides=DistanceOverrides.empty(),
    )
    assert steps == 1
    assert aug.depth.estimate_depth_ft.call_count == 1
    assert fake_orch.ingest.call_count == 1

    ds = duck_store.query_df(f"select source from image_depth_stats where image_id='{image_id}'")
    assert "zoe" in set(ds["source"].tolist())


def test_distance_augmentor_uses_camera_calibration_without_zoe(tmp_path: Path, duck_store, file_registry) -> None:
    img_path = tmp_path / "img3.png"
    _write_img(img_path)
    image_id, _, _ = hash_decoded_image(img_path)
    _seed_raw_for_image(duck_store, image_id, run_id="r3")

    fake_orch = MagicMock()
    aug = DistanceAugmentor(
        store=duck_store,
        orchestrator=fake_orch,
        sam_checkpoint="/tmp/sam.pth",
        device="cpu",
        camera_profile=IMX219_PROFILE,
    )
    assert aug.depth is None
    aug.segmenter.segment = MagicMock(return_value=[np.ones((64, 64), dtype=np.uint8)])
    plate = np.zeros((64, 64, 3), dtype=np.uint8)
    aug.inpainter.get_or_create_plate = MagicMock(return_value=(plate, str(tmp_path / "plate3.png"), "masksha3"))

    steps = aug.augment_image(
        image_path=img_path,
        run_id="aug_calib",
        d_max_ft=30.0,
        step_ft=1.0,
        source_models=["amod"],
        out_dir=tmp_path / "out_calib",
        auto_run_oracle=False,
        overrides=DistanceOverrides.empty(),
    )
    assert steps > 0

    ds = duck_store.query_df(f"select source from image_depth_stats where image_id='{image_id}'")
    assert "calib" in set(ds["source"].tolist())
