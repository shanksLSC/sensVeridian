"""Store-method tests against PostgreSQL (PgStore), ported from the former
DuckDB store tests. Uses the session `sensveridian_test` database fixture."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from sensveridian.store.types import SummaryRow


def test_pg_store_roundtrip(pg_store) -> None:
    pg_store.ensure_run("r1")
    pg_store.upsert_image("img1", "/tmp/a.jpg", 640, 480)
    pg_store.upsert_model("m1", "Model1", "v1", "/tmp/m.h5", "sha")
    pg_store.upsert_summary("img1", "r1", "m1", SummaryRow(present=True, count=2, extras={"k": 1}))
    pg_store.upsert_raw("img1", "r1", "m1", {"detections": [{"bbox": [1, 2, 3, 4]}]})
    df = pg_store.query_df("select count(*) c from predictions_summary")
    assert int(df["c"].iloc[0]) == 1


def test_insert_augmentation_and_read_back(pg_store) -> None:
    pg_store.upsert_image("img1", "/tmp/a.jpg", 10, 10)  # FK parent must exist
    pg_store.insert_augmentation(
        augmented_image_id="aug1", parent_image_id="img1",
        step_index=1, delta_ft=2.5, params={"step_ft": 1.0, "d_max_ft": 10.0},
    )
    df = pg_store.query_df(
        "select parent_image_id, step_index, delta_ft from augmentations where augmented_image_id='aug1'"
    )
    assert len(df) == 1
    assert df.iloc[0]["parent_image_id"] == "img1"
    assert int(df.iloc[0]["step_index"]) == 1
    assert float(df.iloc[0]["delta_ft"]) == 2.5


def test_upsert_depth_stat_and_bg_plate(pg_store) -> None:
    pg_store.upsert_depth_stat("img1", "amod", 0, [1, 2, 10, 12], 5.5, source="manual")
    pg_store.upsert_depth_stat("img1", "amod", 0, [1, 2, 10, 12], 6.0, source="zoe")  # overwrite
    df = pg_store.query_df("select d_initial_ft, source from image_depth_stats where image_id='img1'")
    assert len(df) == 1
    assert float(df.iloc[0]["d_initial_ft"]) == 6.0
    assert df.iloc[0]["source"] == "zoe"

    pg_store.upsert_bg_plate("img1", "/tmp/plate.png", "abcd", inpainter="lama")
    bg = pg_store.query_df("select plate_path, inpainter from image_bg_plates where image_id='img1'")
    assert bg.iloc[0]["plate_path"] == "/tmp/plate.png"
    assert bg.iloc[0]["inpainter"] == "lama"


def test_export_parquet_writes_file(pg_store, tmp_path: Path) -> None:
    pg_store.upsert_image("img1", "/tmp/a.jpg", 10, 10)
    out = tmp_path / "rows.parquet"
    pg_store.export_parquet("select * from images", out)
    assert out.exists()
    assert len(pd.read_parquet(out)) == 1


def _seed_eval(pg_store, dataset_id: str, image_id: str, gt_anns: list[dict],
               class_map: dict | None = None) -> None:
    """Seed one image with one AMOD prediction + GT for an eval-metrics test."""
    pg_store.ensure_run("baseline")
    pg_store.ensure_dataset(dataset_id, dataset_id.upper(), models=["amod"], mode="eval",
                            class_map=class_map)
    pg_store.upsert_model("amod", "AMOD", "v1", "/w/amod.h5", "sha")
    pg_store.upsert_image(image_id, f"/tmp/{image_id}.jpg", 100, 100, dataset_id=dataset_id)
    pg_store.upsert_raw(image_id, "baseline", "amod",
                        {"detections": [{"bbox": [0.1, 0.1, 0.6, 0.6], "conf": 0.9, "class_id": 0}]})
    pg_store.upsert_layer(f"gt:{dataset_id}", dataset_id, type="ground-truth")
    pg_store.replace_image_gt(f"gt:{dataset_id}", dataset_id, image_id, f"{image_id}.jpg", gt_anns)


def test_compute_eval_metrics_with_class_map(pg_store) -> None:
    # GT label "pedestrian" maps to AMOD class_id 0 ("person"); boxes coincide.
    _seed_eval(pg_store, "d1", "i1",
               [{"cls": "pedestrian", "box": [0.1, 0.1, 0.5, 0.5]}],
               class_map={"pedestrian": "person"})
    met = pg_store.compute_eval_metrics("d1", "amod", "baseline")
    assert (met["tp"], met["fp"], met["fn"]) == (1, 0, 0)
    assert met["precision"] == 1.0 and met["recall"] == 1.0
    assert met["AP50"] == 1.0
    # persisted to eval_metrics
    df = pg_store.query_df("select metrics from eval_metrics where dataset_id='d1' and model_id='amod'")
    assert len(df) == 1


def test_compute_eval_metrics_unmapped_gt_is_ignored(pg_store) -> None:
    # an extra GT class outside the map is dropped, so it is not counted as a miss
    _seed_eval(pg_store, "d2", "i2",
               [{"cls": "pedestrian", "box": [0.1, 0.1, 0.5, 0.5]},
                {"cls": "traffic_cone", "box": [0.8, 0.8, 0.1, 0.1]}],
               class_map={"pedestrian": "person"})
    met = pg_store.compute_eval_metrics("d2", "amod", "baseline")
    assert (met["tp"], met["fp"], met["fn"]) == (1, 0, 0)


def test_compute_eval_metrics_no_class_map_uses_raw_labels(pg_store) -> None:
    # without a class_map, GT labels compare as-is: "person" GT vs "person" pred
    _seed_eval(pg_store, "d3", "i3",
               [{"cls": "person", "box": [0.1, 0.1, 0.5, 0.5]}], class_map=None)
    met = pg_store.compute_eval_metrics("d3", "amod", "baseline")
    assert (met["tp"], met["fp"], met["fn"]) == (1, 0, 0)


def test_wide_view_pivots_summary_columns(pg_store) -> None:
    pg_store.ensure_run("r1")
    pg_store.upsert_image("img1", "/tmp/a.jpg", 640, 480)
    pg_store.upsert_summary("img1", "r1", "amod", SummaryRow(True, 2, {"k": 1}))
    pg_store.upsert_summary("img1", "r1", "qrcode", SummaryRow(False, 0, {"decoded_count": 0}))
    pg_store.refresh_summary_view()  # materialized view in PostgreSQL
    v = pg_store.query_df(
        "select image_id, amod_present, n_amod, qrc_present, n_qrc "
        "from v_image_summary_wide where image_id='img1'"
    )
    assert len(v) == 1
    row = v.iloc[0]
    assert row["image_id"] == "img1"
    assert bool(row["amod_present"]) is True
    assert int(row["n_amod"]) == 2
    assert bool(row["qrc_present"]) is False
    assert int(row["n_qrc"]) == 0
