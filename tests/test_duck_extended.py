from __future__ import annotations

from pathlib import Path

import duckdb

from sensveridian.store.duck import SummaryRow


def test_insert_augmentation_and_read_back(duck_store) -> None:
    duck_store.insert_augmentation(
        augmented_image_id="aug1",
        parent_image_id="img1",
        step_index=1,
        delta_ft=2.5,
        params={"step_ft": 1.0, "d_max_ft": 10.0},
    )
    df = duck_store.query_df("select parent_image_id, step_index, delta_ft from augmentations where augmented_image_id='aug1'")
    assert len(df) == 1
    assert df.iloc[0]["parent_image_id"] == "img1"
    assert int(df.iloc[0]["step_index"]) == 1
    assert float(df.iloc[0]["delta_ft"]) == 2.5


def test_upsert_depth_stat_and_bg_plate(duck_store) -> None:
    duck_store.upsert_depth_stat(
        image_id="img1",
        model_id="amod",
        detection_idx=0,
        bbox_xyxy=[1, 2, 10, 12],
        d_initial_ft=5.5,
        source="manual",
    )
    # Upsert overwrite path.
    duck_store.upsert_depth_stat(
        image_id="img1",
        model_id="amod",
        detection_idx=0,
        bbox_xyxy=[1, 2, 10, 12],
        d_initial_ft=6.0,
        source="zoe",
    )
    df = duck_store.query_df("select d_initial_ft, source from image_depth_stats where image_id='img1'")
    assert len(df) == 1
    assert float(df.iloc[0]["d_initial_ft"]) == 6.0
    assert df.iloc[0]["source"] == "zoe"

    duck_store.upsert_bg_plate("img1", "/tmp/plate.png", "abcd", inpainter="lama")
    bg = duck_store.query_df("select plate_path, inpainter from image_bg_plates where image_id='img1'")
    assert bg.iloc[0]["plate_path"] == "/tmp/plate.png"
    assert bg.iloc[0]["inpainter"] == "lama"


def test_export_parquet_writes_file(duck_store, tmp_path: Path) -> None:
    duck_store.upsert_image("img1", "/tmp/a.jpg", 10, 10)
    out = tmp_path / "rows.parquet"
    duck_store.export_parquet("select * from images", out)
    assert out.exists()
    con = duckdb.connect()
    try:
        df = con.execute(f"select count(*) as c from parquet_scan('{str(out)}')").df()
        assert int(df["c"].iloc[0]) == 1
    finally:
        con.close()


def test_wide_view_pivots_summary_columns(duck_store) -> None:
    duck_store.ensure_run("r1")
    duck_store.upsert_image("img1", "/tmp/a.jpg", 640, 480)
    duck_store.upsert_summary("img1", "r1", "amod", SummaryRow(True, 2, {"k": 1}))
    duck_store.upsert_summary("img1", "r1", "qrcode", SummaryRow(False, 0, {}))
    v = duck_store.query_df("select image_id, amod_present, n_amod, qrc_present, n_qrc from v_image_summary_wide where image_id='img1'")
    assert len(v) == 1
    row = v.iloc[0]
    assert row["image_id"] == "img1"
    assert bool(row["amod_present"]) is True
    assert int(row["n_amod"]) == 2
    assert bool(row["qrc_present"]) is False
    assert int(row["n_qrc"]) == 0
