from pathlib import Path

from sensveridian.store.duck import DuckStore, SummaryRow


def test_duck_store_roundtrip(tmp_path: Path):
    db = tmp_path / "t.duckdb"
    schema = Path(__file__).resolve().parents[1] / "src" / "sensveridian" / "store" / "schema.sql"
    s = DuckStore(db, schema)
    s.migrate()
    s.ensure_run("r1")
    s.upsert_image("img1", "/tmp/a.jpg", 640, 480)
    s.upsert_model("m1", "Model1", "v1", "/tmp/m.h5", "sha")
    s.upsert_summary("img1", "r1", "m1", SummaryRow(present=True, count=2, extras={"k": 1}))
    s.upsert_raw("img1", "r1", "m1", {"detections": [{"bbox": [1, 2, 3, 4]}]})
    df = s.query_df("select count(*) c from predictions_summary")
    assert int(df["c"].iloc[0]) == 1
    s.close()

