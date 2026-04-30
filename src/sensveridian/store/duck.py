from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import duckdb
import pandas as pd


@dataclass
class SummaryRow:
    present: bool
    count: int
    extras: dict


class DuckStore:
    def __init__(self, db_path: Path, schema_path: Path):
        self.db_path = db_path
        self.schema_path = schema_path
        self.con = duckdb.connect(str(db_path))

    def migrate(self) -> None:
        sql = self.schema_path.read_text(encoding="utf-8")
        self.con.execute(sql)

    def close(self) -> None:
        self.con.close()

    def ensure_run(self, run_id: str, code_hash: str = "", notes: str = "") -> None:
        self.con.execute(
            """
            INSERT INTO runs (run_id, code_hash, notes)
            VALUES (?, ?, ?)
            ON CONFLICT (run_id) DO UPDATE
            SET code_hash = excluded.code_hash, notes = excluded.notes
            """,
            [run_id, code_hash, notes],
        )

    def upsert_image(self, image_id: str, path: str, width: int, height: int) -> None:
        self.con.execute(
            """
            INSERT INTO images (image_id, path, width, height)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (image_id) DO UPDATE
            SET path = excluded.path, width = excluded.width, height = excluded.height
            """,
            [image_id, path, width, height],
        )

    def upsert_image_metadata(self, image_id: str, metadata: dict) -> None:
        self.con.execute(
            """
            UPDATE images
            SET metadata = ?::JSON
            WHERE image_id = ?
            """,
            [json.dumps(metadata), image_id],
        )

    def upsert_model(self, model_id: str, display_name: str, version: str, weights_path: str, weights_sha: str) -> None:
        self.con.execute(
            """
            INSERT INTO models (model_id, display_name, version, weights_path, weights_sha)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (model_id) DO UPDATE
            SET display_name = excluded.display_name,
                version = excluded.version,
                weights_path = excluded.weights_path,
                weights_sha = excluded.weights_sha
            """,
            [model_id, display_name, version, weights_path, weights_sha],
        )

    def upsert_summary(self, image_id: str, run_id: str, model_id: str, summary: SummaryRow) -> None:
        self.con.execute(
            """
            INSERT INTO predictions_summary (image_id, run_id, model_id, present, count, extras)
            VALUES (?, ?, ?, ?, ?, ?::JSON)
            ON CONFLICT (image_id, run_id, model_id) DO UPDATE
            SET present = excluded.present, count = excluded.count, extras = excluded.extras
            """,
            [image_id, run_id, model_id, summary.present, summary.count, json.dumps(summary.extras)],
        )

    def upsert_raw(self, image_id: str, run_id: str, model_id: str, payload: dict) -> None:
        self.con.execute(
            """
            INSERT INTO predictions_raw (image_id, run_id, model_id, payload)
            VALUES (?, ?, ?, ?::JSON)
            ON CONFLICT (image_id, run_id, model_id) DO UPDATE
            SET payload = excluded.payload
            """,
            [image_id, run_id, model_id, json.dumps(payload)],
        )

    def insert_augmentation(
        self,
        augmented_image_id: str,
        parent_image_id: str,
        step_index: int,
        delta_ft: float,
        params: dict,
        method: str = "distance_sweep",
    ) -> None:
        self.con.execute(
            """
            INSERT INTO augmentations (augmented_image_id, parent_image_id, method, step_index, delta_ft, params)
            VALUES (?, ?, ?, ?, ?, ?::JSON)
            ON CONFLICT (augmented_image_id) DO UPDATE
            SET parent_image_id = excluded.parent_image_id,
                method = excluded.method,
                step_index = excluded.step_index,
                delta_ft = excluded.delta_ft,
                params = excluded.params
            """,
            [augmented_image_id, parent_image_id, method, step_index, delta_ft, json.dumps(params)],
        )

    def upsert_depth_stat(
        self,
        image_id: str,
        model_id: str,
        detection_idx: int,
        bbox_xyxy: list[float],
        d_initial_ft: float,
        source: str = "zoe",
    ) -> None:
        self.con.execute(
            """
            INSERT INTO image_depth_stats (image_id, model_id, detection_idx, bbox_xyxy, d_initial_ft, source)
            VALUES (?, ?, ?, ?::JSON, ?, ?)
            ON CONFLICT (image_id, model_id, detection_idx) DO UPDATE
            SET bbox_xyxy = excluded.bbox_xyxy,
                d_initial_ft = excluded.d_initial_ft,
                source = excluded.source
            """,
            [image_id, model_id, detection_idx, json.dumps(bbox_xyxy), d_initial_ft, source],
        )

    def upsert_bg_plate(self, image_id: str, plate_path: str, mask_sha: str, inpainter: str = "lama") -> None:
        self.con.execute(
            """
            INSERT INTO image_bg_plates (image_id, plate_path, mask_sha, inpainter)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (image_id) DO UPDATE
            SET plate_path = excluded.plate_path, mask_sha = excluded.mask_sha, inpainter = excluded.inpainter
            """,
            [image_id, plate_path, mask_sha, inpainter],
        )

    def query_df(self, sql: str) -> pd.DataFrame:
        return self.con.execute(sql).df()

    def export_parquet(self, sql: str, out_path: Path) -> None:
        self.con.execute(f"COPY ({sql}) TO '{str(out_path)}' (FORMAT PARQUET)")

