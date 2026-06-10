"""Storage types shared across the codebase.

``SummaryRow`` and the ``Store`` typing Protocol used to live in ``duck.py``;
they moved here when DuckDB was removed so the Orchestrator, augmentors, CLI,
and the PostgreSQL store depend on a backend-neutral contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

import pandas as pd


@dataclass
class SummaryRow:
    present: bool
    count: int
    extras: dict


@runtime_checkable
class Store(Protocol):
    """The synchronous store contract the Orchestrator + augmentors rely on.

    Implemented by :class:`sensveridian.store.pg.PgStore`. Method names mirror
    the historical DuckStore surface so existing call sites are unchanged.
    """

    def migrate(self) -> None: ...
    def close(self) -> None: ...
    def ensure_run(self, run_id: str, code_hash: str = "", notes: str = "") -> None: ...
    def upsert_image(self, image_id: str, path: str, width: int, height: int,
                     dataset_id: Optional[str] = None) -> None: ...
    def upsert_image_metadata(self, image_id: str, metadata: dict) -> None: ...
    def upsert_model(self, model_id: str, display_name: str, version: str,
                     weights_path: str, weights_sha: str) -> None: ...
    def upsert_summary(self, image_id: str, run_id: str, model_id: str, summary: Any) -> None: ...
    def upsert_raw(self, image_id: str, run_id: str, model_id: str, payload: dict) -> None: ...
    def insert_augmentation(self, augmented_image_id: str, parent_image_id: str,
                            step_index: int, delta_ft: float, params: dict,
                            method: str = "distance_sweep") -> None: ...
    def upsert_depth_stat(self, image_id: str, model_id: str, detection_idx: int,
                          bbox_xyxy: list, d_initial_ft: float, source: str = "zoe") -> None: ...
    def upsert_bg_plate(self, image_id: str, plate_path: str, mask_sha: str,
                        inpainter: str = "lama") -> None: ...
    def query_df(self, sql: str) -> "pd.DataFrame": ...
    def export_parquet(self, sql: str, out_path: Path) -> None: ...
