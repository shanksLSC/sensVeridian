"""PgStore — synchronous PostgreSQL successor to ``DuckStore``.

This is the store the :class:`~sensveridian.orchestrator.Orchestrator` and the
ingest worker use. It deliberately keeps the **same method names and synchronous
semantics** as :class:`sensveridian.store.duck.DuckStore` so the Orchestrator
runs unchanged — it just receives a ``PgStore`` instead of a ``DuckStore``.

Why synchronous?  The Orchestrator does blocking CPU/GPU work (TensorFlow
inference, OpenCV decode) and calls ``store.query_df(sql)`` then immediately
consumes a pandas ``DataFrame``. That code path must not live on the asyncio
event loop. The FastAPI request handlers, which are I/O bound, use the *async*
read store in :mod:`sensveridian.store.pg_async` instead. Both talk to the same
schema (``schema_pg.sql``).

The only dialect work versus DuckDB is ``?::JSON`` -> ``CAST(:p AS jsonb)`` and
qmark -> named parameters; ``ON CONFLICT ... DO UPDATE SET col = excluded.col``
is valid in both engines.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine

# Re-exported so callers can build summaries from the store package.
from .types import SummaryRow  # noqa: F401

DEFAULT_SCHEMA = Path(__file__).resolve().parent / "schema_pg.sql"


def _json(value: Any) -> str:
    """Serialize a Python value for a jsonb bind parameter."""
    return json.dumps(value if value is not None else {})


class PgStore:
    """Synchronous, DuckStore-compatible store backed by PostgreSQL.

    Parameters
    ----------
    database_url:
        A SQLAlchemy URL using a *synchronous* driver, e.g.
        ``postgresql+psycopg://user:pw@host/db``. An ``+asyncpg`` URL is
        accepted and rewritten to the sync driver for convenience.
    schema_path:
        Path to ``schema_pg.sql`` (defaults to the file next to this module).
    schema:
        Optional PostgreSQL schema to place + resolve all objects in (sets the
        connection ``search_path`` to ``<schema>,public``). Used by the test
        suite to isolate a ``sv_test`` schema from the live ``public`` data
        without needing a separate database / CREATEDB privilege.
    """

    def __init__(self, database_url: str, schema_path: Path | str = DEFAULT_SCHEMA,
                 schema: Optional[str] = None):
        self.database_url = sync_url(database_url)
        self.schema_path = Path(schema_path)
        self.schema = schema
        connect_args = {"options": f"-csearch_path={schema},public"} if schema else {}
        self.engine: Engine = create_engine(self.database_url, future=True,
                                            pool_pre_ping=True, connect_args=connect_args)
        self._con: Optional[Connection] = None

    # ---- connection management (mirrors DuckStore's single-connection model) --
    @property
    def con(self) -> Connection:
        """A lazily-opened autocommit connection (matches DuckDB semantics:
        every statement is durable immediately)."""
        if self._con is None or self._con.closed:
            self._con = self.engine.connect().execution_options(isolation_level="AUTOCOMMIT")
        return self._con

    def close(self) -> None:
        if self._con is not None and not self._con.closed:
            self._con.close()
        self._con = None
        self.engine.dispose()

    def migrate(self) -> None:
        """Apply schema_pg.sql. The canonical path in production is
        ``psql -f schema_pg.sql``; this is the convenience equivalent."""
        if self.schema:
            self.con.exec_driver_sql(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"')
        sql = self.schema_path.read_text(encoding="utf-8")
        # exec_driver_sql sends the script straight to the DBAPI driver, which
        # accepts multiple ';'-separated statements in one call.
        self.con.exec_driver_sql(sql)

    # ---- writes used by Orchestrator (same names/shape as DuckStore) ----------
    def ensure_run(self, run_id: str, code_hash: str = "", notes: str = "") -> None:
        self.con.execute(
            text(
                """
                INSERT INTO runs (run_id, code_hash, notes)
                VALUES (:r, :c, :n)
                ON CONFLICT (run_id) DO UPDATE
                SET code_hash = excluded.code_hash, notes = excluded.notes
                """
            ),
            {"r": run_id, "c": code_hash, "n": notes},
        )

    def upsert_image(
        self,
        image_id: str,
        path: str,
        width: int,
        height: int,
        dataset_id: Optional[str] = None,
    ) -> None:
        # dataset_id is an additive parameter (the worker sets it; the
        # Orchestrator does not). COALESCE keeps an existing owner if not given.
        self.con.execute(
            text(
                """
                INSERT INTO images (image_id, path, width, height, dataset_id)
                VALUES (:i, :p, :w, :h, :d)
                ON CONFLICT (image_id) DO UPDATE
                SET path = excluded.path, width = excluded.width, height = excluded.height,
                    dataset_id = COALESCE(excluded.dataset_id, images.dataset_id)
                """
            ),
            {"i": image_id, "p": path, "w": width, "h": height, "d": dataset_id},
        )

    def upsert_image_metadata(self, image_id: str, metadata: dict) -> None:
        self.con.execute(
            text("UPDATE images SET metadata = CAST(:m AS jsonb) WHERE image_id = :i"),
            {"m": _json(metadata), "i": image_id},
        )

    def upsert_model(
        self, model_id: str, display_name: str, version: str, weights_path: str, weights_sha: str
    ) -> None:
        self.con.execute(
            text(
                """
                INSERT INTO models (model_id, display_name, version, weights_path, weights_sha)
                VALUES (:m, :d, :v, :p, :s)
                ON CONFLICT (model_id) DO UPDATE
                SET display_name = excluded.display_name,
                    version = excluded.version,
                    weights_path = excluded.weights_path,
                    weights_sha = excluded.weights_sha
                """
            ),
            {"m": model_id, "d": display_name, "v": version, "p": weights_path, "s": weights_sha},
        )

    def upsert_summary(self, image_id: str, run_id: str, model_id: str, summary: SummaryRow) -> None:
        self.con.execute(
            text(
                """
                INSERT INTO predictions_summary (image_id, run_id, model_id, present, count, extras)
                VALUES (:i, :r, :m, :present, :count, CAST(:extras AS jsonb))
                ON CONFLICT (image_id, run_id, model_id) DO UPDATE
                SET present = excluded.present, count = excluded.count, extras = excluded.extras
                """
            ),
            {
                "i": image_id,
                "r": run_id,
                "m": model_id,
                "present": summary.present,
                "count": summary.count,
                "extras": _json(summary.extras),
            },
        )

    def upsert_raw(self, image_id: str, run_id: str, model_id: str, payload: dict) -> None:
        self.con.execute(
            text(
                """
                INSERT INTO predictions_raw (image_id, run_id, model_id, payload)
                VALUES (:i, :r, :m, CAST(:p AS jsonb))
                ON CONFLICT (image_id, run_id, model_id) DO UPDATE
                SET payload = excluded.payload
                """
            ),
            {"i": image_id, "r": run_id, "m": model_id, "p": _json(payload)},
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
            text(
                """
                INSERT INTO augmentations (augmented_image_id, parent_image_id, method, step_index, delta_ft, params)
                VALUES (:a, :par, :method, :step, :delta, CAST(:params AS jsonb))
                ON CONFLICT (augmented_image_id) DO UPDATE
                SET parent_image_id = excluded.parent_image_id,
                    method = excluded.method,
                    step_index = excluded.step_index,
                    delta_ft = excluded.delta_ft,
                    params = excluded.params
                """
            ),
            {
                "a": augmented_image_id,
                "par": parent_image_id,
                "method": method,
                "step": step_index,
                "delta": delta_ft,
                "params": _json(params),
            },
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
            text(
                """
                INSERT INTO image_depth_stats (image_id, model_id, detection_idx, bbox_xyxy, d_initial_ft, source)
                VALUES (:i, :m, :idx, CAST(:bbox AS jsonb), :d, :src)
                ON CONFLICT (image_id, model_id, detection_idx) DO UPDATE
                SET bbox_xyxy = excluded.bbox_xyxy,
                    d_initial_ft = excluded.d_initial_ft,
                    source = excluded.source
                """
            ),
            {
                "i": image_id,
                "m": model_id,
                "idx": detection_idx,
                "bbox": _json(bbox_xyxy),
                "d": d_initial_ft,
                "src": source,
            },
        )

    def upsert_bg_plate(self, image_id: str, plate_path: str, mask_sha: str, inpainter: str = "lama") -> None:
        self.con.execute(
            text(
                """
                INSERT INTO image_bg_plates (image_id, plate_path, mask_sha, inpainter)
                VALUES (:i, :p, :s, :inp)
                ON CONFLICT (image_id) DO UPDATE
                SET plate_path = excluded.plate_path, mask_sha = excluded.mask_sha, inpainter = excluded.inpainter
                """
            ),
            {"i": image_id, "p": plate_path, "s": mask_sha, "inp": inpainter},
        )

    def query_df(self, sql: str) -> pd.DataFrame:
        """Run an arbitrary SQL string and return a DataFrame.

        Uses ``exec_driver_sql`` so the statement is passed verbatim to the
        driver (no SQLAlchemy ``:name`` bind-parameter parsing) — the
        Orchestrator builds fully-formed SQL strings. jsonb columns come back as
        Python dict/list objects, which ``Orchestrator._loads_json`` handles.
        """
        result = self.con.exec_driver_sql(sql)
        cols = list(result.keys())
        rows = result.fetchall()
        return pd.DataFrame(rows, columns=cols)

    def export_parquet(self, sql: str, out_path: Path) -> None:
        """DuckDB used ``COPY (...) TO ... (FORMAT PARQUET)``; PostgreSQL has no
        direct equivalent, so materialize via pandas."""
        self.query_df(sql).to_parquet(out_path)

    # ---- additional writes the Veridian worker / API need --------------------
    def ensure_dataset(
        self,
        dataset_id: str,
        name: str,
        kind: str = "vision",
        descr: str = "",
        models: Optional[list[str]] = None,
        palette: Optional[str] = None,
        run_id: Optional[str] = None,
        mode: str = "eval",
        label_format: Optional[str] = None,
        labels_dir: Optional[str] = None,
        class_names: Optional[dict] = None,
        class_map: Optional[dict] = None,
    ) -> None:
        self.con.execute(
            text(
                """
                INSERT INTO datasets (dataset_id, name, descr, kind, models, palette, run_id,
                                      mode, label_format, labels_dir, class_names, class_map)
                VALUES (:id, :name, :descr, :kind, :models, :palette, :run_id,
                        :mode, :lfmt, :ldir, CAST(:cnames AS jsonb), CAST(:cmap AS jsonb))
                ON CONFLICT (dataset_id) DO UPDATE
                SET name = excluded.name, descr = excluded.descr, kind = excluded.kind,
                    models = excluded.models, palette = excluded.palette,
                    run_id = COALESCE(excluded.run_id, datasets.run_id),
                    mode = excluded.mode,
                    label_format = COALESCE(excluded.label_format, datasets.label_format),
                    labels_dir = COALESCE(excluded.labels_dir, datasets.labels_dir),
                    class_names = COALESCE(excluded.class_names, datasets.class_names),
                    -- existing (UI-edited) class_map wins; only seed when unset
                    class_map = COALESCE(datasets.class_map, excluded.class_map)
                """
            ),
            {
                "id": dataset_id,
                "name": name,
                "descr": descr,
                "kind": kind,
                "models": models or [],
                "palette": palette,
                "run_id": run_id,
                "mode": mode,
                "lfmt": label_format,
                "ldir": labels_dir,
                "cnames": _json(class_names) if class_names is not None else None,
                "cmap": _json(class_map) if class_map is not None else None,
            },
        )

    def set_class_map(self, dataset_id: str, class_map: dict) -> None:
        self.con.execute(
            text("UPDATE datasets SET class_map = CAST(:m AS jsonb) WHERE dataset_id = :d"),
            {"m": _json(class_map), "d": dataset_id},
        )

    def upsert_eval_metrics(self, dataset_id: str, model_id: str, run_id: str, metrics: dict) -> None:
        self.con.execute(
            text(
                """
                INSERT INTO eval_metrics (dataset_id, model_id, run_id, metrics, computed_at)
                VALUES (:d, :m, :r, CAST(:met AS jsonb), now())
                ON CONFLICT (dataset_id, model_id, run_id) DO UPDATE
                SET metrics = excluded.metrics, computed_at = now()
                """
            ),
            {"d": dataset_id, "m": model_id, "r": run_id, "met": _json(metrics)},
        )

    def compute_eval_metrics(self, dataset_id: str, model_id: str, run_id: str = "baseline") -> dict:
        """Build per-image (pred vs mapped-GT) samples for one model+run over a
        dataset, compute Path-2 metrics, and persist them to ``eval_metrics``.

        Synchronous twin of the API's on-demand compute, used at the end of an
        ``eval`` ingest. The pure metric math lives in
        :mod:`sensveridian.api.metrics`; prediction decoding + GT class
        alignment reuse :mod:`sensveridian.api.fusion`.
        """
        from ..api import fusion
        from ..api import metrics as M

        def _esc(s: Any) -> str:
            return str(s).replace("'", "''")

        def _obj(v: Any) -> Any:
            if isinstance(v, str):
                try:
                    return json.loads(v)
                except Exception:
                    return None
            return v

        did, mid, rid = _esc(dataset_id), _esc(model_id), _esc(run_id)

        cm = self.query_df(f"SELECT class_map FROM datasets WHERE dataset_id = '{did}'")
        class_map = _obj(cm.iloc[0]["class_map"]) if not cm.empty else None

        imgs = self.query_df(f"SELECT image_id, width, height FROM images WHERE dataset_id = '{did}'")
        samples: list[dict] = []
        for _, im in imgs.iterrows():
            iid = _esc(im["image_id"])
            w = int(im["width"] or 0)
            h = int(im["height"] or 0)
            pr = self.query_df(
                "SELECT payload FROM predictions_raw "
                f"WHERE image_id = '{iid}' AND run_id = '{rid}' AND model_id = '{mid}'"
            )
            payload = _obj(pr.iloc[0]["payload"]) if not pr.empty else {}
            preds = fusion.extract_pred_detections(model_id, payload or {}, w, h)
            gtd = self.query_df(
                f"SELECT cls, box FROM gt_boxes WHERE dataset_id = '{did}' AND image_id = '{iid}'"
            )
            gt: list[dict] = []
            for _, g in gtd.iterrows():
                cls = g["cls"]
                box = _obj(g["box"])
                if class_map:
                    if cls not in class_map:
                        continue  # GT class the model never predicts -> ignore
                    cls = class_map[cls]
                if box:
                    gt.append({"box": box, "cls": cls})
            samples.append(
                {
                    "preds": [
                        {"box": p["box"], "cls": p["cls"], "conf": p.get("conf", 0.0)}
                        for p in preds
                        if p.get("box")
                    ],
                    "gt": gt,
                }
            )

        result = M.compute_metrics(samples)
        self.upsert_eval_metrics(dataset_id, model_id, run_id, result)
        return result

    def replace_image_gt(self, layer_id: str, dataset_id: str, image_id: str,
                         image_ref: str, anns: list[dict]) -> int:
        """Replace the ground-truth boxes for one image (idempotent on
        re-ingest): delete this image's rows in the layer, then insert ``anns``
        (each ``{cls, box:[x,y,w,h] normalized}``). Returns the count inserted."""
        self.con.execute(
            text("DELETE FROM gt_boxes WHERE layer_id = :lid AND image_id = :iid"),
            {"lid": layer_id, "iid": image_id},
        )
        n = 0
        for a in anns:
            box = a.get("box") or [0, 0, 0, 0]
            x, y, w, h = (max(0.0, min(1.0, float(v))) for v in box[:4])
            self.con.execute(
                text(
                    """
                    INSERT INTO gt_boxes (layer_id, dataset_id, image_id, image_ref, cls, box, meta)
                    VALUES (:lid, :did, :iid, :iref, :cls, CAST(:box AS jsonb), CAST(:meta AS jsonb))
                    """
                ),
                {"lid": layer_id, "did": dataset_id, "iid": image_id, "iref": image_ref,
                 "cls": a.get("cls"), "box": _json([x, y, w, h]),
                 "meta": _json(a.get("meta") or {})},
            )
            n += 1
        return n

    def register_model_card(
        self,
        model_id: str,
        display_name: str,
        version: str,
        weights_path: str,
        weights_sha: str,
        input_spec: str = "",
        n_classes: int = 0,
        depends_on: Optional[str] = None,
    ) -> None:
        """Full model registration including the UI metadata columns
        (``input_spec``/``n_classes``/``depends_on``) that ``upsert_model`` does
        not carry (it stays DuckStore-compatible)."""
        self.con.execute(
            text(
                """
                INSERT INTO models (model_id, display_name, version, weights_path, weights_sha, input_spec, n_classes, depends_on)
                VALUES (:m, :d, :v, :p, :s, :ispec, :nc, :dep)
                ON CONFLICT (model_id) DO UPDATE
                SET display_name = excluded.display_name, version = excluded.version,
                    weights_path = excluded.weights_path, weights_sha = excluded.weights_sha,
                    input_spec = excluded.input_spec, n_classes = excluded.n_classes,
                    depends_on = excluded.depends_on
                """
            ),
            {
                "m": model_id,
                "d": display_name,
                "v": version,
                "p": weights_path,
                "s": weights_sha,
                "ispec": input_spec,
                "nc": n_classes,
                "dep": depends_on,
            },
        )

    def upsert_model_version(
        self,
        model_id: str,
        version: str,
        weights_sha: str = "",
        released_on: Optional[str] = None,
        metrics: Optional[dict] = None,
        notes: str = "",
        is_current: bool = False,
    ) -> None:
        self.con.execute(
            text(
                """
                INSERT INTO model_versions (model_id, version, weights_sha, released_on, metrics, notes, is_current)
                VALUES (:m, :v, :s, :date, CAST(:metrics AS jsonb), :notes, :cur)
                ON CONFLICT (model_id, version) DO UPDATE
                SET weights_sha = excluded.weights_sha, released_on = excluded.released_on,
                    metrics = excluded.metrics, notes = excluded.notes, is_current = excluded.is_current
                """
            ),
            {
                "m": model_id,
                "v": version,
                "s": weights_sha,
                "date": released_on,
                "metrics": _json(metrics),
                "notes": notes,
                "cur": is_current,
            },
        )

    def set_model_version_metrics(self, model_id: str, metrics: dict) -> None:
        """Update only the current model_version's metrics blob (no change to
        ``is_current``). Used to surface the latest eval metrics in the model
        manager; ``eval_metrics`` remains the per-dataset source of truth."""
        self.con.execute(
            text(
                """
                UPDATE model_versions SET metrics = CAST(:m AS jsonb)
                WHERE model_id = :mid AND is_current = TRUE
                """
            ),
            {"m": _json(metrics), "mid": model_id},
        )

    def upsert_layer(
        self,
        layer_id: str,
        dataset_id: str,
        type: str,
        model_id: Optional[str] = None,
        version: Optional[str] = None,
        run_id: Optional[str] = None,
        source: Optional[str] = None,
    ) -> None:
        self.con.execute(
            text(
                """
                INSERT INTO layers (layer_id, dataset_id, type, model_id, version, run_id, source)
                VALUES (:lid, :did, :type, :mid, :ver, :run, :src)
                ON CONFLICT (layer_id) DO UPDATE
                SET type = excluded.type, model_id = excluded.model_id, version = excluded.version,
                    run_id = excluded.run_id, source = excluded.source
                """
            ),
            {
                "lid": layer_id,
                "did": dataset_id,
                "type": type,
                "mid": model_id,
                "ver": version,
                "run": run_id,
                "src": source,
            },
        )

    def write_review(
        self,
        target_id: str,
        target_kind: str,
        verdict: str,
        dataset_id: Optional[str] = None,
        box: Optional[list[float]] = None,
        reviewer: Optional[str] = None,
    ) -> None:
        """Confidence-gated auto-accept writes its verdicts through here."""
        self.con.execute(
            text(
                """
                INSERT INTO reviews (target_id, target_kind, dataset_id, verdict, box, reviewer)
                VALUES (:t, :k, :d, :v, CAST(:b AS jsonb), :rev)
                ON CONFLICT (target_id) DO UPDATE
                SET verdict = excluded.verdict, box = excluded.box, dataset_id = excluded.dataset_id,
                    reviewer = excluded.reviewer, ts = now()
                """
            ),
            {
                "t": target_id,
                "k": target_kind,
                "d": dataset_id,
                "v": verdict,
                "b": json.dumps(box) if box is not None else None,
                "rev": reviewer,
            },
        )

    def update_job(self, job_id: str, **fields: Any) -> None:
        """Patch an ingest_jobs row (status/stage/progress/frames/error)."""
        if not fields:
            return
        allowed = {"status", "stage", "progress", "frames_done", "frames_total", "error", "finished_at"}
        sets = {k: v for k, v in fields.items() if k in allowed}
        if not sets:
            return
        assignments = ", ".join(f"{k} = :{k}" for k in sets)
        sets["job_id"] = job_id
        self.con.execute(text(f"UPDATE ingest_jobs SET {assignments} WHERE job_id = :job_id"), sets)

    def refresh_summary_view(self, concurrently: bool = False) -> None:
        kw = "CONCURRENTLY " if concurrently else ""
        self.con.exec_driver_sql(f"REFRESH MATERIALIZED VIEW {kw}v_image_summary_wide")


def sync_url(database_url: str) -> str:
    """Normalize a database URL to a synchronous driver.

    The API config stores the async URL (``postgresql+asyncpg://...``); the
    synchronous store needs ``postgresql+psycopg://...``. Plain
    ``postgresql://`` is left as-is (SQLAlchemy picks the default driver).
    """
    if "+asyncpg" in database_url:
        return database_url.replace("+asyncpg", "+psycopg")
    return database_url
