"""AsyncPgStore — async read/build store for the Veridian Studio API.

The FastAPI request handlers are I/O bound and run on the event loop, so they
use this asyncpg-backed store. The *write* path that drives the synchronous
Orchestrator uses :class:`sensveridian.store.pg.PgStore` instead. Both target
the same schema (``schema_pg.sql``).

The heavy lifting of fusing predictions with ground truth lives in
:mod:`sensveridian.api.fusion` (a pure, unit-tested module); this store is the
SQL around it.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..api import classmaps, fusion

_IMAGE_FILTERS = {
    "conflict": "rollup_conflicts > 0",
    "verified": "status = 'verified'",
    "flagged": "status = 'flagged'",
    "unreviewed": "status = 'unreviewed'",
}


def _loads(value: Any) -> Any:
    """jsonb may arrive as a Python object (typed columns) or a str (raw text()
    selects, depending on the asyncpg codec). Normalize to a Python object."""
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return None
    return value


class AsyncPgStore:
    def __init__(self, session: AsyncSession):
        self.s = session

    # ---- low-level reads ---------------------------------------------------
    async def get_dataset_row(self, dataset_id: str) -> Optional[dict]:
        res = await self.s.execute(
            text(
                "SELECT dataset_id AS id, name, descr AS desc, kind, models, palette, "
                "run_id AS \"runId\", mode, label_format, labels_dir, class_names, class_map "
                "FROM datasets WHERE dataset_id = :d"
            ),
            {"d": dataset_id},
        )
        r = res.mappings().first()
        if not r:
            return None
        row = dict(r)
        row["class_names"] = _loads(row.get("class_names"))
        row["class_map"] = _loads(row.get("class_map"))
        return row

    async def dataset_class_map(self, dataset_id: str) -> Optional[dict]:
        res = await self.s.execute(
            text("SELECT class_map FROM datasets WHERE dataset_id = :d"), {"d": dataset_id}
        )
        return _loads(res.scalar())

    async def set_class_map(self, dataset_id: str, class_map: dict) -> None:
        """Persist the per-dataset GT-label -> model-class alignment (UI editor)."""
        await self.s.execute(
            text("UPDATE datasets SET class_map = CAST(:m AS jsonb) WHERE dataset_id = :d"),
            {"m": json.dumps(class_map or {}), "d": dataset_id},
        )
        await self.s.commit()

    async def get_eval_metrics(self, dataset_id: str) -> list[dict]:
        """Stored Path-2 metrics for a dataset (one row per model+run)."""
        res = await self.s.execute(
            text(
                "SELECT dataset_id AS \"datasetId\", model_id AS model, run_id AS \"runId\", "
                "metrics, computed_at AS \"computedAt\" FROM eval_metrics "
                "WHERE dataset_id = :d ORDER BY model_id, run_id"
            ),
            {"d": dataset_id},
        )
        out: list[dict] = []
        for r in res.mappings().all():
            row = dict(r)
            row["metrics"] = _loads(row.get("metrics")) or {}
            row["computedAt"] = str(row["computedAt"]) if row.get("computedAt") else ""
            out.append(row)
        return out

    async def get_image_row(self, image_id: str) -> Optional[dict]:
        res = await self.s.execute(
            text("SELECT image_id, path, width, height, dataset_id, metadata FROM images WHERE image_id = :i"),
            {"i": image_id},
        )
        r = res.mappings().first()
        if not r:
            return None
        row = dict(r)
        row["metadata"] = _loads(row.get("metadata")) or {}
        return row

    async def resolve_dataset_id(self, target_id: str) -> Optional[str]:
        """Find the dataset a review target belongs to, server-side, so the UI
        need not pass it (it only round-trips the target id)."""
        if target_id.startswith("img:"):
            image_id = target_id[4:]
        elif target_id.startswith("seg_") or target_id.split(":", 1)[0] == "seg":
            res = await self.s.execute(
                text("SELECT c.dataset_id FROM segments s JOIN clips c ON c.clip_id = s.clip_id WHERE s.segment_id = :t"),
                {"t": target_id},
            )
            r = res.scalar()
            return r
        else:
            image_id = fusion.detection_image_id(target_id)
        if not image_id:
            return None
        res = await self.s.execute(text("SELECT dataset_id FROM images WHERE image_id = :i"), {"i": image_id})
        return res.scalar()

    async def _reviews_for_dataset(self, dataset_id: str) -> dict[str, dict]:
        res = await self.s.execute(
            text("SELECT target_id, verdict, box FROM reviews WHERE dataset_id = :d"),
            {"d": dataset_id},
        )
        out: dict[str, dict] = {}
        for r in res.mappings().all():
            out[r["target_id"]] = {"verdict": r["verdict"], "box": _loads(r["box"])}
        return out

    async def _gt_items_for_image(self, image_id: str) -> list[dict]:
        res = await self.s.execute(
            text("SELECT cls, box, meta FROM gt_boxes WHERE image_id = :i"),
            {"i": image_id},
        )
        items: list[dict] = []
        for r in res.mappings().all():
            meta = _loads(r["meta"]) or {}
            item = {"cls": r["cls"], "box": _loads(r["box"])}
            if "identity" in meta:
                item["identity"] = meta["identity"]
            if "decoded" in meta:
                item["decoded"] = meta["decoded"]
            items.append(item)
        return items

    async def _preds_by_model(self, image_id: str, run_id: str,
                              model_id: Optional[str] = None) -> dict[str, dict]:
        sql = "SELECT model_id, payload FROM predictions_raw WHERE image_id = :i AND run_id = :r"
        params = {"i": image_id, "r": run_id}
        if model_id:
            sql += " AND model_id = :m"
            params["m"] = model_id
        res = await self.s.execute(text(sql), params)
        return {row["model_id"]: (_loads(row["payload"]) or {}) for row in res.mappings().all()}

    # ---- detections (delegates to fusion) ----------------------------------
    async def build_image_detections(
        self, dataset_id: str, image_id: str, run_id: str = "baseline",
        model_id: Optional[str] = None,
    ) -> list[dict]:
        img = await self.get_image_row(image_id)
        if not img:
            return []
        preds = await self._preds_by_model(image_id, run_id, model_id)
        reviews = await self._reviews_for_dataset(dataset_id)
        gt_items = await self._gt_items_for_image(image_id)
        class_map = await self.dataset_class_map(dataset_id)
        return fusion.fuse_detections(
            image_id, preds, int(img["width"] or 0), int(img["height"] or 0),
            gt_items=gt_items, reviews=reviews, class_map=class_map,
        )

    # ---- Path 1: curation write-back --------------------------------------
    async def build_committed_gt(self, dataset_id: str, image_id: str,
                                 run_id: str = "baseline") -> Optional[dict]:
        """Derive the human-verified ground truth for one image from predictions
        + imported GT + review verdicts, in the dataset's own label vocabulary.

        Curation semantics (the human curates *predicted* boxes):
        - rejected prediction -> not GT; if it overlapped an imported GT box,
          that GT box is dropped too (the human overrode it);
        - edited prediction -> its corrected box becomes GT;
        - accepted / matched prediction -> becomes GT (keeping the imported GT
          label when it matches one, else the model class mapped back through the
          class-map);
        - imported GT boxes no prediction touched are kept verbatim.

        Returns ``{ds, path, boxes:[{cls, box}]}`` or None if the image/dataset
        is missing. Class alignment uses the inverse of the dataset class-map so
        boxes are written in the label file's vocabulary.
        """
        ds = await self.get_dataset_row(dataset_id)
        img = await self.get_image_row(image_id)
        if not ds or not img:
            return None
        preds_by_model = await self._preds_by_model(image_id, run_id)
        gt_items = await self._gt_items_for_image(image_id)
        reviews = await self._reviews_for_dataset(dataset_id)
        boxes = fusion.committed_gt(
            image_id, preds_by_model, int(img["width"] or 0), int(img["height"] or 0),
            gt_items=gt_items, reviews=reviews, class_map=ds.get("class_map") or {},
        )
        return {"ds": ds, "path": img["path"], "boxes": boxes}

    async def replace_image_gt(self, layer_id: str, dataset_id: str, image_id: str,
                               image_ref: str, anns: list[dict]) -> int:
        """Async twin of PgStore.replace_image_gt: replace one image's GT boxes
        in a layer (delete + insert). Boxes clamped to [0,1]. Caller commits."""
        await self.s.execute(
            text("DELETE FROM gt_boxes WHERE layer_id = :l AND image_id = :i"),
            {"l": layer_id, "i": image_id},
        )
        n = 0
        for a in anns:
            box = a.get("box") or [0, 0, 0, 0]
            x, y, w, h = (max(0.0, min(1.0, float(v))) for v in box[:4])
            await self.s.execute(
                text(
                    """
                    INSERT INTO gt_boxes (layer_id, dataset_id, image_id, image_ref, cls, box, meta)
                    VALUES (:l, :d, :i, :r, :c, CAST(:b AS jsonb), CAST(:m AS jsonb))
                    """
                ),
                {"l": layer_id, "d": dataset_id, "i": image_id, "r": image_ref,
                 "c": a.get("cls"), "b": json.dumps([x, y, w, h]),
                 "m": json.dumps(a.get("meta") or {})},
            )
            n += 1
        return n

    @staticmethod
    def image_src(dataset_id: str, image_id: str) -> str:
        return f"/api/v1/datasets/{dataset_id}/images/{image_id}/raw"

    async def get_image(self, dataset_id: str, image_id: str, run_id: str = "baseline") -> Optional[dict]:
        img = await self.get_image_row(image_id)
        if not img:
            return None
        meta = img["metadata"]
        objects = await self.build_image_detections(dataset_id, image_id, run_id)
        return {
            "id": image_id,
            "datasetId": dataset_id,
            "w": int(img["width"] or 0),
            "h": int(img["height"] or 0),
            "d0_ft": meta.get("d0_ft", 6.0),
            "augmented": bool(meta.get("augmented_flag", False)),
            "status": await self._image_status(dataset_id, image_id, meta),
            "captured": meta.get("captured"),
            "src": self.image_src(dataset_id, image_id),
            "objects": objects,
        }

    async def image_path(self, image_id: str) -> Optional[str]:
        res = await self.s.execute(text("SELECT path FROM images WHERE image_id = :i"), {"i": image_id})
        return res.scalar()

    async def get_dataset_grid(self, dataset_id: str, run_id: str = "baseline",
                               offset: int = 0, limit: int = 200) -> list[dict]:
        """Batched grid data: every image with its fused detections + rollup +
        raw-image src, in one response (the grid filters on objects[], the
        review queue aggregates them, and the canvas reads them from cache)."""
        rows = await self.s.execute(
            text(
                "SELECT image_id, width, height, metadata FROM images "
                "WHERE dataset_id = :d ORDER BY ingested_at OFFSET :off LIMIT :lim"
            ),
            {"d": dataset_id, "off": offset, "lim": limit},
        )
        reviews = await self._reviews_for_dataset(dataset_id)
        class_map = await self.dataset_class_map(dataset_id)
        out = []
        for r in rows.mappings().all():
            image_id = r["image_id"]
            meta = _loads(r["metadata"]) or {}
            preds = await self._preds_by_model(image_id, run_id)
            gt_items = await self._gt_items_for_image(image_id)
            objects = fusion.fuse_detections(
                image_id, preds, int(r["width"] or 0), int(r["height"] or 0),
                gt_items=gt_items, reviews=reviews, class_map=class_map,
            )
            roll = fusion.image_rollup(objects)
            out.append(
                {
                    "id": image_id,
                    "datasetId": dataset_id,
                    "w": int(r["width"] or 0),
                    "h": int(r["height"] or 0),
                    "d0_ft": meta.get("d0_ft", 6.0),
                    "augmented": bool(meta.get("augmented_flag", False)),
                    "status": await self._image_status(dataset_id, image_id, meta),
                    "captured": meta.get("captured"),
                    "src": self.image_src(dataset_id, image_id),
                    "agreement": roll["agreement"],
                    "conflicts": roll["conflicts"],
                    "objects": objects,
                }
            )
        return out

    async def review_queue(self, limit: int = 200) -> list[dict]:
        """Cross-dataset triage list: every prediction↔GT disagreement, lowest
        confidence first. Bounded so it does not fuse unbounded data — caps the
        images scanned per dataset and stops once enough rows are collected."""
        rows: list[dict] = []
        ds = await self.s.execute(
            text("SELECT dataset_id, name, run_id, kind FROM datasets ORDER BY created_at")
        )
        for d in ds.mappings().all():
            if d["kind"] == "audio":
                continue
            run_id = d["run_id"] or "baseline"
            imgs = await self.s.execute(
                text("SELECT image_id FROM images WHERE dataset_id = :d ORDER BY ingested_at LIMIT 500"),
                {"d": d["dataset_id"]},
            )
            for r in imgs.mappings().all():
                dets = await self.build_image_detections(d["dataset_id"], r["image_id"], run_id)
                for o in dets:
                    if o.get("state") != "match":
                        rows.append({
                            "datasetId": d["dataset_id"], "datasetName": d["name"],
                            "imageId": r["image_id"], "cls": o.get("cls"),
                            "state": o.get("state"), "conf": float(o.get("conf") or 0.0),
                            "iou": float(o.get("iou") or 0.0), "detId": o.get("id"),
                        })
                if len(rows) >= limit * 3:
                    break
        rows.sort(key=lambda x: x["conf"])
        return rows[:limit]

    async def _image_status(self, dataset_id: str, image_id: str, meta: dict) -> str:
        """Status from the frame-level review (img:<id>), else metadata, else
        unreviewed."""
        res = await self.s.execute(
            text("SELECT verdict FROM reviews WHERE target_id = :t"),
            {"t": f"img:{image_id}"},
        )
        r = res.mappings().first()
        if r:
            return "flagged" if r["verdict"] == "flagged" else "verified"
        return meta.get("status", "unreviewed")

    # ---- datasets ----------------------------------------------------------
    async def list_datasets(self) -> list[dict]:
        res = await self.s.execute(
            text(
                """
                SELECT d.dataset_id AS id, d.name, d.descr AS desc, d.kind, d.models,
                       d.run_id AS "runId"
                FROM datasets d ORDER BY d.created_at
                """
            )
        )
        out: list[dict] = []
        for r in res.mappings().all():
            ds = dict(r)
            ds["models"] = list(ds.get("models") or [])
            agg = await self.dataset_aggregates(ds["id"], ds["kind"], ds.get("runId") or "baseline")
            ds.update(agg)
            out.append(ds)
        return out

    async def dataset_aggregates(self, dataset_id: str, kind: str = "vision",
                                 run_id: str = "baseline") -> dict:
        """count / agreement / conflicts / reviewed, matching data.js.

        NOTE: exact agreement/conflicts require fusing each image. For datasets
        with no ground truth that is free (provisional => agreement 1.0). When GT
        exists this fuses per image; for very large datasets this should be
        memoized into a stats table / matview (TODO) rather than computed live.
        """
        if kind == "audio":
            return await self._audio_aggregates(dataset_id)

        count = await self._scalar("SELECT COUNT(*) FROM images WHERE dataset_id = :d", {"d": dataset_id})
        reviewed = await self._scalar(
            "SELECT COUNT(*) FROM reviews WHERE dataset_id = :d AND target_kind = 'image' AND verdict = 'accepted'",
            {"d": dataset_id},
        )
        has_gt = await self._scalar(
            "SELECT COUNT(*) FROM gt_boxes g JOIN images i ON i.image_id = g.image_id WHERE i.dataset_id = :d",
            {"d": dataset_id},
        )
        has_reviews = await self._scalar(
            "SELECT COUNT(*) FROM reviews WHERE dataset_id = :d AND target_kind = 'detection'",
            {"d": dataset_id},
        )
        if not has_gt and not has_reviews:
            # provisional: predictions are treated as ground truth -> full agreement
            return {"count": int(count), "agreement": 1.0 if count else 0.0, "conflicts": 0, "reviewed": int(reviewed)}

        rows = await self.s.execute(
            text("SELECT image_id FROM images WHERE dataset_id = :d"), {"d": dataset_id}
        )
        rollups = []
        for r in rows.mappings().all():
            objects = await self.build_image_detections(dataset_id, r["image_id"], run_id)
            rollups.append(fusion.image_rollup(objects))
        agg = fusion.dataset_rollup(rollups, reviewed=int(reviewed))
        agg["count"] = int(count)
        return agg

    async def _audio_aggregates(self, dataset_id: str) -> dict:
        count = await self._scalar("SELECT COUNT(*) FROM clips WHERE dataset_id = :d", {"d": dataset_id})
        reviewed = await self._scalar(
            "SELECT COUNT(*) FROM reviews WHERE dataset_id = :d AND target_kind IN ('clip','image') AND verdict = 'accepted'",
            {"d": dataset_id},
        )
        seg = await self.s.execute(
            text(
                "SELECT s.state FROM segments s JOIN clips c ON c.clip_id = s.clip_id WHERE c.dataset_id = :d"
            ),
            {"d": dataset_id},
        )
        states = [row["state"] for row in seg.mappings().all()]
        seg_n = max(1, len(states))
        matched = sum(1 for st in states if st == "match")
        conflicts = sum(1 for st in states if st and st != "match")
        return {
            "count": int(count),
            "agreement": round(matched / seg_n, 3),
            "conflicts": conflicts,
            "reviewed": int(reviewed),
        }

    async def list_dataset_images(
        self, dataset_id: str, offset: int = 0, limit: int = 200,
        filter: Optional[str] = None, sort: Optional[str] = None,
    ) -> list[dict]:
        """Lightweight image list for the grid (id + dims + status). Detection
        rollups would require fusion per row; the per-image endpoint provides
        the full objects[]."""
        order = "ORDER BY ingested_at"
        if sort == "index":
            order = "ORDER BY ingested_at"
        res = await self.s.execute(
            text(
                f"SELECT image_id AS id, width AS w, height AS h, metadata FROM images "
                f"WHERE dataset_id = :d {order} OFFSET :off LIMIT :lim"
            ),
            {"d": dataset_id, "off": offset, "lim": limit},
        )
        out = []
        for r in res.mappings().all():
            meta = _loads(r["metadata"]) or {}
            out.append(
                {
                    "id": r["id"],
                    "w": int(r["w"] or 0),
                    "h": int(r["h"] or 0),
                    "augmented": bool(meta.get("augmented_flag", False)),
                    "d0_ft": meta.get("d0_ft", 6.0),
                    "captured": meta.get("captured"),
                }
            )
        return out

    # ---- audio clips -------------------------------------------------------
    async def get_clip(self, dataset_id: str, clip_id: str) -> Optional[dict]:
        res = await self.s.execute(
            text("SELECT clip_id, name, duration_s, waveform FROM clips WHERE clip_id = :c AND dataset_id = :d"),
            {"c": clip_id, "d": dataset_id},
        )
        r = res.mappings().first()
        if not r:
            return None
        seg_res = await self.s.execute(
            text(
                "SELECT segment_id, start_frac, end_frac, gt_label, pred_label, conf, state, keyword "
                "FROM segments WHERE clip_id = :c ORDER BY start_frac"
            ),
            {"c": clip_id},
        )
        segments = [
            {
                "id": s["segment_id"],
                "start": float(s["start_frac"] or 0.0),
                "end": float(s["end_frac"] or 0.0),
                "gt": s["gt_label"],
                "pred": s["pred_label"],
                "conf": float(s["conf"] or 0.0),
                "state": s["state"] or "match",
                "keyword": s["keyword"],
            }
            for s in seg_res.mappings().all()
        ]
        return {
            "id": r["clip_id"],
            "name": r["name"],
            "dur": float(r["duration_s"] or 0.0),
            "wave": _loads(r["waveform"]) or [],
            "segments": segments,
        }

    # ---- models ------------------------------------------------------------
    async def list_models(self) -> list[dict]:
        res = await self.s.execute(
            text(
                "SELECT model_id, display_name, version, weights_path, weights_sha, "
                "input_spec, n_classes, depends_on FROM models ORDER BY model_id"
            )
        )
        models = []
        for r in res.mappings().all():
            mid = r["model_id"]
            card = classmaps.model_card(mid)
            versions = await self._model_versions(mid, r["version"], r["weights_sha"])
            models.append(
                {
                    "id": mid,
                    "display_name": r["display_name"] or card["display_name"],
                    "short": classmaps.short_name(mid),
                    "input": r["input_spec"] or card["input"],
                    "weights_path": r["weights_path"] or "",
                    "classes": r["n_classes"] if r["n_classes"] is not None else card["classes"],
                    "depends_on": r["depends_on"] if r["depends_on"] is not None else card["depends_on"],
                    "versions": versions,
                }
            )
        return models

    async def _model_versions(self, model_id: str, fallback_version: str,
                              fallback_sha: str) -> list[dict]:
        res = await self.s.execute(
            text(
                "SELECT version, weights_sha, released_on, metrics, notes, is_current "
                "FROM model_versions WHERE model_id = :m ORDER BY released_on DESC NULLS LAST, version DESC"
            ),
            {"m": model_id},
        )
        rows = res.mappings().all()
        if not rows:
            # synthesize a single current version from the models row
            return [
                {
                    "version": fallback_version or "0",
                    "weights_sha": fallback_sha or "",
                    "date": "",
                    "metrics": {},
                    "notes": "current",
                    "current": True,
                }
            ]
        out = []
        for r in rows:
            out.append(
                {
                    "version": r["version"],
                    "weights_sha": r["weights_sha"] or "",
                    "date": str(r["released_on"]) if r["released_on"] else "",
                    "metrics": _loads(r["metrics"]) or {},
                    "notes": r["notes"] or "",
                    "current": bool(r["is_current"]),
                }
            )
        return out

    async def promote_version(self, model_id: str, version: str) -> None:
        await self.s.execute(
            text("UPDATE model_versions SET is_current = false WHERE model_id = :m"), {"m": model_id}
        )
        await self.s.execute(
            text("UPDATE model_versions SET is_current = true WHERE model_id = :m AND version = :v"),
            {"m": model_id, "v": version},
        )
        await self.s.commit()

    async def regressions(self, model_id: str, base_run: str, candidate_run: str) -> list[dict]:
        """Detections whose correctness flipped between two runs of ``model_id``,
        scored against the GT layer. ``regress=True`` = was correct, now wrong.

        ``base``/``candidate`` are treated as run ids (a layer == a run). When
        the UI passes version strings, resolve version->run upstream; here we
        assume the run is named by the version (TODO: a model_versions.run_id
        column would make this explicit).
        """
        img_res = await self.s.execute(
            text(
                """
                SELECT DISTINCT i.image_id, i.dataset_id
                FROM predictions_summary a
                JOIN predictions_summary b
                  ON a.image_id = b.image_id AND a.model_id = b.model_id
                JOIN images i ON i.image_id = a.image_id
                WHERE a.model_id = :m AND a.run_id = :base AND b.run_id = :cand
                """
            ),
            {"m": model_id, "base": base_run, "cand": candidate_run},
        )
        flips: list[dict] = []
        for row in img_res.mappings().all():
            image_id, dataset_id = row["image_id"], row["dataset_id"]
            base_dets = {d["id"].rsplit("_", 1)[-1]: d
                         for d in await self.build_image_detections(dataset_id, image_id, base_run, model_id)}
            cand_dets = await self.build_image_detections(dataset_id, image_id, candidate_run, model_id)
            for cd in cand_dets:
                key = cd["id"].rsplit("_", 1)[-1]
                bd = base_dets.get(key)
                if not bd:
                    continue
                base_ok = bd["state"] == "match"
                cand_ok = cd["state"] == "match"
                if base_ok == cand_ok:
                    continue
                flips.append(
                    {
                        "datasetId": dataset_id,
                        "imageId": image_id,
                        "cls": cd["cls"],
                        "regress": base_ok and not cand_ok,
                        "confA": round(float(bd["conf"]), 3),
                        "confB": round(float(cd["conf"]), 3),
                        "detId": cd["id"],
                    }
                )
        return flips

    # ---- reviews (writes) --------------------------------------------------
    async def save_review(self, target_id: str, target_kind: str, verdict: str,
                          dataset_id: Optional[str], box: Optional[list[float]]) -> None:
        await self.s.execute(
            text(
                """
                INSERT INTO reviews (target_id, target_kind, dataset_id, verdict, box, ts)
                VALUES (:t, :k, :d, :v, CAST(:b AS jsonb), now())
                ON CONFLICT (target_id) DO UPDATE
                SET verdict = excluded.verdict, box = excluded.box,
                    dataset_id = COALESCE(excluded.dataset_id, reviews.dataset_id), ts = now()
                """
            ),
            {"t": target_id, "k": target_kind, "d": dataset_id, "v": verdict,
             "b": json.dumps(box) if box is not None else None},
        )
        await self.s.commit()

    async def bulk_review(self, target_ids: list[str], target_kinds: list[str],
                          verdict: str, dataset_id: Optional[str]) -> int:
        for tid, kind in zip(target_ids, target_kinds):
            await self.s.execute(
                text(
                    """
                    INSERT INTO reviews (target_id, target_kind, dataset_id, verdict, ts)
                    VALUES (:t, :k, :d, :v, now())
                    ON CONFLICT (target_id) DO UPDATE
                    SET verdict = excluded.verdict,
                        dataset_id = COALESCE(excluded.dataset_id, reviews.dataset_id), ts = now()
                    """
                ),
                {"t": tid, "k": kind, "d": dataset_id, "v": verdict},
            )
        await self.s.commit()
        return len(target_ids)

    # ---- layers ------------------------------------------------------------
    async def list_layers(self, dataset_id: str) -> list[dict]:
        res = await self.s.execute(
            text(
                "SELECT layer_id AS id, type, model_id AS model, version, run_id AS \"runId\", "
                "source, created_at AS \"createdAt\" FROM layers WHERE dataset_id = :d ORDER BY created_at"
            ),
            {"d": dataset_id},
        )
        return [dict(r) for r in res.mappings().all()]

    # ---- import / create writes -------------------------------------------
    async def ensure_dataset(self, dataset_id: str, name: str, kind: str = "vision",
                             descr: str = "", models: Optional[list[str]] = None,
                             palette: Optional[str] = None, run_id: Optional[str] = None) -> None:
        await self.s.execute(
            text(
                """
                INSERT INTO datasets (dataset_id, name, descr, kind, models, palette, run_id)
                VALUES (:id, :name, :descr, :kind, :models, :palette, :run_id)
                ON CONFLICT (dataset_id) DO UPDATE
                SET name = excluded.name, descr = excluded.descr, kind = excluded.kind,
                    models = excluded.models, palette = excluded.palette,
                    run_id = COALESCE(excluded.run_id, datasets.run_id)
                """
            ),
            {"id": dataset_id, "name": name, "descr": descr, "kind": kind,
             "models": models or [], "palette": palette, "run_id": run_id},
        )

    async def upsert_layer(self, layer_id: str, dataset_id: str, type: str,
                           model_id: Optional[str] = None, version: Optional[str] = None,
                           run_id: Optional[str] = None, source: Optional[str] = None) -> None:
        await self.s.execute(
            text(
                """
                INSERT INTO layers (layer_id, dataset_id, type, model_id, version, run_id, source)
                VALUES (:lid, :did, :type, :mid, :ver, :run, :src)
                ON CONFLICT (layer_id) DO UPDATE
                SET type = excluded.type, model_id = excluded.model_id, version = excluded.version,
                    run_id = excluded.run_id, source = excluded.source
                """
            ),
            {"lid": layer_id, "did": dataset_id, "type": type, "mid": model_id,
             "ver": version, "run": run_id, "src": source},
        )

    async def insert_gt_boxes(self, layer_id: str, dataset_id: str, annotations: list[dict]) -> int:
        """Insert imported ground-truth boxes. ``image_id`` is left for the
        label<->frame matching step; ``image_ref`` carries the original key.
        Boxes are clamped to [0,1] on insert."""
        n = 0
        for a in annotations:
            box = a.get("box") or [0, 0, 0, 0]
            x, y, w, h = (max(0.0, min(1.0, float(v))) for v in box[:4])
            meta = {k: a[k] for k in ("identity", "decoded", "img_w", "img_h") if k in a}
            await self.s.execute(
                text(
                    """
                    INSERT INTO gt_boxes (layer_id, dataset_id, image_id, image_ref, cls, box, meta)
                    VALUES (:lid, :did, :iid, :iref, :cls, CAST(:box AS jsonb), CAST(:meta AS jsonb))
                    """
                ),
                {"lid": layer_id, "did": dataset_id, "iid": a.get("image_id"),
                 "iref": a.get("image"), "cls": a.get("cls"),
                 "box": json.dumps([x, y, w, h]), "meta": json.dumps(meta)},
            )
            n += 1
        return n

    async def create_ingest_job(self, job_id: str, source: str, spec: dict, frames_total: int) -> None:
        await self.s.execute(
            text(
                """
                INSERT INTO ingest_jobs (job_id, source, status, spec, frames_total)
                VALUES (:j, :src, 'queued', CAST(:spec AS jsonb), :ft)
                """
            ),
            {"j": job_id, "src": source, "spec": json.dumps(spec), "ft": frames_total},
        )

    async def get_ingest_job(self, job_id: str) -> Optional[dict]:
        res = await self.s.execute(
            text(
                "SELECT job_id AS \"jobId\", source, status, stage, progress, "
                "frames_done AS \"framesDone\", frames_total AS \"framesTotal\", error "
                "FROM ingest_jobs WHERE job_id = :j"
            ),
            {"j": job_id},
        )
        r = res.mappings().first()
        return dict(r) if r else None

    async def create_connection(self, connection_id: str, uri: str, schedule: str, pipeline: str) -> None:
        await self.s.execute(
            text(
                "INSERT INTO connections (connection_id, uri, schedule, pipeline) "
                "VALUES (:c, :u, :s, :p)"
            ),
            {"c": connection_id, "u": uri, "s": schedule, "p": pipeline},
        )

    async def commit(self) -> None:
        await self.s.commit()

    # ---- lineage -----------------------------------------------------------
    async def lineage(self) -> dict:
        """Assemble the content-addressed provenance DAG from connections,
        ingest sources, datasets, runs, reviews, augmentations and exports."""
        nodes: list[dict] = []
        edges: list[list[str]] = []
        seen: set[str] = set()

        def add(node_id: str, ntype: str, label: str, meta: str = "") -> str:
            if node_id not in seen:
                nodes.append({"id": node_id, "type": ntype, "label": label, "meta": meta})
                seen.add(node_id)
            return node_id

        # watched sources
        for r in (await self.s.execute(text("SELECT connection_id, uri FROM connections"))).mappings().all():
            add(f"src:{r['connection_id']}", "source", r["uri"], "connection")

        # datasets
        ds_rows = (await self.s.execute(text("SELECT dataset_id, name, kind, run_id FROM datasets"))).mappings().all()
        for d in ds_rows:
            dnode = add(f"d:{d['dataset_id']}", "dataset", d["name"], d["kind"])
            # runs feeding this dataset
            runs = (
                await self.s.execute(
                    text(
                        "SELECT DISTINCT s.run_id, s.model_id FROM predictions_summary s "
                        "JOIN images i ON i.image_id = s.image_id WHERE i.dataset_id = :d"
                    ),
                    {"d": d["dataset_id"]},
                )
            ).mappings().all()
            for run in runs:
                rnode = add(f"r:{run['run_id']}:{run['model_id']}", "run",
                            f"{run['model_id']} @ {run['run_id']}", "prediction")
                edges.append([dnode, rnode])
            # verified?
            n_rev = await self._scalar(
                "SELECT COUNT(*) FROM reviews WHERE dataset_id = :d AND verdict = 'accepted'",
                {"d": d["dataset_id"]},
            )
            if n_rev:
                vnode = add(f"v:{d['dataset_id']}", "verified", f"{d['name']} (verified)", f"{n_rev} accepted")
                edges.append([dnode, vnode])

        # ingest-job sources -> datasets
        for j in (await self.s.execute(text("SELECT job_id, source, spec FROM ingest_jobs"))).mappings().all():
            spec = _loads(j["spec"]) or {}
            snode = add(f"job:{j['job_id']}", "source", j["source"] or "ingest", "ingest job")
            for g in spec.get("groups", []):
                ds = g.get("dataset")
                if ds and f"d:{ds}" in seen:
                    edges.append([snode, f"d:{ds}"])

        # augmentations (collapsed parent->child dataset membership)
        for a in (
            await self.s.execute(
                text(
                    "SELECT a.method, ip.dataset_id AS parent_ds, ic.dataset_id AS child_ds "
                    "FROM augmentations a "
                    "JOIN images ip ON ip.image_id = a.parent_image_id "
                    "JOIN images ic ON ic.image_id = a.augmented_image_id"
                )
            )
        ).mappings().all():
            if a["parent_ds"] and a["child_ds"] and a["parent_ds"] != a["child_ds"]:
                anode = add(f"aug:{a['parent_ds']}:{a['child_ds']}", "aug", a["method"], "augmentation")
                if f"d:{a['parent_ds']}" in seen:
                    edges.append([f"d:{a['parent_ds']}", anode])
                if f"d:{a['child_ds']}" in seen:
                    edges.append([anode, f"d:{a['child_ds']}"])

        # exports
        for e in (await self.s.execute(text("SELECT export_id, name, dataset_id FROM exports"))).mappings().all():
            enode = add(f"exp:{e['export_id']}", "export", e["name"] or e["export_id"], "export")
            if e["dataset_id"] and f"d:{e['dataset_id']}" in seen:
                edges.append([f"d:{e['dataset_id']}", enode])

        return {"nodes": nodes, "edges": edges}

    # ---- seeding -----------------------------------------------------------
    async def seed_models(self) -> None:
        """Seed the models + a current model_versions row from the static model
        cards + sensveridian.config.MODELS, so /models and the ingest model
        picker work before any ingest. Idempotent."""
        from ..api import classmaps as cm
        from ..config import MODELS

        weights = {
            "amod": str(MODELS.amod),
            "qrcode": str(MODELS.qrcode),
            "fd": str(MODELS.fd),
            "fr": str(MODELS.fr),
            "aed": "",
        }
        for mid in ("amod", "qrcode", "fd", "fr", "aed"):
            card = cm.model_card(mid)
            await self.s.execute(
                text(
                    """
                    INSERT INTO models (model_id, display_name, version, weights_path, weights_sha, input_spec, n_classes, depends_on)
                    VALUES (:m, :d, :v, :p, '', :ispec, :nc, :dep)
                    ON CONFLICT (model_id) DO UPDATE
                    SET display_name = excluded.display_name, version = excluded.version,
                        weights_path = excluded.weights_path, input_spec = excluded.input_spec,
                        n_classes = excluded.n_classes, depends_on = excluded.depends_on
                    """
                ),
                {"m": mid, "d": card["display_name"], "v": card["version"], "p": weights[mid],
                 "ispec": card["input"], "nc": card["classes"], "dep": card["depends_on"]},
            )
            await self.s.execute(
                text(
                    """
                    INSERT INTO model_versions (model_id, version, weights_sha, metrics, notes, is_current)
                    VALUES (:m, :v, '', '{}'::jsonb, 'seeded', true)
                    ON CONFLICT (model_id, version) DO NOTHING
                    """
                ),
                {"m": mid, "v": card["version"]},
            )
        await self.s.commit()

    # ---- helpers -----------------------------------------------------------
    async def _scalar(self, sql: str, params: dict) -> Any:
        res = await self.s.execute(text(sql), params)
        return res.scalar() or 0
