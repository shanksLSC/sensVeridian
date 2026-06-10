"""Datasets, images, clips, predictions, layers, import."""
from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from sqlalchemy import text

from .. import importers, lint
from ..config import settings
from ..deps import get_store, require_auth
from ..schemas import Clip, DatasetSummary, Detection, Image, ImportResult, ImportSpec
from ...store.pg_async import AsyncPgStore

router = APIRouter(dependencies=[Depends(require_auth)])


def _roots() -> list[Path]:
    return [Path(settings.datasets_root).resolve(), Path(settings.frames_root).resolve()]


def _under_roots(path: str) -> Path | None:
    """Resolve a path and confirm it lives under an allowed root (datasets_root
    or frames_root). Does not require the path to exist (used for label dirs)."""
    if not path:
        return None
    p = Path(path).resolve()
    return p if any(p == r or r in p.parents for r in _roots()) else None


def _allowed_image_path(path: str) -> Path | None:
    """Resolve a stored image path and confirm it lives under an allowed root
    (datasets_root or frames_root) before serving its bytes."""
    p = _under_roots(path)
    return p if (p and p.is_file()) else None


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return (s or "dataset")[:32]


@router.get("/datasets", response_model=list[DatasetSummary])
async def list_datasets(store: AsyncPgStore = Depends(get_store)):
    return await store.list_datasets()


@router.get("/queue")
async def review_queue(limit: int = Query(200, ge=1, le=1000), store: AsyncPgStore = Depends(get_store)):
    """Cross-dataset review queue: prediction↔GT disagreements, lowest conf first.
    Server-side aggregation so the UI need not hydrate every dataset."""
    return await store.review_queue(limit=limit)


@router.get("/datasets/{dataset_id}")
async def get_dataset(
    dataset_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=1000),
    store: AsyncPgStore = Depends(get_store),
):
    ds = await store.get_dataset_row(dataset_id)
    if not ds:
        raise HTTPException(404, "dataset not found")
    ds["models"] = list(ds.get("models") or [])
    run_id = ds.get("runId") or "baseline"
    ds.update(await store.dataset_aggregates(dataset_id, ds["kind"], run_id))
    if ds["kind"] == "audio":
        ds["clips"] = await _list_clips(store, dataset_id)
    else:
        # batched grid: images with fused detections + rollups + raw-image src
        ds["images"] = await store.get_dataset_grid(dataset_id, run_id, offset=offset, limit=limit)
    return ds


async def _list_clips(store: AsyncPgStore, dataset_id: str) -> list[dict]:
    res = await store.s.execute(
        text("SELECT clip_id AS id, name, duration_s AS dur FROM clips WHERE dataset_id = :d ORDER BY name"),
        {"d": dataset_id},
    )
    return [dict(r) for r in res.mappings().all()]


@router.get("/datasets/{dataset_id}/images")
async def list_images(
    dataset_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=1000),
    filter: str | None = None,
    sort: str | None = None,
    store: AsyncPgStore = Depends(get_store),
):
    return await store.list_dataset_images(dataset_id, offset=offset, limit=limit, filter=filter, sort=sort)


@router.get("/datasets/{dataset_id}/images/{image_id}/raw")
async def get_image_raw(dataset_id: str, image_id: str, store: AsyncPgStore = Depends(get_store)):
    """Serve the raw image bytes from its on-disk path (content-addressed by
    image_id; path validated to live under the datasets/frames roots)."""
    path = await store.image_path(image_id)
    safe = _allowed_image_path(path or "")
    if not safe:
        raise HTTPException(404, "image bytes not found")
    return FileResponse(str(safe))


@router.get("/datasets/{dataset_id}/images/{image_id}", response_model=Image)
async def get_image(dataset_id: str, image_id: str, run_id: str = "baseline",
                    store: AsyncPgStore = Depends(get_store)):
    img = await store.get_image(dataset_id, image_id, run_id)
    if not img:
        raise HTTPException(404, "image not found")
    return Image(**img)


@router.get("/datasets/{dataset_id}/images/{image_id}/predictions", response_model=list[Detection])
async def get_predictions(dataset_id: str, image_id: str,
                          run_id: str = Query("baseline"), model_id: str | None = None,
                          store: AsyncPgStore = Depends(get_store)):
    objs = await store.build_image_detections(dataset_id, image_id, run_id, model_id)
    return [Detection(**o) for o in objs]


@router.get("/datasets/{dataset_id}/clips/{clip_id}", response_model=Clip)
async def get_clip(dataset_id: str, clip_id: str, store: AsyncPgStore = Depends(get_store)):
    clip = await store.get_clip(dataset_id, clip_id)
    if not clip:
        raise HTTPException(404, "clip not found")
    return Clip(**clip)


@router.get("/datasets/{dataset_id}/layers")
async def get_layers(dataset_id: str, store: AsyncPgStore = Depends(get_store)):
    return await store.list_layers(dataset_id)


# ---- Path 2: evaluation metrics + class-map ---------------------------------
@router.get("/datasets/{dataset_id}/metrics")
async def get_metrics(dataset_id: str, store: AsyncPgStore = Depends(get_store)):
    """Stored predicted-vs-GT metrics for the dataset (one entry per model+run),
    plus the dataset's mode and class-map so the UI can render the eval panel."""
    ds = await store.get_dataset_row(dataset_id)
    if not ds:
        raise HTTPException(404, "dataset not found")
    return {
        "datasetId": dataset_id,
        "mode": ds.get("mode"),
        "classMap": ds.get("class_map") or {},
        "classNames": ds.get("class_names") or {},
        "models": list(ds.get("models") or []),
        "metrics": await store.get_eval_metrics(dataset_id),
    }


@router.post("/datasets/{dataset_id}/metrics:compute")
async def compute_metrics(
    dataset_id: str,
    run_id: str = Query("baseline"),
    model_id: str | None = Query(None),
    store: AsyncPgStore = Depends(get_store),
):
    """Recompute metrics on demand for one model (``model_id``) or every model
    the dataset declares. The compute is synchronous (decodes payloads, fuses
    against mapped GT), so it runs in a worker thread off the event loop."""
    ds = await store.get_dataset_row(dataset_id)
    if not ds:
        raise HTTPException(404, "dataset not found")
    models = [model_id] if model_id else list(ds.get("models") or [])
    if not models:
        raise HTTPException(400, "dataset declares no models to evaluate")

    def _run() -> list[dict]:
        from ...store.pg import PgStore

        s = PgStore(settings.database_url)
        try:
            return [
                {"model": m, "runId": run_id, "metrics": s.compute_eval_metrics(dataset_id, m, run_id)}
                for m in models
            ]
        finally:
            s.close()

    results = await run_in_threadpool(_run)
    return {"datasetId": dataset_id, "results": results}


@router.put("/datasets/{dataset_id}/class-map")
async def put_class_map(
    dataset_id: str,
    class_map: dict = Body(...),
    store: AsyncPgStore = Depends(get_store),
):
    """Replace the per-dataset GT-label -> model-class alignment used by fusion
    and metrics. Pass ``{}`` to clear it (then every GT label is compared as-is)."""
    ds = await store.get_dataset_row(dataset_id)
    if not ds:
        raise HTTPException(404, "dataset not found")
    cleaned = {str(k): str(v) for k, v in (class_map or {}).items() if v}
    await store.set_class_map(dataset_id, cleaned)
    return {"ok": True, "datasetId": dataset_id, "classMap": cleaned}


# ---- Path 1: curation label write-back --------------------------------------
@router.post("/datasets/{dataset_id}/images/{image_id}/commit-labels")
async def commit_labels(
    dataset_id: str,
    image_id: str,
    run_id: str = Query("baseline"),
    store: AsyncPgStore = Depends(get_store),
):
    """Persist the human-verified ground truth for one image (curate mode only):
    recompute the verified boxes from predictions + reviews, **overwrite** the
    native label file in place (keeping a one-time ``.bak``), update ``gt_boxes``,
    and mark the frame verified. Eval-mode datasets are read-only and rejected."""
    ds = await store.get_dataset_row(dataset_id)
    if not ds:
        raise HTTPException(404, "dataset not found")
    if (ds.get("mode") or "eval") != "curate":
        raise HTTPException(409, "commit-labels is only allowed for curate-mode datasets")
    labels_dir = ds.get("labels_dir")
    if not labels_dir:
        raise HTTPException(400, "dataset has no labels directory to write to")
    safe_labels = _under_roots(labels_dir)
    if not safe_labels:
        raise HTTPException(400, "labels directory is not under an allowed root")

    built = await store.build_committed_gt(dataset_id, image_id, run_id)
    if not built:
        raise HTTPException(404, "image not found")
    safe_img = _allowed_image_path(built["path"] or "")
    if not safe_img:
        raise HTTPException(400, "image path is not under an allowed root")

    boxes = built["boxes"]
    fmt = ds.get("label_format") or "yolo"
    class_names = {int(k): v for k, v in (ds.get("class_names") or {}).items()}

    from ...ingest import label_io

    def _write() -> str:
        return str(label_io.write_image_labels(safe_img, safe_labels, fmt, class_names, boxes))

    label_file = await run_in_threadpool(_write)

    # reflect the committed GT in the DB and mark the frame verified
    gt_layer = f"gt:{dataset_id}"
    await store.upsert_layer(gt_layer, dataset_id, type="ground-truth", source=f"labels:{fmt}")
    await store.replace_image_gt(gt_layer, dataset_id, image_id, safe_img.name, boxes)
    await store.save_review(f"img:{image_id}", "image", "accepted", dataset_id, None)
    await store.commit()

    return {"ok": True, "datasetId": dataset_id, "imageId": image_id,
            "labelFile": label_file, "n": len(boxes), "boxes": boxes}


@router.post("/datasets:import", response_model=ImportResult)
async def import_labels(spec: ImportSpec, store: AsyncPgStore = Depends(get_store)):
    """Create a dataset from an imported label set: parse (when inline
    annotations/segments are supplied), run the label-lint health check, persist
    a ground-truth layer, and optionally queue a compare run.

    The UI sends ``format`` + ``mapping``; the actual label bytes arrive via the
    presigned-upload flow and are parsed server-side into ``spec.annotations`` /
    ``spec.segments`` before this handler runs. Tests/automation may pass them
    inline.
    """
    dataset_id = _slug(spec.name)
    annotations = spec.annotations
    # If raw labels were uploaded and not yet parsed, parse here when possible.
    if annotations is None and spec.format and spec.mapping and "raw" in (spec.mapping or {}):
        annotations = importers.parse_labels(spec.format, spec.mapping["raw"], spec.mapping)

    report = lint.run_checks(
        spec, annotations=annotations, image_refs=spec.image_refs, segments=spec.segments
    )

    await store.ensure_dataset(
        dataset_id, name=spec.name, kind=spec.kind, descr=spec.desc,
        models=spec.models, palette=spec.palette,
        run_id=spec.compareRun,
    )
    layer_id = f"gt:{dataset_id}"
    await store.upsert_layer(
        layer_id, dataset_id, type="ground-truth",
        source=f"import:{spec.format}" if spec.format else "import",
    )
    if annotations:
        # errors (sev='error') are auto-skipped: drop unmatched if image_refs known
        keep = annotations
        if spec.image_refs is not None:
            refs = set(spec.image_refs)
            keep = [a for a in annotations if a.get("image") in refs]
        await store.insert_gt_boxes(layer_id, dataset_id, keep)
    await store.commit()

    return ImportResult(ok=True, datasetId=dataset_id, score=report["score"],
                        checks=report["checks"])
