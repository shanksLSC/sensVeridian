"""Ingest worker — frame-sample + auto-label + dedup + upsert + combine.

Reuses the EXISTING sensVeridian pieces unchanged: the dataset-generator's
OpenCV target-fps frame sampler (``frames.py``), ``Orchestrator.ingest`` for the
oracle pipeline, and content-addressed hashing for exact dedup. The Orchestrator
is synchronous, so the heavy work runs in a worker thread against the
synchronous :class:`sensveridian.store.pg.PgStore`; progress is streamed back to
the API's WebSocket over an in-process bus (dev) or Redis pub/sub (prod, arq).

Run the queue worker (prod):  ``arq sensveridian.ingest.worker.WorkerSettings``
Dev: ``POST /ingest/jobs`` runs the pipeline in-process (no Redis required).
"""
from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import AsyncIterator, Callable, Optional

from ..api.config import settings
from .frames import discover_images, discover_videos, sample_video

STAGES = ["Decoding", "Sampling frames", "Auto-labelling", "Hash + dedup", "Writing to Postgres"]


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")
    return (s or "dataset")[:32]


# ---- progress pub/sub ------------------------------------------------------
# Dev default: an in-process bus shared by the worker thread and the WS handler.
# Prod (arq, separate processes): publish to Redis and have the API subscribe —
# see redis_progress_stream() (SEAM).
_LOCAL_BUS: dict[str, asyncio.Queue] = {}


def _bus(job_id: str) -> asyncio.Queue:
    return _LOCAL_BUS.setdefault(job_id, asyncio.Queue())


def _thread_emitter(job_id: str, loop: asyncio.AbstractEventLoop) -> Callable[[dict], None]:
    """A thread-safe emit() the synchronous pipeline calls; hands frames back to
    the event loop's per-job queue."""
    q = _bus(job_id)

    def emit(frame: dict) -> None:
        loop.call_soon_threadsafe(q.put_nowait, frame)

    return emit


async def job_progress_stream(job_id: str) -> AsyncIterator[dict]:
    """Consumed by the WebSocket handler in api.main (dev / in-process)."""
    q = _bus(job_id)
    while True:
        frame = await q.get()
        yield frame
        if frame.get("status") in ("done", "error"):
            return


# ---- enqueue ---------------------------------------------------------------
async def enqueue_ingest(job_id: str, spec: dict) -> None:
    """Push the job onto the worker queue.

    With arq configured (``VERIDIAN_USE_ARQ=1``) this enqueues ``run_ingest``;
    otherwise (dev) it runs the synchronous pipeline in a background thread and
    streams progress to the in-process bus so the WebSocket works without Redis.
    """
    if _arq_enabled():
        from arq import create_pool
        from arq.connections import RedisSettings

        pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
        await pool.enqueue_job("run_ingest", job_id, spec)
        return

    loop = asyncio.get_running_loop()
    emit = _thread_emitter(job_id, loop)
    asyncio.create_task(asyncio.to_thread(run_ingest_sync, job_id, spec, emit))


def _arq_enabled() -> bool:
    import os

    return os.getenv("VERIDIAN_USE_ARQ", "").lower() in ("1", "true", "yes")


# ---- the job (synchronous; runs the real Orchestrator) ---------------------
def _resolve_source_dir(group: dict, source: str) -> Path:
    """Resolve a group's source directory, guarding local paths against escapes
    of the datasets root."""
    if source == "local" and group.get("path"):
        root = Path(settings.datasets_root).resolve()
        base = (root / group["path"]).resolve()
        if base != root and root not in base.parents:
            raise ValueError(f"path escapes datasets root: {group.get('path')!r}")
        return base
    # video-upload staging convention: {media_root}/{tag}/
    return Path(settings.media_root) / (group.get("tag") or "")


def _plan_group(group: dict, source: str) -> dict:
    """Decide what to process for a group: sampled video frames or a capped list
    of images."""
    base = _resolve_source_dir(group, source)
    kind = group.get("kind", "auto")
    max_images = int(group.get("maxImages") or settings.max_ingest_images)
    videos = discover_videos(base) if kind in ("auto", "video") else []
    if videos and kind != "image":
        return {"base": base, "kind": "video", "videos": videos, "images": [], "max_images": max_images}
    images = discover_images(base, limit=max_images) if kind in ("auto", "image") else []
    return {"base": base, "kind": "image", "videos": [], "images": images, "max_images": max_images}


def run_ingest_sync(job_id: str, spec: dict, emit: Callable[[dict], None]) -> None:
    """The actual pipeline. Synchronous on purpose: the Orchestrator and the
    sync PgStore do blocking CPU/GPU work and must stay off the event loop.

    Handles both local sources: a folder of **images** (cap to maxImages via
    symlink staging so model weights hash once) and **videos** (the existing
    OpenCV target-fps sampler). Predictions land in predictions_*; with a
    trustThreshold, confident detections are auto-accepted into the GT layer.
    """
    from ..hashing import hash_decoded_image
    from ..orchestrator import Orchestrator
    from ..store.faces_registry import FaceRegistry
    from ..store.pg import PgStore

    source = spec.get("source", "local")
    done = 0
    store: Optional[PgStore] = None

    # Plan first so framesTotal/progress are meaningful.
    plans = []
    for group in spec.get("groups", []):
        plan = _plan_group(group, source)
        plan["group"] = group
        plan["dataset_id"] = group.get("dataset") or _slug(group.get("label") or group.get("tag"))
        plan["models"] = set(group.get("models") or [])
        plans.append(plan)
    total = sum(
        len(p["images"]) if p["kind"] == "image" else int(p["group"].get("frames", 0))
        for p in plans
    ) or 0

    def progress(stage: str, status: str = "running") -> None:
        pct = int(100 * done / total) if total else (100 if status == "done" else 0)
        emit({"jobId": job_id, "stage": stage, "progress": pct,
              "framesDone": done, "framesTotal": total, "status": status})

    try:
        store = PgStore(settings.database_url)
        store.update_job(job_id, status="running", stage=STAGES[0], frames_total=total)
        registry = FaceRegistry(settings.redis_url)
        orch = Orchestrator(store, registry)

        for plan in plans:
            group, dataset_id, models = plan["group"], plan["dataset_id"], plan["models"]
            run_id = "baseline"
            if group.get("isNew"):
                store.ensure_dataset(dataset_id, group.get("label") or dataset_id,
                                     models=sorted(models), run_id=run_id)
            progress(STAGES[0])  # Decoding

            frames_dir = Path(settings.frames_root) / job_id / dataset_id
            frames_dir.mkdir(parents=True, exist_ok=True)
            process_paths: list[Path] = []

            if plan["kind"] == "video":
                for v in plan["videos"]:
                    process_paths.extend(
                        sample_video(v, frames_dir, settings.sample_fps,
                                     dedup_stride=settings.dedup_stride,
                                     jpeg_quality=settings.jpeg_quality)
                    )
                progress(STAGES[1])  # Sampling frames
                if process_paths:
                    orch.ingest(frames_dir, run_id=run_id, selected_models=models, progress=False)
            else:
                # Stage the capped images as symlinks into one dir so the
                # Orchestrator runs once (and hashes the model weights once).
                staged_to_src: dict[Path, Path] = {}
                for i, src in enumerate(plan["images"]):
                    link = frames_dir / f"{i:06d}_{src.name}"
                    if not link.exists():
                        try:
                            os.symlink(src.resolve(), link)
                        except FileExistsError:
                            pass
                    staged_to_src[link] = src
                progress(STAGES[1])
                if staged_to_src:
                    orch.ingest(frames_dir, run_id=run_id, selected_models=models, progress=False)
                process_paths = list(staged_to_src.values())  # original paths to store/serve

            progress(STAGES[2])  # Auto-labelling

            for fp in process_paths:
                image_id, w, h = hash_decoded_image(fp)
                store.upsert_image(image_id, str(fp), w, h, dataset_id=dataset_id)
                if spec.get("trustThreshold") is not None:
                    _auto_accept(store, dataset_id, image_id, run_id, models,
                                 float(spec["trustThreshold"]))
                done += 1
                progress(STAGES[3])  # Hash + dedup

            progress(STAGES[4])  # Writing to Postgres

        store.refresh_summary_view()
        done = total or done
        store.update_job(job_id, status="done", progress=100, frames_done=done,
                         stage="done", finished_at=_now())
        progress("done", status="done")
    except Exception as exc:  # noqa: BLE001 — surface to the job row + WS
        if store is not None:
            try:
                store.update_job(job_id, status="error", error=str(exc))
            except Exception:
                pass
        emit({"jobId": job_id, "stage": "error", "progress": 0, "framesDone": done,
              "framesTotal": total, "status": "error", "error": str(exc)})
    finally:
        if store is not None:
            store.close()


def _auto_accept(store, dataset_id: str, image_id: str, run_id: str,
                 models: set[str], threshold: float) -> None:
    """Write 'accepted' reviews server-side for confident detections — the
    confidence-gated pre-labelling that seeds the ground-truth layer. Detection
    ids match fusion's scheme (``<image_id>_<model_id>_<idx>``)."""
    df = store.query_df(
        f"SELECT model_id, payload FROM predictions_raw "
        f"WHERE image_id='{image_id}' AND run_id='{run_id}'"
    )
    if df.empty:
        return
    for _, row in df.iterrows():
        model_id = row["model_id"]
        payload = row["payload"] if isinstance(row["payload"], dict) else {}
        if model_id == "fr":
            confs = [d.get("score") or 0.0 for d in payload.get("recognized", [])]
        else:
            confs = [d.get("conf") or 0.0 for d in payload.get("detections", [])]
        for idx, conf in enumerate(confs):
            if float(conf) >= threshold:
                store.write_review(f"{image_id}_{model_id}_{idx}", "detection", "accepted",
                                   dataset_id=dataset_id, reviewer="auto")


def _now():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)


# ---- arq worker (prod) -----------------------------------------------------
async def run_ingest(ctx, job_id: str, spec: dict) -> None:
    """arq entrypoint: run the sync pipeline in a thread, publish frames to Redis
    pub/sub (the API subscribes via redis_progress_stream)."""
    import json

    redis = ctx.get("redis") if isinstance(ctx, dict) else None
    loop = asyncio.get_running_loop()

    def emit(frame: dict) -> None:
        if redis is not None:
            asyncio.run_coroutine_threadsafe(redis.publish(f"job:{job_id}", json.dumps(frame)), loop)

    await asyncio.to_thread(run_ingest_sync, job_id, spec, emit)


async def redis_progress_stream(job_id: str, redis_url: Optional[str] = None) -> AsyncIterator[dict]:
    """Prod WS source: subscribe to Redis pub/sub for a job's progress frames.
    Swap api.main's WS handler to this when running the arq worker out-of-process."""
    import json

    import redis.asyncio as aioredis

    client = aioredis.from_url(redis_url or settings.redis_url)
    pubsub = client.pubsub()
    await pubsub.subscribe(f"job:{job_id}")
    try:
        async for msg in pubsub.listen():
            if msg.get("type") != "message":
                continue
            frame = json.loads(msg["data"])
            yield frame
            if frame.get("status") in ("done", "error"):
                return
    finally:
        await pubsub.unsubscribe(f"job:{job_id}")
        await client.close()


class WorkerSettings:
    """arq worker settings. Run: ``arq sensveridian.ingest.worker.WorkerSettings``."""

    functions = [run_ingest]

    @staticmethod
    def redis_settings():
        from arq.connections import RedisSettings

        return RedisSettings.from_dsn(settings.redis_url)
