"""Ingest jobs (video auto-label) + upload handshake."""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from ..deps import get_store, require_auth
from ..schemas import IngestJobSpec, JobHandle
from ...store.pg_async import AsyncPgStore

router = APIRouter(dependencies=[Depends(require_auth)])


@router.post("/ingest/jobs", response_model=JobHandle)
async def create_ingest_job(spec: IngestJobSpec, store: AsyncPgStore = Depends(get_store)):
    job_id = "job_" + uuid.uuid4().hex[:8]
    total = sum(g.frames for g in spec.groups)
    await store.create_ingest_job(job_id, spec.source, spec.model_dump(), total)
    await store.commit()
    # Hand to the worker: ffmpeg frame-split -> Orchestrator.ingest -> dedup ->
    # upsert -> combine per tag -> WS progress. Imported lazily so the API does
    # not hard-depend on the arq/Redis stack just to enqueue.
    from ...ingest.worker import enqueue_ingest

    await enqueue_ingest(job_id, spec.model_dump())
    return JobHandle(jobId=job_id, status="queued")


@router.get("/ingest/jobs/{job_id}")
async def get_job(job_id: str, store: AsyncPgStore = Depends(get_store)):
    job = await store.get_ingest_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job


@router.post("/ingest/uploads")
async def stage_uploads(body: dict, _: None = Depends(require_auth)):
    """Return local staging targets for uploaded media.

    The raw data store is the local filesystem (no S3): media is staged under
    ``media_root`` and ingested from there. The primary flow ingests folders
    already on disk under ``datasets_root`` and needs no upload at all.
    """
    files = body.get("files", [])
    from ..config import settings

    return {
        "uploads": [
            {"key": f, "path": str(Path(settings.media_root) / f)}
            for f in files
        ]
    }
