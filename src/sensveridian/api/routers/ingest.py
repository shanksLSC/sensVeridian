"""Ingest jobs (video auto-label) + upload handshake."""
from __future__ import annotations

import uuid

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
async def presign_uploads(body: dict, _: None = Depends(require_auth)):
    """Return presigned PUT URLs for media upload.

    SEAM: wire to S3 (boto3 ``generate_presigned_url``) or a local object store.
    Until then this returns local staging paths so the flow is exercisable.
    """
    files = body.get("files", [])
    from ..config import settings

    return {
        "uploads": [
            {"key": f, "url": f"file://{settings.media_root}/{f}", "method": "PUT"}
            for f in files
        ]
    }
