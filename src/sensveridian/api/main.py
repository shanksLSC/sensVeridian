"""Veridian Studio API — FastAPI app entrypoint.

Run:  uvicorn sensveridian.api.main:app --reload --port 8000

Endpoints map 1:1 to the design handoff's API contract and are exercised by the
front-end RestAdapter (design_reference/app/api.js). The route handlers are thin
wrappers over :class:`sensveridian.store.pg_async.AsyncPgStore`; the ingest
worker (sensveridian.ingest.worker) drives the synchronous Orchestrator.
"""
from __future__ import annotations

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routers import connections, datasets, ingest, lineage, models, reviews

app = FastAPI(title="Veridian Studio API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,  # the studio origin(s)
    allow_methods=["*"],
    allow_headers=["*"],
)

API = "/api/v1"
app.include_router(datasets.router, prefix=API, tags=["datasets"])
app.include_router(models.router, prefix=API, tags=["models"])
app.include_router(reviews.router, prefix=API, tags=["reviews"])
app.include_router(ingest.router, prefix=API, tags=["ingest"])
app.include_router(lineage.router, prefix=API, tags=["lineage"])
app.include_router(connections.router, prefix=API, tags=["connections"])


@app.get(f"{API}/health")
async def health():
    return {"ok": True, "service": "veridian", "storage": "postgres"}


@app.websocket(f"{API}/ws/jobs/{{job_id}}")
async def ws_job(ws: WebSocket, job_id: str):
    """Stream ingest job progress to the pipeline animation in the UI."""
    from ..ingest.worker import job_progress_stream

    await ws.accept()
    try:
        async for frame in job_progress_stream(job_id):
            await ws.send_json(frame)  # {jobId, stage, progress, framesDone, framesTotal, status}
    except WebSocketDisconnect:
        return
