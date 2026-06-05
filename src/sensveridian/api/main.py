"""Veridian Studio API — FastAPI app entrypoint.

Run:  uvicorn sensveridian.api.main:app --reload --port 8000
UI:   http://localhost:8000/studio   (served from the bundled no-build front-end)

Endpoints map 1:1 to the design handoff's API contract and are exercised by the
front-end RestAdapter (studio/app/api.js). The route handlers are thin wrappers
over :class:`sensveridian.store.pg_async.AsyncPgStore`; the ingest worker
(sensveridian.ingest.worker) drives the synchronous Orchestrator.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import settings
from .db import SessionLocal
from .routers import connections, datasets, fs, ingest, lineage, models, reviews
from ..store.pg_async import AsyncPgStore

log = logging.getLogger("sensveridian.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Seed the model catalogue so /models + the ingest picker work before any
    # ingest. Best-effort: a missing DB must not block boot; skip under pytest
    # so the test suite never touches a live database.
    if "PYTEST_CURRENT_TEST" not in os.environ:
        try:
            async with SessionLocal() as session:
                await AsyncPgStore(session).seed_models()
        except Exception as exc:  # noqa: BLE001
            log.warning("model seeding skipped (%s)", exc)
    yield


app = FastAPI(title="Veridian Studio API", version="1.0.0", lifespan=lifespan)

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
app.include_router(fs.router, prefix=API, tags=["fs"])


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


# Bundled no-build front-end at /studio (mounted last so it cannot shadow /api).
if os.path.isdir(settings.studio_dir):
    app.mount("/studio", StaticFiles(directory=settings.studio_dir, html=True), name="studio")
