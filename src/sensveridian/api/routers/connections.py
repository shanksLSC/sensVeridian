"""Watched external sources (S3 bucket / folder)."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends

from ..deps import get_store, require_auth
from ..schemas import ConnectionSpec
from ...store.pg_async import AsyncPgStore

router = APIRouter(dependencies=[Depends(require_auth)])


@router.post("/connections")
async def create_connection(spec: ConnectionSpec, store: AsyncPgStore = Depends(get_store)):
    cid = "conn_" + uuid.uuid4().hex[:8]
    await store.create_connection(cid, spec.uri, spec.schedule, spec.pipeline)
    await store.commit()
    # SEAM: register a scheduled watcher (arq cron / S3 event notification) that
    # creates ingest jobs as new objects land. See ingest/worker.py.
    return {"id": cid}
