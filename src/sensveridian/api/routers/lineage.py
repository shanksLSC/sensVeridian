"""Provenance / lineage DAG."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import get_store, require_auth
from ..schemas import LineageGraph
from ...store.pg_async import AsyncPgStore

router = APIRouter(dependencies=[Depends(require_auth)])


@router.get("/lineage", response_model=LineageGraph)
async def lineage(store: AsyncPgStore = Depends(get_store)):
    """Content-addressed provenance DAG assembled from connections, ingest
    sources, datasets, prediction runs, reviews, augmentations and exports."""
    graph = await store.lineage()
    return LineageGraph(**graph)
