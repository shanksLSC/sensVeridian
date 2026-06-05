"""Models, version history, regression diffs, promotion."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ..deps import get_store, require_auth
from ..schemas import Flip, Model
from ...store.pg_async import AsyncPgStore

router = APIRouter(dependencies=[Depends(require_auth)])


@router.get("/models", response_model=list[Model])
async def list_models(store: AsyncPgStore = Depends(get_store)):
    return [Model(**m) for m in await store.list_models()]


@router.get("/models/{model_id}/regressions", response_model=list[Flip])
async def regressions(model_id: str, base: str = Query(...), candidate: str = Query(...),
                      store: AsyncPgStore = Depends(get_store)):
    """Detections whose correctness flipped between two runs of the model,
    scored against the verified GT layer (regress=True => was-correct, now-wrong)."""
    return [Flip(**f) for f in await store.regressions(model_id, base, candidate)]


@router.post("/models/{model_id}/versions/{version}:promote")
async def promote(model_id: str, version: str, store: AsyncPgStore = Depends(get_store)):
    await store.promote_version(model_id, version)
    return {"ok": True, "model": model_id, "current": version}
