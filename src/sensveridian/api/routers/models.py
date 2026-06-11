"""Models, version history, regression diffs, promotion, registration."""
from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from ..deps import get_store, require_auth
from ..schemas import Flip, Model
from ...store.pg_async import AsyncPgStore

router = APIRouter(dependencies=[Depends(require_auth)])


class RegisterModelSpec(BaseModel):
    # model_ prefix would collide with pydantic's protected namespace
    model_config = {"protected_namespaces": ()}
    model_id: str
    display_name: str
    weights_path: str
    config_path: str | None = None
    runner_kind: str | None = None
    input_spec: str = ""
    version: str = "1"
    n_classes: int = 1
    depends_on: str | None = None


@router.get("/models", response_model=list[Model])
async def list_models(store: AsyncPgStore = Depends(get_store)):
    return [Model(**m) for m in await store.list_models()]


@router.post("/models")
async def register_model(spec: RegisterModelSpec, store: AsyncPgStore = Depends(get_store)):
    """Register a model (or new weights) so it appears in the registry and is
    selectable for ingest. Validates the weights/config paths exist on disk."""
    if not re.fullmatch(r"[a-z0-9_]+", spec.model_id or ""):
        raise HTTPException(400, "model_id must be lowercase letters, digits, or underscore")
    if not Path(spec.weights_path).is_file():
        raise HTTPException(400, f"weights file not found: {spec.weights_path}")
    if spec.config_path and not Path(spec.config_path).is_file():
        raise HTTPException(400, f"config file not found: {spec.config_path}")
    await store.register_model(
        model_id=spec.model_id, display_name=spec.display_name, weights_path=spec.weights_path,
        config_path=spec.config_path, runner_kind=spec.runner_kind, input_spec=spec.input_spec,
        version=spec.version, n_classes=spec.n_classes, depends_on=spec.depends_on,
    )
    return {"ok": True, "model": spec.model_id, "version": spec.version}


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
