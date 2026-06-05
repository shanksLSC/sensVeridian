"""Human review verdicts (verify / accept / reject / edit / flag).

The UI round-trips only the target id and the patch; the dataset a target
belongs to is resolved server-side (AsyncPgStore.resolve_dataset_id) so an
accepted detection correctly extends that dataset's ground-truth layer.
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from ..deps import get_store, require_auth
from ..schemas import BulkReview, ReviewPatch
from ...store.pg_async import AsyncPgStore

router = APIRouter(dependencies=[Depends(require_auth)])


def _kind(target_id: str) -> str:
    if target_id.startswith("img:"):
        return "image"
    if target_id.startswith("flip:"):
        return "flip"
    if target_id.startswith("seg_") or target_id.split(":", 1)[0] == "seg":
        return "segment"
    return "detection"


@router.put("/reviews/{target_id}")
async def save_review(target_id: str, patch: ReviewPatch, store: AsyncPgStore = Depends(get_store)):
    dataset_id = await store.resolve_dataset_id(target_id)
    await store.save_review(target_id, _kind(target_id), patch.verdict, dataset_id, patch.box)
    return {"ok": True, "targetId": target_id, "serverTs": int(time.time() * 1000)}


@router.post("/reviews:bulk")
async def bulk_review(body: BulkReview, store: AsyncPgStore = Depends(get_store)):
    kinds = [_kind(t) for t in body.target_ids]
    # all targets in a bulk verify belong to one frame/clip -> one dataset
    dataset_id = None
    for t in body.target_ids:
        dataset_id = await store.resolve_dataset_id(t)
        if dataset_id:
            break
    n = await store.bulk_review(body.target_ids, kinds, body.patch.verdict, dataset_id)
    return {"ok": True, "n": n}
