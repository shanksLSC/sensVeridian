"""Shared FastAPI dependencies: DB-backed store and (optional) bearer auth."""
from __future__ import annotations

from typing import Optional

from fastapi import Depends, Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ..store.pg_async import AsyncPgStore
from .config import settings
from .db import get_session


async def get_store(session: AsyncSession = Depends(get_session)) -> AsyncPgStore:
    """The async read/build store the routers use."""
    return AsyncPgStore(session)


async def require_auth(authorization: Optional[str] = Header(default=None)) -> None:
    """Enforce ``Authorization: Bearer <token>`` only when VERIDIAN_AUTH_TOKEN
    is configured; otherwise the API is open (local/dev default)."""
    if not settings.auth_token:
        return
    expected = f"Bearer {settings.auth_token}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="missing or invalid bearer token")
