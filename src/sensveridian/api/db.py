"""Async SQLAlchemy engine/session for the API request handlers.

The API is I/O bound and runs on the event loop, so it uses asyncpg. The ingest
worker, which drives the synchronous Orchestrator, uses
``sensveridian.store.pg.PgStore`` (sync) instead — see that module's docstring.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .config import settings

engine = create_async_engine(settings.database_url, pool_size=10, max_overflow=20, future=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session
