from collections.abc import AsyncGenerator
from typing import Annotated

import redis.asyncio as aioredis
from fastapi import Depends
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import Settings, get_settings

# Синглтоны - инициализируются один раз при первом обращении
_engine = None
_session_factory = None
_qdrant_client = None
_redis_client = None


def _get_engine(settings: Settings):
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            settings.postgres_dsn,
            pool_size=10,
            max_overflow=20,
            echo=settings.app_env == "development",
        )
    return _engine


def _get_session_factory(settings: Settings) -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            _get_engine(settings),
            expire_on_commit=False,
            class_=AsyncSession
        )
    return _session_factory


async def get_db(
    settings: Annotated[Settings, Depends(get_settings)]
) -> AsyncGenerator[AsyncSession, None]:
    """Предоставляет сессию PostgreSQL на время одного запроса"""
    factory = _get_session_factory(settings)
    async with factory() as session:
        yield session
    

async def get_qdrant(
    settings: Annotated[Settings, Depends(get_settings)]
) -> AsyncQdrantClient:
    """Возвращает асинхронный клиент Qdrant (синглтон)"""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )
    return _qdrant_client


async def get_redis(
    settings: Annotated[Settings, Depends(get_settings)]
) -> aioredis.Redis:
    """Возвращает асинхронный клиент Redis (синглтон)"""
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True
        )
    return _redis_client


# Аннотированные типы для удобного использования в роутерах
DbDep = Annotated[AsyncSession, Depends(get_db)]
QdrantDep = Annotated[AsyncQdrantClient, Depends(get_qdrant)]
RedisDep = Annotated[aioredis.Redis, Depends(get_redis)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
