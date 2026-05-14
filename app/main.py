from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from app.api.v1.routes import admin, chat, health
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import setup_logging

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация и завершение работы приложения."""
    settings = get_settings()

    logger.info("app_startup | env=%s llm_model=%s embedder_model=%s", settings.app_env, settings.vllm_llm_model, settings.vllm_embedder_model)

    # Обеспечиваем существование Qdrant коллекции при старте
    try:
        from app.core.dependencies import get_qdrant
        from app.repositories.qdrant import QdrantRepository

        qdrant_client = await get_qdrant(settings)
        qdrant_repo = QdrantRepository(qdrant_client)
        await qdrant_repo.ensure_collection_exists()
        logger.info("qdrant_collection_ready")
    except Exception as exc:
        logger.warning("qdrant_init_failed | error=%s", str(exc))

    yield

    logger.info("app_shutdown")


def create_app() -> FastAPI:
    setup_logging()
    settings = get_settings()

    app = FastAPI(
        title="MAGTU Consultant API",
        description="Система автоматизированного консультирования абитуриентов МГТУ им. Г.И. Носова",
        version="1.0.0",
        docs_url="/docs" if settings.app_env == "development" else None,
        redoc_url="/redoc" if settings.app_env == "development" else None,
        lifespan=lifespan
    )

    register_exception_handlers(app)

    app.include_router(health.router, tags=["system"])
    app.include_router(chat.router, prefix="/v1", tags=["chat"])
    app.include_router(admin.router, prefix="/v1", tags=["admin"])

    return app


app = create_app()
