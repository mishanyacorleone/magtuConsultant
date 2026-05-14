from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.core.dependencies import DbDep, QdrantDep, RedisDep, SettingsDep
from app.core.logging import get_logger
from app.services.embedding.client import EmbeddingClient
from app.services.llm.circuit_breaker import CircuitBreaker
from app.services.llm.client import LLMClient

router = APIRouter()
logger = get_logger(__name__)


@router.get("/health")
async def health(
    db: DbDep,
    qdrant: QdrantDep,
    redis: RedisDep,
    settings: SettingsDep
) -> JSONResponse:
    """
    Проверяет доступность всех зависимостей системы.
 
    Используется Docker healthcheck и мониторингом.
    Возвращает 200 если всё ок, 503 если хотя бы один сервис упал.
    """
    services: dict[str, str] = {}
    all_ok = True

    # PostgreSQL
    try:
        await db.execute(__import__("sqlalchemy", fromlist=["text"]).text("SELECT 1"))
        services["postgres"] = "ok"
    except Exception as exc:
        services["postgres"] = f"error: {exc}"
        all_ok = False

    # Qdrant
    try:
        await qdrant.get_collections()
        services["qdrant"] = "ok"
    except Exception as exc:
        services["qdrant"] = f"error: {exc}"
        all_ok = False

    # Redis
    try:
        await redis.ping()
        services["redis"] = "ok"
    except Exception as exc:
        services["redis"] = f"error: {exc}"
        all_ok = False

    # vLLM LLM
    try:
        circuit_breaker = CircuitBreaker(redis)
        llm = LLMClient(circuit_breaker)
        ok = await llm.health_check()
        services["vllm_llm"] = "ok" if ok else f"error: no models"
        if not ok:
            all_ok = False
    except Exception as exc:
        services["vllm_llm"] = f"error: {exc}"
        all_ok = False

    # vLLM Embedder
    try:
        embedder = EmbeddingClient()
        ok = await embedder.health_check()
        services["vllm_embedder"] = "ok" if ok else "error: no models"
        if not ok:
            all_ok = False
    except Exception as exc:
        services["vllm_embedder"] = f"error: {exc}"
        all_ok = False

    status_code = 200 if all_ok else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ok" if all_ok else "degraded",
            "services": services
        }
    )
