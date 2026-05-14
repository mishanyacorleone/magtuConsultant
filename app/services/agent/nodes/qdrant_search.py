import asyncio

from app.core.config import get_settings
from app.core.exceptions import NodeTimeoutError, QdrantSearchError
from app.core.logging import get_logger
from app.repositories.qdrant import QdrantRepository, SearchResult
from app.services.agent.state import AgentState
from app.services.embedding.client import EmbeddingClient

logger = get_logger(__name__)
settings = get_settings()


async def qdrant_search_node(
    state: AgentState,
    embedding_client: EmbeddingClient,
    qdrant_repo: QdrantRepository,
) -> dict:
    """Векторный поиск по вопросу пользователя.

    Векторизует вопрос и ищет top-k релевантных фрагментов.
    При fallback из postgres path — использует те же данные.

    Returns:
        Обновление state: {"fragments": [...], "source": "qdrant"}
    """
    logger.info("node_qdrant_search_start | trace_id=%s", state["trace_id"])

    try:
        vector = await asyncio.wait_for(
            embedding_client.embed(state["question"]),
            timeout=settings.timeout_qdrant_search,
        )
        results: list[SearchResult] = await asyncio.wait_for(
            qdrant_repo.search(vector),
            timeout=settings.timeout_qdrant_search,
        )
    except asyncio.TimeoutError:
        raise NodeTimeoutError("qdrant_search", settings.timeout_qdrant_search)
    except Exception as exc:
        raise QdrantSearchError(f"Qdrant search failed: {exc}") from exc

    logger.info("node_qdrant_search_done | trace_id=%s results_count=%s top_score=%s", state["trace_id"], len(results), results[0].score if results else None)

    return {
        "fragments": results,
        "source": "qdrant",
    }