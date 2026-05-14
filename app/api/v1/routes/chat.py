import uuid

from fastapi import APIRouter

from app.api.v1.schemas.chat import ChatRequest, ChatResponse, Fragment, SourceLink
from app.core.dependencies import DbDep, QdrantDep, RedisDep
from app.core.logging import get_logger
from app.repositories.qdrant import SearchResult
from app.services.agent.graph import build_graph
from app.services.agent.state import make_initial_state

router = APIRouter()
logger = get_logger(__name__)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: DbDep,
    qdrant: QdrantDep,
    redis: RedisDep,
) -> ChatResponse:
    """Основной эндпоинт генерации ответа.

    Принимает вопрос пользователя, прогоняет через агентный граф
    и возвращает ответ с релевантными фрагментами.
    """
    trace_id = str(uuid.uuid4())

    logger.info("chat_request | trace_id=%s user_id=%s question=%s", trace_id, request.user_id, request.question[:100])

    initial_state = make_initial_state(
        user_id=request.user_id,
        question=request.question,
        trace_id=trace_id,
    )

    graph = build_graph(session=db, qdrant_client=qdrant, redis=redis)
    final_state = await graph.ainvoke(initial_state)

    fragments = _build_fragments(final_state)
    source_link = _build_source_link(final_state)

    logger.info("chat_response | trace_id=%s source=%s fragments=%s answer=%s", trace_id, final_state.get("source", "unknown"), len(fragments), len(final_state.get("answer", "")))

    return ChatResponse(
        answer=final_state.get("answer", ""),
        source=final_state.get("source", "unknown"),
        fragments=fragments,
        source_link=source_link,
        trace_id=trace_id,
    )


def _build_fragments(state: dict) -> list[Fragment]:
    """Конвертирует SearchResult из state в schema Fragment."""
    raw: list[SearchResult] = state.get("fragments", [])
    return [
        Fragment(
            text=f.text,
            source=f.source,
            group=f.group,
            score=round(f.score, 4),
        )
        for f in raw
    ]


def _build_source_link(state: dict) -> SourceLink | None:
    """Возвращает ссылку на страницу сайта МГТУ если ответ из PostgreSQL."""
    link = state.get("source_link")
    if not link:
        return None
    return SourceLink(url=link.get("url", ""), title=link.get("title", ""))