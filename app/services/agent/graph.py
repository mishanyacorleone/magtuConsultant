from functools import partial

import redis.asyncio as aioredis
from langgraph.graph import END, START, StateGraph
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.repositories.postgres import SQLRepository
from app.repositories.qdrant import QdrantRepository
from app.services.agent.edges import (
    route_after_sql_execution,
    route_after_sql_validation,
    route_after_table_selector,
    route_by_source,
    route_retry_or_fallback,
)
from app.services.agent.nodes.answer_generator import answer_generator_node
from app.services.agent.nodes.qdrant_search import qdrant_search_node
from app.services.agent.nodes.router import router_node
from app.services.agent.nodes.sample_data import sample_data_node
from app.services.agent.nodes.sql_executor import sql_executor_node
from app.services.agent.nodes.sql_generator import sql_generator_node
from app.services.agent.nodes.sql_validator import sql_validator_node
from app.services.agent.nodes.subject_matcher import subject_matcher_node
from app.services.agent.nodes.table_selector import table_selector_node
from app.services.agent.state import AgentState
from app.services.embedding.client import EmbeddingClient
from app.services.history.service import HistoryService
from app.services.llm.circuit_breaker import CircuitBreaker
from app.services.llm.client import LLMClient

logger = get_logger(__name__)


def build_graph(
    session: AsyncSession,
    qdrant_client: AsyncQdrantClient,
    redis: aioredis.Redis,
) -> StateGraph:
    """Собирает и компилирует LangGraph граф агента."""

    circuit_breaker = CircuitBreaker(redis)
    llm = LLMClient(circuit_breaker)
    embedding_client = EmbeddingClient()
    qdrant_repo = QdrantRepository(qdrant_client)
    sql_repo = SQLRepository(session)
    history_service = HistoryService(session)

    async def load_history(state: AgentState) -> dict:
        history = await history_service.get_history(state["user_id"])
        return {"history": history}

    async def save_history(state: AgentState) -> dict:
        if state.get("should_save_history", True) and state.get("answer"):
            await history_service.save(
                user_id=state["user_id"],
                question=state["question"],
                answer=state["answer"],
                source=state.get("source"),
            )
        return {}

    graph = StateGraph(AgentState)

    # Ноды
    graph.add_node("load_history", load_history)
    graph.add_node("router", partial(router_node, llm=llm))
    graph.add_node("qdrant_search", partial(
        qdrant_search_node,
        embedding_client=embedding_client,
        qdrant_repo=qdrant_repo,
    ))
    graph.add_node("table_selector", partial(table_selector_node, llm=llm))
    graph.add_node("subject_matcher", partial(
        subject_matcher_node,
        llm=llm,
        db_session=session,
    ))
    graph.add_node("sample_data", partial(sample_data_node, sql_repo=sql_repo))
    graph.add_node("sql_generator", partial(sql_generator_node, llm=llm))
    graph.add_node("sql_validator", sql_validator_node)
    graph.add_node("sql_executor", partial(sql_executor_node, sql_repo=sql_repo))
    graph.add_node("answer_generator", partial(answer_generator_node, llm=llm))
    graph.add_node("save_history", save_history)

    # Рёбра
    graph.add_edge(START, "load_history")
    graph.add_edge("load_history", "router")

    graph.add_conditional_edges(
        "router",
        route_by_source,
        {"qdrant": "qdrant_search", "postgres": "table_selector"},
    )

    graph.add_edge("qdrant_search", "answer_generator")

    # После table_selector: vi_soo_vo → subject_matcher, остальное → sample_data
    graph.add_conditional_edges(
        "table_selector",
        route_after_table_selector,
        {"subject_matcher": "subject_matcher", "sample_data": "sample_data"},
    )

    # subject_matcher сразу идёт в answer_generator — данные уже готовы
    graph.add_edge("subject_matcher", "answer_generator")

    # Обычный postgres path
    graph.add_edge("sample_data", "sql_generator")
    graph.add_edge("sql_generator", "sql_validator")

    graph.add_conditional_edges(
        "sql_validator",
        route_after_sql_validation,
        {"execute": "sql_executor", "retry_or_fallback": "retry_or_fallback_router"},
    )

    graph.add_conditional_edges(
        "sql_executor",
        route_after_sql_execution,
        {"generate_answer": "answer_generator", "retry_or_fallback": "retry_or_fallback_router"},
    )

    graph.add_node("retry_or_fallback_router", lambda state: {})
    graph.add_conditional_edges(
        "retry_or_fallback_router",
        route_retry_or_fallback,
        {"retry": "sql_generator", "fallback_qdrant": "qdrant_search"},
    )

    graph.add_edge("answer_generator", "save_history")
    graph.add_edge("save_history", END)

    return graph.compile()