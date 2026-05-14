import asyncio

from app.core.config import get_settings
from app.core.exceptions import NodeTimeoutError
from app.core.logging import get_logger
from app.prompts.router import ROUTER_SYSTEM_PROMPT, ROUTER_USER_TEMPLATE
from app.services.agent.state import AgentState
from app.services.llm.client import LLMClient

logger = get_logger(__name__)
settings = get_settings()


async def router_node(state: AgentState, llm: LLMClient) -> dict:
    """Определяет источник данных для ответа на вопрос.

    LLM классифицирует вопрос как структурированный (postgres)
    или неструктурированный (qdrant).

    Returns:
        Обновление state: {"source": "qdrant" | "postgres"}
    """
    logger.info("node_router_start | trace_id=%s question=%s", state["trace_id"], state["question"][:100])

    messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": ROUTER_USER_TEMPLATE.format(question=state["question"])},
    ]

    try:
        raw = await asyncio.wait_for(
            llm.chat(messages, temperature=0.0, max_tokens=10),
            timeout=settings.timeout_router,
        )
    except asyncio.TimeoutError:
        raise NodeTimeoutError("router", settings.timeout_router)

    source = raw.strip().lower()
    if source not in ("qdrant", "postgres"):
        # Если модель ответила что-то нестандартное — fallback на qdrant
        logger.warning("router_unexpected_response | trace_id=%s raw=%s", state["trace_id"], raw)
        source = "qdrant"

    logger.info("node_router_done | trace_id=%s source=%s", state["trace_id"], source)
    return {"source": source}