from app.core.config import get_settings
from app.services.agent.state import AgentState

settings = get_settings()


def route_by_source(state: AgentState) -> str:
    """После router_node: направляет в qdrant или postgres path."""
    return state.get("source", "qdrant")


def route_after_sql_validation(state: AgentState) -> str:
    """
    После sql_validator_node: выполнять SQL или генерировать заново.
 
    Если sql_error есть — нода уже инкрементировала retry_count.
    Здесь читаем актуальное значение.
    """
    if state.get("sql_error") is None:
        return "execute"
    return "retry_or_fallback"


def route_after_sql_execution(state: AgentState) -> str:
    """После sql_executor_node: генерировать ответ или идти в retry/fallback."""
    if state.get("sql_error") is None:
        return "generate_answer"
    return "retry_or_fallback"


def route_retry_or_fallback(state: AgentState) -> str:
    """
    Решает: ещё одна попытка SQL или fallback в Qdrant.
 
    retry_count уже инкрементирован в validator или executor.
    """
    if state["retry_count"] <= settings.sql_max_retries:
        return "retry"
    return "fallback_qdrant"
