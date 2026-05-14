import asyncio

from app.core.config import get_settings
from app.core.exceptions import NodeTimeoutError
from app.core.logging import get_logger
from app.repositories.postgres import SQLRepository
from app.services.agent.state import AgentState

logger = get_logger(__name__)
settings = get_settings()


async def sql_executor_node(state: AgentState, sql_repo: SQLRepository) -> dict:
    """Выполняет валидированный SQL запрос в PostgreSQL.

    При ошибке выполнения возвращает sql_error для retry логики.
    Репозиторий содержит второй рубеж защиты от не-SELECT запросов.

    Returns:
        При успехе:  {"sql_result": [...], "sql_error": None}
        При ошибке:  {"sql_result": [], "sql_error": "...", "retry_count": N+1}
    """
    sql = state.get("sql_query", "")

    logger.info("node_sql_executor_start | trace_id=%s sql=%s retry_count=%s", state["trace_id"], sql[:200], state["retry_count"])

    try:
        rows = await asyncio.wait_for(
            sql_repo.execute_select(sql),
            timeout=settings.timeout_sql_executor,
        )
    except asyncio.TimeoutError:
        raise NodeTimeoutError("sql_executor", settings.timeout_sql_executor)
    except Exception as exc:
        error = str(exc)
        logger.warning("node_sql_executor_error | trace_id=%s error=%s sql=%s retry_count=%s", state["trace_id"], error, sql[:200], state["retry_count"])
        return {
            "sql_result": [],
            "sql_error": error,
            "retry_count": state["retry_count"] + 1,
        }

    logger.info("node_sql_executor_done | trace_id=%s rows_count=%s", state["trace_id"], len(rows))

    return {
        "sql_result": rows,
        "sql_error": None,
    }