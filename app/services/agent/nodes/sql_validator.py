import sqlglot
import sqlglot.errors

from app.core.logging import get_logger
from app.services.agent.state import AgentState

logger = get_logger(__name__)


def sql_validator_node(state: AgentState) -> dict:
    """Валидирует сгенерированный SQL через sqlglot.

    Синхронная нода — sqlglot работает без IO.

    Проверки:
    1. Синтаксическая корректность (sqlglot парсинг)
    2. Только SELECT запросы — никаких DML/DDL

    Returns:
        Обновление state:
        - {"sql_error": None} если запрос валиден
        - {"sql_error": "<описание ошибки>", "retry_count": N+1} если невалиден
    """
    sql = state.get("sql_query", "")

    if not sql:
        return {
            "sql_error": "SQL query is empty",
            "retry_count": state["retry_count"] + 1,
        }

    # Проверка: только SELECT
    normalized = sql.strip().upper()
    if not normalized.startswith("SELECT"):
        error = f"Only SELECT queries are allowed. Got: {sql[:50]}"
        logger.warning("sql_validation_failed | trace_id=%s reason=%s sql=%s", state["trace_id"], "not_select", sql[:200])
        return {
            "sql_error": error,
            "retry_count": state["retry_count"] + 1,
        }

    # Синтаксическая проверка через sqlglot
    try:
        statements = sqlglot.parse(sql, dialect="postgres")

        if not statements or statements[0] is None:
            raise sqlglot.errors.ParseError("Empty parse result")

        # Дополнительно убеждаемся что это SELECT
        statement = statements[0]
        if not isinstance(statement, sqlglot.exp.Select):
            raise sqlglot.errors.ParseError(
                f"Expected SELECT, got {type(statement).__name__}"
            )

    except sqlglot.errors.ParseError as exc:
        error = f"SQL syntax error: {exc}"
        logger.warning("sql_validation_failed | trace_id=%s reason=%s sql=%s error=%s", state["trace_id"], "parse_error", sql[:200], str(exc))
        return {
            "sql_error": error,
            "retry_count": state["retry_count"] + 1,
        }

    logger.debug("sql_validation_passed | trace_id=%s sql=%s", state["trace_id"], sql[:200])

    return {"sql_error": None}