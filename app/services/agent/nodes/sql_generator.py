import asyncio
import json
from pathlib import Path

from app.core.config import get_settings
from app.core.exceptions import NodeTimeoutError
from app.core.logging import get_logger
from app.prompts.sql import (
    SQL_FIX_SYSTEM_PROMPT,
    SQL_FIX_USER_TEMPLATE,
    SQL_GENERATOR_SYSTEM_PROMPT,
    SQL_GENERATOR_USER_TEMPLATE,
)
from app.services.agent.state import AgentState
from app.services.llm.client import LLMClient

logger = get_logger(__name__)
settings = get_settings()

_SCHEMAS_PATH = Path(__file__).parent.parent.parent.parent / "prompts" / "tables" / "schemas.json"


def _load_schemas() -> dict:
    with _SCHEMAS_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _format_table_description(schemas: dict, selected_tables: list[str]) -> str:
    """Собирает описание выбранных таблиц для промпта."""
    parts = []
    for table_name in selected_tables:
        schema = schemas.get(table_name)
        if not schema:
            continue
        parts.append(f"Таблица: {table_name}")
        parts.append(f"Описание: {schema['description']}")
        parts.append("Колонки:")
        for col, desc in schema["columns"].items():
            parts.append(f"  {col}: {desc}")
        if schema.get("rules"):
            parts.append("Правила:")
            for rule in schema["rules"]:
                parts.append(f"  - {rule}")
        if schema.get("query_examples"):
            parts.append("Примеры запросов:")
            for ex in schema["query_examples"]:
                parts.append(f"  {ex}")
        parts.append("")
    return "\n".join(parts)


async def sql_generator_node(state: AgentState, llm: LLMClient) -> dict:
    """Генерирует SQL запрос или исправляет предыдущий при retry.

    При первой попытке (retry_count == 0) генерирует новый запрос.
    При повторных попытках передаёт предыдущий SQL и ошибку для исправления.

    Returns:
        Обновление state: {"sql_query": "SELECT ..."}
    """
    logger.info("node_sql_generator_start | trace_id=%s retry_count=%s tables=%s", state["trace_id"], state["retry_count"], state["selected_tables"])

    schemas = _load_schemas()
    table_description = _format_table_description(schemas, state["selected_tables"])

    is_retry = state["retry_count"] > 0 and state["sql_query"] is not None

    if is_retry:
        system_prompt = SQL_FIX_SYSTEM_PROMPT.format(
            table_description=table_description,
            table_sample=state["table_sample"],
            distinct_values="(см. выше в примерах данных)",
            previous_sql=state["sql_query"],
            error=state["sql_error"] or "неизвестная ошибка",
        )
        user_content = SQL_FIX_USER_TEMPLATE.format(question=state["question"])
    else:
        system_prompt = SQL_GENERATOR_SYSTEM_PROMPT.format(
            table_description=table_description,
            table_sample=state["table_sample"],
            distinct_values="(см. выше в примерах данных)",
        )
        user_content = SQL_GENERATOR_USER_TEMPLATE.format(question=state["question"])

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    try:
        raw_sql = await asyncio.wait_for(
            llm.chat(messages, temperature=0.0, max_tokens=512),
            timeout=settings.timeout_sql_generator,
        )
    except asyncio.TimeoutError:
        raise NodeTimeoutError("sql_generator", settings.timeout_sql_generator)

    # Убираем markdown блоки если модель их добавила
    sql = raw_sql.strip()
    if sql.startswith("```"):
        lines = sql.split("\n")
        sql = "\n".join(
            line for line in lines
            if not line.startswith("```")
        ).strip()

    logger.info("node_sql_generator_done | trace_id=%s sql=%s is_retry=%s", state["trace_id"], sql[:200], is_retry)

    return {"sql_query": sql}