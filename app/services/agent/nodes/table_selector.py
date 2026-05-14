import asyncio
import json
from pathlib import Path

from app.core.config import get_settings
from app.core.exceptions import NodeTimeoutError
from app.core.logging import get_logger
from app.prompts.sql import TABLE_SELECTOR_SYSTEM_PROMPT, TABLE_SELECTOR_USER_TEMPLATE
from app.services.agent.state import AgentState
from app.services.llm.client import LLMClient

logger = get_logger(__name__)
settings = get_settings()

_SCHEMAS_PATH = Path(__file__).parent.parent.parent.parent / "prompts" / "tables" / "schemas.json"


def _load_schemas() -> dict:
    with _SCHEMAS_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    # Убираем служебный ключ _comment
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _format_tables_description(schemas: dict) -> str:
    """Формирует читаемое описание таблиц для промпта."""
    lines = []
    for table_name, schema in schemas.items():
        lines.append(f"- {table_name}: {schema['description']}")
    return "\n".join(lines)


async def table_selector_node(state: AgentState, llm: LLMClient) -> dict:
    """Выбирает таблицы PostgreSQL для ответа на вопрос.

    Загружает schemas.json, передаёт список таблиц с описаниями в LLM,
    получает названия таблиц для следующего шага.

    Returns:
        Обновление state: {"selected_tables": ["table1", "table2"]}
    """
    logger.info("node_table_selector_start | trace_id=%s", state["trace_id"])

    schemas = _load_schemas()
    tables_description = _format_tables_description(schemas)

    messages = [
        {
            "role": "system",
            "content": TABLE_SELECTOR_SYSTEM_PROMPT.format(
                tables_description=tables_description
            ),
        },
        {
            "role": "user",
            "content": TABLE_SELECTOR_USER_TEMPLATE.format(question=state["question"]),
        },
    ]

    try:
        raw = await asyncio.wait_for(
            llm.chat(messages, temperature=0.0, max_tokens=50),
            timeout=settings.timeout_table_selector,
        )
    except asyncio.TimeoutError:
        raise NodeTimeoutError("table_selector", settings.timeout_table_selector)

    # Парсим ответ: "passing_scores, specialties" → ["passing_scores", "specialties"]
    available = set(schemas.keys())
    selected = [
        t.strip()
        for t in raw.split(",")
        if t.strip() in available
    ]

    if not selected:
        # Модель вернула что-то невалидное — берём первую таблицу как fallback
        logger.warning("table_selector_invalid_response | trace_id=%s raw=%s", state["trace_id"], raw)
        selected = [next(iter(available))]

    logger.info("node_table_selector_done | trace_id=%s selected_tables=%s", state["trace_id"], selected)

    return {"selected_tables": selected}