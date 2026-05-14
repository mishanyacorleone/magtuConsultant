import asyncio
import json
from pathlib import Path

from app.core.config import get_settings
from app.core.exceptions import NodeTimeoutError
from app.core.logging import get_logger
from app.repositories.postgres import SQLRepository
from app.services.agent.state import AgentState

logger = get_logger(__name__)
settings = get_settings()

_SCHEMAS_PATH = Path(__file__).parent.parent.parent.parent / "prompts" / "tables" / "schemas.json"


def _load_schemas() -> dict:
    with _SCHEMAS_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _format_sample(table_name: str, rows: list[dict]) -> str:
    """Форматирует сэмпл строк в читаемый текст для промпта."""
    if not rows:
        return f"Таблица {table_name}: данных нет."
    header = ", ".join(rows[0].keys())
    lines = [f"Таблица {table_name} (колонки: {header}):"]
    for row in rows:
        lines.append("  " + ", ".join(f"{k}={v!r}" for k, v in row.items()))
    return "\n".join(lines)


def _format_distinct(table_name: str, column: str, values: list[str]) -> str:
    """Форматирует уникальные значения колонки для промпта."""
    values_str = ", ".join(f"'{v}'" for v in values[:30])
    return f"Уникальные значения {table_name}.{column}: {values_str}"


async def sample_data_node(state: AgentState, sql_repo: SQLRepository) -> dict:
    """Получает сэмпл данных из выбранных таблиц.

    Для каждой таблицы:
    1. SELECT * ORDER BY RANDOM() LIMIT 10 — живой срез данных
    2. SELECT DISTINCT <col> для name_columns — реальные значения для матчинга

    Результат передаётся в промпт SQL генератора.

    Returns:
        Обновление state: {"table_sample": "<форматированный текст>"}
    """
    logger.info("node_sample_data_start | trace_id=%s tables=%s", state["trace_id"], state["selected_tables"])

    schemas = _load_schemas()
    sample_parts: list[str] = []

    for table_name in state["selected_tables"]:
        schema = schemas.get(table_name, {})

        try:
            # Случайный сэмпл строк
            rows = await asyncio.wait_for(
                sql_repo.sample_table(table_name, settings.sql_sample_limit),
                timeout=settings.timeout_sample_data,
            )
            sample_parts.append(_format_sample(table_name, rows))

            # Уникальные значения name-колонок
            for col in schema.get("name_columns", []):
                distinct = await asyncio.wait_for(
                    sql_repo.get_distinct_values(
                        table_name, col, settings.sql_distinct_limit
                    ),
                    timeout=settings.timeout_sample_data,
                )
                sample_parts.append(_format_distinct(table_name, col, distinct))

        except asyncio.TimeoutError:
            raise NodeTimeoutError("sample_data", settings.timeout_sample_data)
        except Exception as exc:
            # Не критично — продолжаем без сэмпла этой таблицы
            logger.warning("sample_data_failed | trace_id=%s table=%s error=%s", state["trace_id"], table_name, str(exc))
            sample_parts.append(f"Таблица {table_name}: не удалось получить примеры данных.")

    table_sample = "\n\n".join(sample_parts)

    logger.info("node_sample_data_done | trace_id=%s sample_length=%s", state["trace_id"], len(table_sample))

    return {"table_sample": table_sample}