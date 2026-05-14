import asyncio
import json
from datetime import datetime
from pathlib import Path

from app.core.config import get_settings
from app.core.exceptions import NodeTimeoutError
from app.core.logging import get_logger
from app.prompts.system import SYSTEM_PROMPT
from app.repositories.qdrant import SearchResult
from app.services.agent.state import AgentState
from app.services.llm.client import LLMClient

logger = get_logger(__name__)
settings = get_settings()

_NO_ANSWER = (
    "К сожалению, у меня нет точной информации по этому вопросу. "
    "Рекомендуем обратиться в приёмную комиссию: 8-800-100-1934."
)

_TABLE_URLS_PATH = Path(__file__).parent.parent.parent.parent / "prompts" / "tables" / "table_urls.json"


def _load_table_urls() -> dict:
    """Загружает ссылки на страницы сайта МГТУ для каждой таблицы."""
    try:
        with _TABLE_URLS_PATH.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _get_source_link(selected_tables: list[str]) -> dict | None:
    """Возвращает ссылку для первой из выбранных таблиц."""
    urls = _load_table_urls()
    for table in selected_tables:
        if table in urls:
            return urls[table]
    return None


def _format_qdrant_context(fragments: list[SearchResult]) -> str:
    """Форматирует qdrant фрагменты в контекст для промпта."""
    if not fragments:
        return ""
    parts = ["## Найденная информация\n"]
    for i, frag in enumerate(fragments, 1):
        parts.append(f"[{i}] {frag.text}")
        if frag.source:
            parts.append(f"Источник: {frag.source}")
        parts.append("")
    return "\n".join(parts)


def _format_sql_context(sql_result: list[dict], selected_tables: list[str]) -> str:
    """Форматирует результат SQL запроса в контекст для промпта."""
    tables_str = ", ".join(selected_tables) if selected_tables else "неизвестно"
    if not sql_result:
        return f"## Данные из базы данных (таблицы: {tables_str})\n\nДанные не найдены. НЕ ДОДУМЫВАЙ ответ — сообщи пользователю что информация отсутствует."
    parts = [f"## Данные из базы данных (таблицы: {tables_str})\n"]
    parts.append(json.dumps(sql_result, ensure_ascii=False, indent=2))
    return "\n".join(parts)


def _build_messages(state: AgentState) -> list[dict]:
    """Собирает список сообщений для финальной генерации ответа.

    Структура:
    1. System: роль + правила + извлечённый контекст (приоритет)
    2. History: предыдущие пары вопрос-ответ
    3. User: текущий вопрос
    """
    # Формируем контекст в зависимости от источника
    if state["source"] == "postgres" and state["sql_result"]:
        context = _format_sql_context(state["sql_result"], state["selected_tables"])
    elif state["fragments"]:
        context = _format_qdrant_context(state["fragments"])
    else:
        context = ""

    # Текущая дата — передаём модели чтобы она понимала актуальность данных
    today = datetime.now().strftime("%d %B %Y года, %A")
    # System промпт = роль + дата + контекст
    system_content = SYSTEM_PROMPT + f"\n\nСегодня: {today}."
    if context:
        system_content += (
            "\n\n---\n"
            "Используй следующую информацию для ответа. "
            "Она является приоритетной по отношению к любым другим знаниям.\n\n"
            + context
        )
    else:
        system_content += (
            "\n\n---\n"
            "По данному вопросу релевантная информация не найдена. "
            "Сообщи об этом пользователю вежливо."
        )

    messages: list[dict] = [{"role": "system", "content": system_content}]

    # История диалога
    messages.extend(state.get("history", []))

    # Текущий вопрос
    messages.append({"role": "user", "content": state["question"]})

    return messages


async def answer_generator_node(state: AgentState, llm: LLMClient) -> dict:
    """Генерирует финальный ответ пользователю.

    Приоритет информации:
    1. Извлечённые фрагменты (qdrant или postgres результат)
    2. История диалога (только для контекста)
    3. Параметры знаний модели (не используются для фактов)

    Returns:
        Обновление state: {"answer": "..."}
    """
    logger.info("node_answer_generator_start | trace_id=%s source=%s has_fragments=%s has_sql_result=%s", state["trace_id"], state["source"], bool(state["fragments"]), bool(state["sql_result"]))

    messages = _build_messages(state)

    try:
        answer = await asyncio.wait_for(
            llm.chat(messages, temperature=0.1),
            timeout=settings.timeout_answer_generator,
        )
    except asyncio.TimeoutError:
        raise NodeTimeoutError("answer_generator", settings.timeout_answer_generator)

    if not answer or not answer.strip():
        answer = _NO_ANSWER

    logger.info("node_answer_generator_done | trace_id=%s answer_length=%s", state["trace_id"], len(answer))

    # Определяем ссылку на источник для postgres ответов
    source_link = None
    sql_fragments: list[SearchResult] = []
    if state.get("source") == "postgres" and state.get("selected_tables"):
        source_link = _get_source_link(state["selected_tables"])
        url = source_link.get("url", "") if source_link else ""
        table_name = state["selected_tables"][0] if state["selected_tables"] else ""
        for row in state.get("sql_result", []):
            sql_fragments.append(SearchResult(
                text=json.dumps(row, ensure_ascii=False),
                source=url,
                group=table_name,
                score=1.0,
            ))

    return {
        "answer": answer,
        "source_link": source_link,
        "fragments": sql_fragments if sql_fragments else state.get("fragments", []),
    }