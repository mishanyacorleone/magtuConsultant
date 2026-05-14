from dataclasses import dataclass, field
from typing import TypedDict

from app.repositories.qdrant import SearchResult


class AgentState(TypedDict):
    """
    Состояние агентного графа LangGraph.
 
    Передаётся между нодами и обновляется на каждом шаге.
    Все поля опциональны кроме user_id, question, trace_id.
    """

    # Входные данные
    user_id: str
    question: str
    trace_id: str

    # История диалога (загружается в load_history)
    history: list[dict]

    # Решение роутера
    source: str | None  # "qdrant" | "postgres"

    # PostgreSQL path
    selected_tables: list[str] # Выбранные таблицы
    table_sample: str          # Форматированный сэмпл данных для промпта
    sql_query: str | None      # Сгенерированный SQL
    sql_result: list[dict]     # Результат выполнения запроса
    sql_error: str | None      # Последняя ошибка SQL (для retry промпта)
    retry_count: int           # счётчик попыток SQL генерации

    # Qdrant path
    fragments: list[SearchResult]

    # Финальный ответ
    answer: str | None

    # Ссылка на страницу сайта МГТУ (заполняется при postgres path)
    source_link: dict | None # {"url": "...", "title": "..."}

    # Флаг - нужно ли сохранять историю (False при ошибках)
    should_save_history: bool


def make_initial_state(user_id: str, question: str, trace_id: str) -> AgentState:
    """Создаёт начальное состояние с дефолтными значениями."""
    return AgentState(
        user_id=user_id,
        question=question,
        trace_id=trace_id,
        history=[],
        source=None,
        selected_tables=[],
        table_sample="",
        sql_query=None,
        sql_result=[],
        sql_error=None,
        retry_count=0,
        fragments=[],
        answer=None,
        source_link=None,
        should_save_history=True
    )
