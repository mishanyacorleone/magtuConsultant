from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.history import ChatHistory

logger = get_logger(__name__)
settings = get_settings()


class ChatHistoryRepository:
    """Репозиторий истории диалогов пользователей."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_history(self, user_id: str) -> list[dict]:
        """Возвращает последние HISTORY_MAX_PAIRS пар вопрос-ответ пользователя.

        Записи отсортированы от старых к новым, чтобы LLM читал хронологически.
        """
        max_pairs = settings.history_max_pairs

        # Берём последние N записей (по убыванию), затем разворачиваем
        subq = (
            select(ChatHistory)
            .where(ChatHistory.user_id == user_id)
            .order_by(ChatHistory.created_at.desc())
            .limit(max_pairs)
            .subquery()
        )
        stmt = select(subq).order_by(subq.c.created_at.asc())

        result = await self._session.execute(stmt)
        rows = result.mappings().all()

        return _rows_to_messages(rows)

    async def save_history(
        self,
        user_id: str,
        question: str,
        answer: str,
        source: str | None = None,
    ) -> None:
        """Сохраняет пару вопрос-ответ в историю."""
        record = ChatHistory(
            user_id=user_id,
            question=question,
            answer=answer,
            source=source,
        )
        self._session.add(record)
        await self._session.commit()
        logger.debug("history_saved | user_id=%s source=%s", user_id, source)


def _rows_to_messages(rows: list) -> list[dict]:
    """Конвертирует строки БД в формат сообщений для LLM."""
    messages = []
    for row in rows:
        messages.append({"role": "user", "content": row["question"]})
        messages.append({"role": "assistant", "content": row["answer"]})
    return messages


class SQLRepository:
    """Репозиторий для выполнения сгенерированных SELECT запросов.

    Принимает только SELECT. Все остальные запросы отклоняются
    на уровне валидатора (sqlglot), но здесь добавлен второй рубеж.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def execute_select(self, sql: str) -> list[dict]:
        """Выполняет SELECT запрос и возвращает результат как список словарей.

        Args:
            sql: Валидированный SELECT запрос.

        Returns:
            Список строк в виде словарей {колонка: значение}.

        Raises:
            ValueError: Если запрос не является SELECT.
            Exception: При ошибке выполнения запроса.
        """
        normalized = sql.strip().upper()
        if not normalized.startswith("SELECT"):
            raise ValueError(f"Only SELECT queries are allowed, got: {sql[:50]}")

        try:
            result = await self._session.execute(text(sql))
            rows = result.mappings().all()
        except Exception:
            await self._session.rollback()
            raise

        logger.debug("sql_executed | sql=%s rows_count=%s", sql[:200], len(rows))

        return [dict(row) for row in rows]

    async def sample_table(self, table_name: str, limit: int) -> list[dict]:
        """Возвращает случайный сэмпл строк из таблицы.

        Используется для передачи в промпт перед генерацией SQL,
        чтобы LLM понимал реальные значения в таблице.
        """
        # Имя таблицы валидируем: только буквы, цифры, подчёркивания
        if not table_name.replace("_", "").isalnum():
            raise ValueError(f"Invalid table name: {table_name!r}")

        sql = f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT {limit}"  # noqa: S608
        try:
            result = await self._session.execute(text(sql))
            rows = result.mappings().all()
        except Exception:
            await self._session.rollback()
            raise

        logger.debug("table_sampled | table=%s rows_count=%s", table_name, len(rows))
        return [dict(row) for row in rows]

    async def get_distinct_values(
        self, table_name: str, column_name: str, limit: int
    ) -> list[str]:
        """Возвращает уникальные значения из указанной колонки.

        Используется для name-колонок, чтобы LLM видел реальные
        названия специальностей и мог точно сформировать WHERE условие.
        """
        if not table_name.replace("_", "").isalnum():
            raise ValueError(f"Invalid table name: {table_name!r}")
        if not column_name.replace("_", "").isalnum():
            raise ValueError(f"Invalid column name: {column_name!r}")

        sql = (  # noqa: S608
            f"SELECT DISTINCT {column_name} FROM {table_name} "
            f"WHERE {column_name} IS NOT NULL "
            f"ORDER BY {column_name} "
            f"LIMIT {limit}"
        )
        result = await self._session.execute(text(sql))
        rows = result.scalars().all()

        logger.debug("distinct_values_fetched | table=%s column=%s count=%s", table_name, column_name, len(rows))
        return [str(row) for row in rows]