from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.repositories.postgres import ChatHistoryRepository

logger = get_logger(__name__)


class HistoryService:
    """
    Сервис управления историей диалогов.
 
    Тонкая обёртка над репозиторием. Здесь можно добавить
    бизнес-логику поверх хранилища: фильтрацию, форматирование,
    ограничения без изменения репозитория.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._repo = ChatHistoryRepository(session)

    async def get_history(self, user_id: str) -> list[dict]:
        """Возвращает историю диалога в формате сообщений для LLM.
 
        Формат: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        Записи отсортированы от старых к новым (хронологически).
 
        Args:
            user_id: Идентификатор пользователя (например, Telegram user_id).
 
        Returns:
            Список пар сообщений. Пустой список если истории нет.
        """
        try:
            history = await self._repo.get_history(user_id)
            logger.debug("history_loaded | user_id=%s pairs_count=%s", user_id, len(history) // 2)
            return history
        except Exception as exc:
            # История не критична — при ошибке продолжаем без неё
            logger.warning("history_load_failed | user_id=%s error=%s", user_id, str(exc))
            return []
        
    async def save(
        self,
        user_id: str,
        question: str,
        answer: str,
        source: str | None = None
    ) -> None:
        """
        Сохраняет пару вопрос-ответ в историю.
 
        Args:
            user_id: Идентификатор пользователя.
            question: Вопрос пользователя.
            answer: Ответ системы.
            source: Источник ответа ("qdrant" | "postgres").
        """
        try:
            await self._repo.save_history(
                user_id=user_id,
                question=question,
                answer=answer,
                source=source
            )
        except Exception as exc:
            # Ошибка сохранения истории не должна ломать ответ пользователю
            logger.error("history_save_failed | user_id=%s error=%s", user_id, str(exc))
            