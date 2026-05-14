from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Index, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ChatHistory(Base):
    """
    История диалогов пользователей.

    Хранит пары вопрос-ответ. На одного пользователя используются последние
    HISTORY_MAX_PAIRS пар (настраивается в config)
    """

    __tablename__ = "chat_history"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str | None] = mapped_column(
        String(50), nullable=True, comment="qdrant | postgres"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    __table_args__ = (
        Index("idx_chat_history_user_created", "user_id", "created_at"),
    )

    def __repr__(self) -> str:
        return "<ChatHistory id={self.id} user_id={self.user_id!r}>"
