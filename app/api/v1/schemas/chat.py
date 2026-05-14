from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: str = Field(..., description="Идентификатор пользователя (например, Telegram user_id)")
    question: str = Field(..., min_length=1, max_length=4096, description="Вопрос пользователя")


class Fragment(BaseModel):
    text: str
    source: str = Field(description="Ссылка-источник фрагмента")
    group: str = Field(description="Группа документа (например, 'Правила приёма')")
    score: float = Field(description="Score релевантности от 0 до 1")


class SourceLink(BaseModel):
    url: str = Field(description="Ссылка на страницу сайта МГТУ")
    title: str = Field(description="Название раздела (например, 'Проходные баллы прошлых лет')")


class ChatResponse(BaseModel):
    answer: str
    source: str = Field(description="Источник ответа: 'qdrant' | 'postgres'")
    fragments: list[Fragment] = Field(default_factory=list)
    source_link: SourceLink | None = Field(
        default=None,
        description="Ссылка на страницу сайта МГТУ — заполняется при source=postgres"
    )
    trace_id: str