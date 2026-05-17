"""
Нода subject_matcher — подбор направлений подготовки по предметам ЕГЭ.

Срабатывает когда table_selector выбрал таблицу vi_soo_vo.
Использует structured output (json_object) для извлечения предметов,
затем фильтрует направления через Python без SQL.
"""

import asyncio
import json
import logging

from pydantic import BaseModel, field_validator

from app.core.config import get_settings
from app.core.exceptions import NodeTimeoutError
from app.prompts.subject_matcher import SUBJECT_EXTRACTOR_SYSTEM, SUBJECT_EXTRACTOR_USER
from app.services.agent.state import AgentState
from app.services.llm.client import LLMClient

logger = logging.getLogger(__name__)
settings = get_settings()

# Предметы которые нужно нормализовать к стандартным названиям
FOREIGN_LANGUAGES = {"английский", "немецкий", "французский", "испанский", "китайский",
                     "английский язык", "немецкий язык", "французский язык",
                     "испанский язык", "китайский язык"}

VALID_SUBJECTS = {
    "русский язык", "математика", "обществознание", "информатика",
    "физика", "химия", "биология", "история", "география",
    "литература", "иностранный язык",
}


class SubjectExtractionResult(BaseModel):
    subjects: list[str]
    has_base_math: bool = False

    @field_validator("subjects")
    @classmethod
    def normalize(cls, v: list[str]) -> list[str]:
        result = []
        for s in v:
            s = s.strip().lower()
            if s in FOREIGN_LANGUAGES:
                s = "иностранный язык"
            if s in VALID_SUBJECTS:
                result.append(s)
        # Русский язык всегда присутствует
        if "русский язык" not in result:
            result.insert(0, "русский язык")
        return list(dict.fromkeys(result))  # убираем дубликаты сохраняя порядок


def _matches_direction(subjects: set[str], required_vi: list[str],
                       optional_vi_ege: list[str]) -> bool:
    """
    Проверяет подходит ли направление для набора предметов абитуриента.

    Направление подходит если:
    1. Все предметы из required_vi есть у абитуриента
    2. optional_vi_ege пустой ИЛИ хотя бы один предмет из optional_vi_ege есть у абитуриента
    """
    # Все обязательные предметы должны быть у абитуриента
    if not all(vi in subjects for vi in required_vi):
        return False

    # Опциональные: если список пустой — подходит, иначе нужен хотя бы один
    if not optional_vi_ege:
        return True

    return any(vi in subjects for vi in optional_vi_ege)


async def subject_matcher_node(state: AgentState, llm: LLMClient,
                                db_session) -> dict:
    """
    Извлекает предметы ЕГЭ из вопроса и фильтрует направления подготовки.

    Returns:
        sql_result: список подходящих направлений из vi_soo_vo
        selected_tables: ["vi_soo_vo"]
    """
    logger.info("node_subject_matcher_start | trace_id=%s question=%s",
                state["trace_id"], state["question"][:80])

    # Шаг 1: извлечь предметы через LLM со structured output
    messages = [
        {"role": "system", "content": SUBJECT_EXTRACTOR_SYSTEM},
        {"role": "user", "content": SUBJECT_EXTRACTOR_USER.format(question=state["question"])},
    ]

    try:
        raw = await asyncio.wait_for(
            llm.chat(messages, temperature=0.0, max_tokens=200,
                     response_format={"type": "json_object"}),
            timeout=settings.timeout_sql_generator,
        )
        data = json.loads(raw)
        extraction = SubjectExtractionResult(**data)
    except asyncio.TimeoutError:
        raise NodeTimeoutError("subject_matcher", settings.timeout_sql_generator)
    except Exception as exc:
        logger.warning("subject_matcher_extraction_failed | error=%s", exc)
        # Fallback — только русский язык
        extraction = SubjectExtractionResult(subjects=["русский язык"])

    subjects = set(extraction.subjects)
    logger.info("node_subject_matcher_extracted | trace_id=%s subjects=%s has_base_math=%s",
                state["trace_id"], sorted(subjects), extraction.has_base_math)

    # Шаг 2: загрузить все записи из vi_soo_vo и отфильтровать в Python
    from sqlalchemy import text
    result = await db_session.execute(text('SELECT * FROM "vi_soo_vo"'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    matched = []
    for row in rows:
        required = row.get("required_vi") or []
        optional_ege = row.get("optional_vi_ege") or []

        if _matches_direction(subjects, required, optional_ege):
            matched.append(row)

    logger.info("node_subject_matcher_done | trace_id=%s subjects=%s matched=%s",
                state["trace_id"], sorted(subjects), len(matched))

    return {
        "sql_result": matched,
        "selected_tables": ["vi_soo_vo"],
        "table_sample": "",  # не нужен — данные уже отфильтрованы
    }