from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Кастомные исключения
# =============================================================================


class MAGTUBaseError(Exception):
    """Базовое исключение приложения."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(message)


class LLMUnavailableError(MAGTUBaseError):
    """vLLM недоступен (circuit breaker открыт или таймаут)."""


class LLMInferenceError(MAGTUBaseError):
    """Ошибка при инференсе LLM."""


class SQLGenerationError(MAGTUBaseError):
    """Не удалось сгенерировать корректный SQL после всех retry."""


class SQLValidationError(MAGTUBaseError):
    """SQL не прошёл валидацию sqlglot."""

    def __init__(self, message: str, sql: str) -> None:
        super().__init__(message, details={"sql": sql})
        self.sql = sql


class SQLExecutionError(MAGTUBaseError):
    """Ошибка выполнения SQL запроса."""

    def __init__(self, message: str, sql: str) -> None:
        super().__init__(message, details={"sql": sql})
        self.sql = sql


class QdrantSearchError(MAGTUBaseError):
    """Ошибка поиска в Qdrant."""


class HistoryLoadError(MAGTUBaseError):
    """Не удалось загрузить историю диалога."""


class NodeTimeoutError(MAGTUBaseError):
    """Нода LangGraph превысила таймаут."""

    def __init__(self, node_name: str, timeout: int) -> None:
        super().__init__(
            f"Node '{node_name}' timed out after {timeout}s",
            details={"node": node_name, "timeout": timeout},
        )
        self.node_name = node_name


# =============================================================================
# FastAPI exception handlers
# =============================================================================


def register_exception_handlers(app: FastAPI) -> None:
    """Регистрирует обработчики исключений в FastAPI приложении."""

    @app.exception_handler(LLMUnavailableError)
    async def llm_unavailable_handler(
        request: Request, exc: LLMUnavailableError
    ) -> JSONResponse:
        logger.warning("llm_unavailable | path=%s error=%s", request.url.path, exc.message)
        return JSONResponse(
            status_code=503,
            content={
                "error": "service_unavailable",
                "message": "Сервис временно недоступен. Попробуйте позже.",
            },
        )

    @app.exception_handler(MAGTUBaseError)
    async def base_error_handler(request: Request, exc: MAGTUBaseError) -> JSONResponse:
        logger.error("application_error | path=%s error=%s details=%s", request.url.path, exc.message, exc.details)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "message": "Произошла внутренняя ошибка. Попробуйте позже.",
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_exception | path=%s error=%s", request.url.path, str(exc))
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "message": "Произошла внутренняя ошибка."},
        )