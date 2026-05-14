import time

import redis.asyncio as aioredis

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Redis ключи
_KEY_FAILURES = "circuit_breaker:failures"
_KEY_STATE = "circuit_breaker:state"    # closed | open | half_open
_KEY_OPENED_AT = "circuit_breaker:opened_at"


class CircuitBreaker:
    """
    Circuit breaker для vLLM на базе Redis.
 
    Состояния:
      closed    — всё работает, запросы проходят
      open      — vLLM упал, запросы блокируются немедленно
      half_open — пробный запрос после recovery_timeout
 
    Переходы:
      closed → open:      failures >= threshold
      open → half_open:   прошло recovery_timeout секунд
      half_open → closed: успешный запрос
      half_open → open:   неуспешный запрос
    """

    def __init__(self, redis: aioredis.Redis) -> None:
        self._redis = redis
        self._threshold = settings.circuit_breaker_failure_threshold
        self._recovery_timeout = settings.circuit_breaker_recovery_timeout

    async def is_available(self) -> bool:
        """Возвращает True если запрос к LLM можно отправлять."""
        state = await self._get_state()

        if state == "closed":
            return True
        
        if state == "open":
            opened_at = await self._redis.get(_KEY_OPENED_AT)
            if opened_at and (time.time() - float(opened_at)) >= self._recovery_timeout:
                await self._set_state("half_open")
                logger.info("circuit_breaker_half_open")
                return True
            return False
        
        if state == "half_open":
            return True
        
        return True
    
    async def record_success(self) -> None:
        """Регистрирует успешный запрос."""
        state = await self._get_state()

        if state == "half_open":
            # Восстановились — сбрасываем в closed
            await self._reset()
            logger.info("circuit_breaker_closed | reason=%s", "successful_probe")
        elif state == "closed":
            # Сбрасываем счётчик ошибок при успехе
            await self._redis.delete(_KEY_FAILURES)
        
    async def record_failure(self) -> None:
        """Регистрирует неуспешный запрос."""
        state = await self._get_state()

        if state == "half_open":
            # Проба не удалась — снова открываем
            await self._open()
            logger.warning("circuit_breaker_opened | reason=%s", "failed_probe")
            return
        
        failures = await self._redis.incr(_KEY_FAILURES)
        logger.warning("circuit_breaker_failure_recorded | failures=%s threshold=%s", failures, self._threshold)

        if failures >= self._threshold:
            await self._open()
            logger.error("circuit_breaker_opened | reason=%s failures=%s", "threshold_reached", failures)

    async def _get_state(self) -> str:
        state = await self._redis.get(_KEY_STATE)
        return state or "closed"
    
    async def _set_state(self, state: str) -> None:
        await self._redis.set(_KEY_STATE, state)

    async def _open(self) -> None:
        pipe = self._redis.pipeline()
        pipe.set(_KEY_STATE, "open")
        pipe.set(_KEY_OPENED_AT, str(time.time()))
        await pipe.execute()

    async def _reset(self) -> None:
        pipe = self._redis.pipeline()
        pipe.set(_KEY_STATE, "closed")
        pipe.delete(_KEY_FAILURES)
        pipe.delete(_KEY_OPENED_AT)
        await pipe.execute()

    async def get_status(self) -> dict:
        """Возвращает текущий статус для health check."""
        state = await self._get_state()
        failures = await self._redis.get(_KEY_FAILURES)
        opened_at = await self._redis.get(_KEY_OPENED_AT)

        return {
            "state": state,
            "failures": int(failures) if failures else 0,
            "threshold": self._threshold,
            "opened_at": float(opened_at) if opened_at else None
        }