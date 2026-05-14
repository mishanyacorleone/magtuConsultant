import logging
from app.core.config import get_settings
from app.core.exceptions import LLMInferenceError
from app.services.llm.circuit_breaker import CircuitBreaker
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)
settings = get_settings()


class LLMClient:
    def __init__(self, circuit_breaker: CircuitBreaker) -> None:
        self._circuit_breaker = circuit_breaker
        self._client = AsyncOpenAI(
            base_url=settings.vllm_llm_base_url,
            api_key="not-needed",
            timeout=settings.vllm_llm_timeout,
        )
        self._model = settings.vllm_llm_model

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        logger.debug("llm_request_start | messages=%s temperature=%s", len(messages), temperature)
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens or 2048,
            )
            content = response.choices[0].message.content or ""
            if response.usage:
                logger.debug("llm_request_done | completion_tokens=%s prompt_tokens=%s", response.usage.completion_tokens, response.usage.prompt_tokens)
            return content
        except Exception as exc:
            await self._circuit_breaker.record_failure()
            logger.error("llm_request_failed | error=%s", str(exc))
            raise LLMInferenceError(f"LLM inference failed: {exc}") from exc
