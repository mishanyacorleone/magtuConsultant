import httpx
from openai import AsyncOpenAI

from app.core.config import get_settings
from app.core.exceptions import LLMInferenceError
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class EmbeddingClient:
    """Асинхронный клиент для infinity-emb.

    infinity-emb использует /embeddings без префикса /v1.
    """

    def __init__(self) -> None:
        # infinity-emb endpoint: http://host:port/embeddings (без /v1)
        base = settings.vllm_embedder_base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]

        self._client = AsyncOpenAI(
            base_url=f"{base}/v1",  # OpenAI клиент добавит /embeddings сам
            api_key="not-needed",
            timeout=30,
        )
        # Для прямых запросов используем base без /v1
        self._embeddings_url = f"{base}/embeddings"
        self._health_url = f"{base}/health"
        self._model = settings.vllm_embedder_model

    async def embed(self, text: str) -> list[float]:
        """Возвращает вектор для одного текста."""
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Векторизует список текстов одним запросом."""
        if not texts:
            return []
        try:
            # Используем httpx напрямую — обходим /v1 prefix OpenAI клиента
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    self._embeddings_url,
                    json={"input": texts, "model": self._model},
                )
                resp.raise_for_status()
                data = resp.json()

            # Сортируем по index и извлекаем векторы
            sorted_data = sorted(data["data"], key=lambda x: x["index"])
            vectors = [item["embedding"] for item in sorted_data]

            logger.debug("embedding_batch_done | texts_count=%s vector_size=%s", len(texts), len(vectors[0]))
            return vectors

        except Exception as exc:
            logger.error("embedding_batch_failed | error=%s texts_count=%s", str(exc), len(texts))
            raise LLMInferenceError(f"Batch embedding failed: {exc}") from exc

    async def health_check(self) -> bool:
        """Проверяет доступность embedding сервера."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(self._health_url)
                return resp.status_code == 200
        except Exception as exc:
            logger.warning("embedder_health_check_failed | error=%s", str(exc))
            return False