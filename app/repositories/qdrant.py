from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

VECTOR_SIZE = 1024


@dataclass
class SearchResult:
    """Один результат векторного поиска."""

    text: str
    source: str
    group: str
    score: float


class QdrantRepository:
    """Репозиторий для работы с Qdrant."""

    def __init__(self, client: AsyncQdrantClient) -> None:
        self._client = client
        self._collection = settings.qdrant_collection

    async def search(
        self,
        query_vector: list[float],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Выполняет векторный поиск и возвращает top-k результатов."""
        k = top_k or settings.qdrant_top_k

        hits = await self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=k,
            with_payload=True,
        )

        results = []
        for hit in hits.points:
            payload = hit.payload or {}
            results.append(
                SearchResult(
                    text=payload.get("text", ""),
                    source=payload.get("source", ""),
                    group=payload.get("group", ""),
                    score=hit.score,
                )
            )

        logger.debug("qdrant_search_done | collection=%s top_k=%s results_count=%s top_score=%s", self._collection, k, len(results), results[0].score if results else None)

        return results

    async def upsert_documents(self, documents: list[dict]) -> int:
        """Добавляет или обновляет документы в коллекции."""
        points = [
            PointStruct(
                id=doc["id"],
                vector=doc["vector"],
                payload={
                    "text": doc["text"],
                    "source": doc["source"],
                    "group": doc["group"],
                },
            )
            for doc in documents
        ]

        await self._client.upsert(
            collection_name=self._collection,
            points=points,
        )

        logger.info("qdrant_upsert_done | count=%s", len(points))
        return len(points)

    async def ensure_collection_exists(self) -> None:
        """Создаёт коллекцию если она не существует."""
        collections = await self._client.get_collections()
        existing = {c.name for c in collections.collections}

        if self._collection not in existing:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("qdrant_collection_created | collection=%s", self._collection)
        else:
            logger.debug("qdrant_collection_exists | collection=%s", self._collection)

    async def get_collection_info(self) -> dict:
        """Возвращает информацию о коллекции для health check."""
        info = await self._client.get_collection(self._collection)
        return {
            "name": self._collection,
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": info.status.value if info.status else "unknown",
        }

    async def delete_all(self) -> None:
        """Удаляет и пересоздаёт коллекцию."""
        await self._client.delete_collection(self._collection)
        await self.ensure_collection_exists()
        logger.warning("qdrant_collection_recreated | collection=%s", self._collection)