"""Скрипт загрузки документов из JSON в Qdrant.
 
Запуск:
    python scripts/ingest_qdrant.py
    python scripts/ingest_qdrant.py --data data/qdrant/documents.json --batch-size 50
 
Формат входного файла (data/qdrant/documents.json):
    [
        {
            "text": "1.1. Приём на обучение...",
            "source": "https://magtu.ru/pk/pravila-priema",
            "group": "Правила приёма"
        },
        ...
    ]
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Добавляем корень проекта в sys.path для импортов
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.repositories.qdrant import QdrantRepository
from app.services.embedding.client import EmbeddingClient
from qdrant_client import AsyncQdrantClient

setup_logging()
logger = get_logger(__name__)


async def ingest(data_path: Path, batch_size: int) -> None:
    settings = get_settings()

    logger.info("ingest_qdrant_start", path=str(data_path))

    # Загружаем данные
    with data_path.open(encoding="utf-8") as f:
        documents: list[dict] = json.load(f)

    if not documents:
        logger.warning("ingest_qdrant_empty_file")
        return
    
    logger.info("ingest_qdrant_loaded", count=len(documents))

    # Инициализируем клиенты
    qdrant_client = AsyncQdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port
    )
    qdrant_repo = QdrantRepository(qdrant_client)
    embedder = EmbeddingClient()

    # Проверяем/создаём коллекцию
    await qdrant_repo.ensure_collection_exists()

    # Обрабатываем батчами
    total = 0
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        texts = [doc["text"] for doc in batch]

        logger.info(
            "ingest_qdrant_batch",
            batch_num=i // batch_size + 1,
            batch_size=len(batch),
            total_processed=i
        )

        vectors = await embedder.embed_batch(texts)

        docs_with_vectors = [
            {
                **doc,
                "id": doc.get("id", i + j),
                "vector": vector
            }
            for j, (doc, vector) in enumerate(zip(batch, vectors))
        ]

        count = await qdrant_repo.upsert_documents(docs_with_vectors)
        total += count

    logger.info("ingest_qdrant_done", total=total)
    print("✓ Загружено {total} документов в Qdrant (коллекция: {settings.qdrant_collection}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load documents into Qdrant")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/qdrant/documents.json"),
        help="Path to JSON file with documents"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for embedding requests"
    )
    args = parser.parse_args()

    if not args.data.exists():
        print(f"Error: file not found: {args.data}")
        sys.exit(1)
    
    asyncio.run(ingest(args.data, args.batch_size))


if __name__ == "__main__":
    main()
    