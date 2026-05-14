import csv
import io
import json
import logging
import os
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from sqlalchemy import text

from app.api.v1.schemas.admin import (
    AddDocumentBatchRequest,
    AddDocumentRequest,
    AddRowRequest,
    AdminResponse,
    DeleteByTextRequest,
    LoadJsonPathRequest,
    SearchRequest,
    SqlQueryRequest,
)
from app.core.dependencies import DbDep, QdrantDep
from app.models.history import ChatHistory
from app.repositories.qdrant import QdrantRepository
from sqlalchemy import select, func

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["Администрирование"])


# =============================================================================
# Qdrant — Статистика и просмотр
# =============================================================================

@router.get("/qdrant/stats", summary="Статистика коллекции Qdrant")
async def qdrant_stats(qdrant: QdrantDep) -> dict:
    """Количество документов и статус коллекции."""
    repo = QdrantRepository(qdrant)
    try:
        return await repo.get_collection_info()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/qdrant/documents", summary="Все документы с пагинацией")
async def qdrant_get_documents(
    qdrant: QdrantDep,
    limit: int = Query(20, le=100),
    offset: int = Query(0, ge=0),
) -> dict:
    """Получить документы из Qdrant с пагинацией."""
    from qdrant_client.http.models import Filter
    try:
        result = await qdrant.scroll(
            collection_name="magtu_documents",
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points = result[0]
        return {
            "total": len(points),
            "offset": offset,
            "limit": limit,
            "documents": [
                {"id": p.id, **p.payload}
                for p in points
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/qdrant/search", summary="Семантический поиск по документам")
async def qdrant_search(request: SearchRequest, qdrant: QdrantDep) -> dict:
    """Поиск документов по смысловому запросу."""
    from app.services.embedding.client import EmbeddingClient
    embedder = EmbeddingClient()
    vector = await embedder.embed(request.query)
    repo = QdrantRepository(qdrant)
    results = await repo.search(vector, top_k=request.limit)
    return {
        "query": request.query,
        "results": [
            {"text": r.text, "source": r.source, "group": r.group, "score": r.score}
            for r in results
        ],
    }


# =============================================================================
# Qdrant — Добавление документов
# =============================================================================

@router.post("/qdrant/documents", summary="Добавить один документ")
async def qdrant_add_document(
    request: AddDocumentRequest, qdrant: QdrantDep
) -> AdminResponse:
    """Добавить один документ в Qdrant."""
    from app.services.embedding.client import EmbeddingClient
    embedder = EmbeddingClient()
    vector = await embedder.embed(request.text)

    # Генерируем id как хэш текста
    import hashlib
    doc_id = int(hashlib.md5(request.text.encode()).hexdigest()[:8], 16)

    repo = QdrantRepository(qdrant)
    await repo.upsert_documents([{
        "id": doc_id,
        "vector": vector,
        "text": request.text,
        "source": request.source,
        "group": request.group,
    }])
    return AdminResponse(status="ok", message="Документ добавлен", data={"id": doc_id})


@router.post("/qdrant/documents/batch", summary="Добавить пакет документов")
async def qdrant_add_batch(
    request: AddDocumentBatchRequest, qdrant: QdrantDep
) -> AdminResponse:
    """Добавить несколько документов за один запрос."""
    from app.services.embedding.client import EmbeddingClient
    import hashlib

    embedder = EmbeddingClient()
    texts = [d["text"] for d in request.documents]
    vectors = await embedder.embed_batch(texts)

    docs = []
    for i, (doc, vector) in enumerate(zip(request.documents, vectors)):
        doc_id = int(hashlib.md5(doc["text"].encode()).hexdigest()[:8], 16)
        docs.append({
            "id": doc_id,
            "vector": vector,
            "text": doc["text"],
            "source": doc.get("source", ""),
            "group": doc.get("group", ""),
        })

    repo = QdrantRepository(qdrant)
    count = await repo.upsert_documents(docs)
    return AdminResponse(status="ok", message=f"Добавлено {count} документов", data={"count": count})


@router.post("/qdrant/documents/upload", summary="Загрузить документы из JSON файла")
async def qdrant_upload_json(
    qdrant: QdrantDep, file: UploadFile = File(...)
) -> AdminResponse:
    """
    Загрузить документы из JSON файла.
    Формат: [{"text": "...", "source": "...", "group": "..."}, ...]
    """
    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Файл должен быть .json")

    content = await file.read()
    try:
        documents = json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Невалидный JSON")

    if not isinstance(documents, list):
        raise HTTPException(status_code=400, detail="JSON должен содержать массив объектов")

    for i, doc in enumerate(documents):
        if "text" not in doc:
            raise HTTPException(status_code=400, detail=f"Объект {i} не содержит поле 'text'")

    from app.services.embedding.client import EmbeddingClient
    import hashlib

    embedder = EmbeddingClient()
    vectors = await embedder.embed_batch([d["text"] for d in documents])

    docs = []
    for doc, vector in zip(documents, vectors):
        doc_id = int(hashlib.md5(doc["text"].encode()).hexdigest()[:8], 16)
        docs.append({
            "id": doc_id,
            "vector": vector,
            "text": doc["text"],
            "source": doc.get("source", ""),
            "group": doc.get("group", ""),
        })

    repo = QdrantRepository(qdrant)
    await repo.ensure_collection_exists()
    count = await repo.upsert_documents(docs)
    return AdminResponse(
        status="ok",
        message=f"Загружено {count} документов из {file.filename}",
        data={"count": count},
    )


# =============================================================================
# Qdrant — Удаление
# =============================================================================

@router.delete("/qdrant/documents/{doc_id}", summary="Удалить документ по ID")
async def qdrant_delete_document(doc_id: int, qdrant: QdrantDep) -> AdminResponse:
    """Удалить документ по его ID."""
    try:
        await qdrant.delete(
            collection_name="magtu_documents",
            points_selector=[doc_id],
        )
        return AdminResponse(status="ok", message=f"Документ {doc_id} удалён")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/qdrant/documents/delete-by-text", summary="Удалить документы по подстроке")
async def qdrant_delete_by_text(
    request: DeleteByTextRequest, qdrant: QdrantDep
) -> AdminResponse:
    """Удалить все документы, содержащие указанную подстроку в тексте."""
    from qdrant_client.http.models import Filter, FieldCondition, MatchText
    try:
        await qdrant.delete(
            collection_name="magtu_documents",
            points_selector=Filter(
                must=[FieldCondition(key="text", match=MatchText(text=request.substring))]
            ),
        )
        return AdminResponse(
            status="ok",
            message=f"Документы с подстрокой '{request.substring}' удалены",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# =============================================================================
# Qdrant — Управление коллекцией
# =============================================================================

@router.post("/qdrant/clear", summary="Очистить коллекцию")
async def qdrant_clear(qdrant: QdrantDep) -> AdminResponse:
    """Удалить все документы и пересоздать коллекцию."""
    repo = QdrantRepository(qdrant)
    await repo.delete_all()
    return AdminResponse(status="ok", message="Коллекция очищена и пересоздана")


@router.post("/qdrant/reload", summary="Перезагрузить коллекцию из JSON файла")
async def qdrant_reload(request: LoadJsonPathRequest, qdrant: QdrantDep) -> AdminResponse:
    """Очистить коллекцию и загрузить документы из JSON файла на сервере."""
    if not os.path.exists(request.json_path):
        raise HTTPException(status_code=404, detail=f"Файл не найден: {request.json_path}")

    with open(request.json_path, encoding="utf-8") as f:
        documents = json.load(f)

    from app.services.embedding.client import EmbeddingClient
    import hashlib

    embedder = EmbeddingClient()
    vectors = await embedder.embed_batch([d["text"] for d in documents])

    docs = []
    for doc, vector in zip(documents, vectors):
        doc_id = int(hashlib.md5(doc["text"].encode()).hexdigest()[:8], 16)
        docs.append({
            "id": doc_id,
            "vector": vector,
            "text": doc["text"],
            "source": doc.get("source", ""),
            "group": doc.get("group", ""),
        })

    repo = QdrantRepository(qdrant)
    await repo.delete_all()
    count = await repo.upsert_documents(docs)
    return AdminResponse(
        status="ok",
        message=f"Коллекция перезагружена: {count} документов",
        data={"count": count},
    )


# =============================================================================
# PostgreSQL — Статистика и просмотр
# =============================================================================

@router.get("/postgres/stats", summary="Статистика всех таблиц БД")
async def postgres_stats(db: DbDep) -> dict:
    """Все таблицы, количество строк в каждой."""
    result = await db.execute(text(
        "SELECT relname, n_live_tup FROM pg_stat_user_tables ORDER BY relname"
    ))
    tables = [{"table": row[0], "rows": row[1]} for row in result.fetchall()]
    return {"tables": tables, "total_tables": len(tables)}


@router.get("/postgres/table/{table_name}", summary="Информация о таблице")
async def postgres_table_info(table_name: str, db: DbDep) -> dict:
    """Колонки, количество строк и превью первых 5 строк."""
    # Валидация имени таблицы
    if not table_name.replace("_", "").isalnum():
        raise HTTPException(status_code=400, detail="Недопустимое имя таблицы")

    try:
        # Колонки
        cols_result = await db.execute(text(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = :table ORDER BY ordinal_position"
        ), {"table": table_name})
        columns = [{"name": r[0], "type": r[1]} for r in cols_result.fetchall()]

        if not columns:
            raise HTTPException(status_code=404, detail=f"Таблица '{table_name}' не найдена")

        # Количество строк
        count_result = await db.execute(text(f'SELECT COUNT(*) FROM "{table_name}"'))  # noqa: S608
        count = count_result.scalar()

        # Превью
        preview_result = await db.execute(text(f'SELECT * FROM "{table_name}" LIMIT 5'))  # noqa: S608
        preview = [dict(row._mapping) for row in preview_result.fetchall()]

        return {
            "table": table_name,
            "columns": columns,
            "rows_count": count,
            "preview": preview,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# =============================================================================
# PostgreSQL — Загрузка CSV
# =============================================================================

@router.post("/postgres/upload-csv", summary="Загрузить CSV в таблицу")
async def postgres_upload_csv(
    db: DbDep,
    file: UploadFile = File(...),
    table_name: str | None = Query(None, description="Имя таблицы (по умолчанию — имя файла)"),
    mode: str = Query("replace", description="replace — перезаписать, append — добавить строки"),
) -> AdminResponse:
    """
    Загрузить CSV файл в PostgreSQL.
    - Разделитель: табуляция
    - mode=replace: TRUNCATE + INSERT
    - mode=append: только INSERT
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Файл должен быть .csv")
    if mode not in ("replace", "append"):
        raise HTTPException(status_code=400, detail="mode должен быть 'replace' или 'append'")

    if not table_name:
        table_name = file.filename.replace(".csv", "")

    content = await file.read()
    reader = csv.DictReader(io.StringIO(content.decode("utf-8")), delimiter="\t")
    rows = list(reader)

    if not rows:
        raise HTTPException(status_code=400, detail="CSV файл пустой")

    columns = list(rows[0].keys())
    col_str = ", ".join(f'"{c}"' for c in columns)
    placeholders = ", ".join(f":{c}" for c in columns)

    try:
        if mode == "replace":
            await db.execute(text(f'TRUNCATE TABLE "{table_name}" RESTART IDENTITY CASCADE'))  # noqa: S608

        await db.execute(
            text(f'INSERT INTO "{table_name}" ({col_str}) VALUES ({placeholders})'),  # noqa: S608
            rows,
        )
        await db.commit()
        return AdminResponse(
            status="ok",
            message=f"Загружено {len(rows)} строк в таблицу '{table_name}' (mode={mode})",
            data={"rows": len(rows), "table": table_name, "mode": mode},
        )
    except Exception as exc:
        await db.rollback()
        raise HTTPException(status_code=400, detail=str(exc))


# =============================================================================
# PostgreSQL — Управление таблицами
# =============================================================================

@router.delete("/postgres/table/{table_name}", summary="Удалить таблицу")
async def postgres_drop_table(table_name: str, db: DbDep) -> AdminResponse:
    """Полностью удаляет таблицу из БД."""
    if not table_name.replace("_", "").isalnum():
        raise HTTPException(status_code=400, detail="Недопустимое имя таблицы")
    try:
        await db.execute(text(f'DROP TABLE IF EXISTS "{table_name}" CASCADE'))  # noqa: S608
        await db.commit()
        return AdminResponse(status="ok", message=f"Таблица '{table_name}' удалена")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/postgres/table/{table_name}/rows", summary="Очистить таблицу")
async def postgres_clear_table(table_name: str, db: DbDep) -> AdminResponse:
    """Удаляет все строки, таблица остаётся."""
    if not table_name.replace("_", "").isalnum():
        raise HTTPException(status_code=400, detail="Недопустимое имя таблицы")
    try:
        result = await db.execute(text(f'DELETE FROM "{table_name}"'))  # noqa: S608
        await db.commit()
        return AdminResponse(
            status="ok",
            message=f"Таблица '{table_name}' очищена",
            data={"deleted_rows": result.rowcount},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/postgres/table/{table_name}/row", summary="Добавить строку в таблицу")
async def postgres_add_row(table_name: str, request: AddRowRequest, db: DbDep) -> AdminResponse:
    """Добавить одну строку в таблицу."""
    if not table_name.replace("_", "").isalnum():
        raise HTTPException(status_code=400, detail="Недопустимое имя таблицы")
    try:
        columns = list(request.row_data.keys())
        col_str = ", ".join(f'"{c}"' for c in columns)
        placeholders = ", ".join(f":{c}" for c in columns)
        await db.execute(
            text(f'INSERT INTO "{table_name}" ({col_str}) VALUES ({placeholders})'),  # noqa: S608
            request.row_data,
        )
        await db.commit()
        return AdminResponse(status="ok", message=f"Строка добавлена в '{table_name}'")
    except Exception as exc:
        await db.rollback()
        raise HTTPException(status_code=400, detail=str(exc))


# =============================================================================
# PostgreSQL — SQL запрос
# =============================================================================

@router.post("/postgres/query", summary="Выполнить SELECT запрос")
async def postgres_query(request: SqlQueryRequest, db: DbDep) -> dict:
    """
    Выполнить произвольный SELECT запрос к БД.
    Только SELECT — остальные операции отклоняются.
    """
    sql = request.query.strip()
    if not sql.upper().startswith("SELECT"):
        raise HTTPException(status_code=400, detail="Разрешены только SELECT запросы")
    try:
        result = await db.execute(text(sql))
        rows = [dict(row._mapping) for row in result.fetchall()]
        return {"rows": rows, "count": len(rows)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# =============================================================================
# История чатов
# =============================================================================

@router.get("/chat/stats", summary="Статистика чатов")
async def chat_stats(db: DbDep) -> dict:
    """Общая статистика по диалогам."""
    total = (await db.execute(select(func.count(ChatHistory.id)))).scalar() or 0
    qdrant_count = (await db.execute(
        select(func.count(ChatHistory.id)).where(ChatHistory.source == "qdrant")
    )).scalar() or 0
    postgres_count = (await db.execute(
        select(func.count(ChatHistory.id)).where(ChatHistory.source == "postgres")
    )).scalar() or 0

    # Топ-10 пользователей по количеству запросов
    top_users_result = await db.execute(text(
        "SELECT user_id, COUNT(*) as requests FROM chat_history "
        "GROUP BY user_id ORDER BY requests DESC LIMIT 10"
    ))
    top_users = [{"user_id": r[0], "requests": r[1]} for r in top_users_result.fetchall()]

    return {
        "total_requests": total,
        "qdrant_requests": qdrant_count,
        "postgres_requests": postgres_count,
        "top_users": top_users,
    }


@router.get("/chat/history/{user_id}", summary="История диалога пользователя")
async def chat_history(
    user_id: str,
    db: DbDep,
    limit: int = Query(20, le=100),
) -> dict:
    """Получить историю диалога конкретного пользователя."""
    result = await db.execute(
        select(ChatHistory)
        .where(ChatHistory.user_id == user_id)
        .order_by(ChatHistory.created_at.desc())
        .limit(limit)
    )
    rows = result.scalars().all()
    return {
        "user_id": user_id,
        "total": len(rows),
        "history": [
            {
                "question": r.question,
                "answer": r.answer,
                "source": r.source,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ],
    }