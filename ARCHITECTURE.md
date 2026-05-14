# MAGTU Consultant — Architecture Document

> Система автоматизированного консультирования абитуриентов  
> МГТУ им. Г.И. Носова  
> Версия: 1.0 | Статус: В разработке

---

## Оглавление

1. [Обзор системы](#1-обзор-системы)
2. [Стек технологий](#2-стек-технологий)
3. [Архитектурные решения](#3-архитектурные-решения)
4. [Структура проекта](#4-структура-проекта)
5. [Компоненты инфраструктуры](#5-компоненты-инфраструктуры)
6. [Агентный граф (LangGraph)](#6-агентный-граф-langgraph)
7. [Схема данных](#7-схема-данных)
8. [API контракт](#8-api-контракт)
9. [Конфигурация и переменные окружения](#9-конфигурация-и-переменные-окружения)
10. [Тестирование](#10-тестирование)
11. [Запуск и деплой](#11-запуск-и-деплой)
12. [Узкие места и решения](#12-узкие-места-и-решения)

---

## 1. Обзор системы

RAG-чатбот для приёмной комиссии МГТУ, отвечающий на вопросы абитуриентов.

### Пользовательский путь

```
Вопрос пользователя (user_id + question)
    │
    ▼
Загрузка истории диалога из PostgreSQL
(последние 5 пар вопрос-ответ)
    │
    ▼
LangGraph Agent
    │
    ├─── [tool: search_qdrant] ──────────────────────────┐
    │    Неструктурированные данные:                      │
    │    справочная информация, правила приёма, FAQ        │
    │                                                      │
    └─── [tool: query_postgres] ────────────────────────┐ │
         Структурированные данные:                       │ │
         проходные баллы, специальности, экзамены        │ │
              │                                          │ │
              ▼                                          │ │
         Выбор таблицы (LLM)                             │ │
              │                                          │ │
              ▼                                          │ │
         Сэмпл данных:                                   │ │
         SELECT * ORDER BY RANDOM() LIMIT 10             │ │
         SELECT DISTINCT <name_col> LIMIT 30             │ │
              │                                          │ │
              ▼                                          │ │
         Генерация SQL (LLM)                             │ │
              │                                          │ │
              ▼                                          │ │
         Валидация SQL (sqlglot, только SELECT)          │ │
              │                                          │ │
              ▼                                          │ │
         Выполнение запроса                              │ │
              │                                          │ │
         При ошибке: retry x3 ──► fallback ─────────────┘ │
                                                           │
    ◄──────────────────────────────────────────────────────┘
    │
    ▼
Генерация финального ответа (LLM)
Приоритет: 1) retrieved fragments, 2) история диалога
    │
    ▼
Сохранение пары вопрос-ответ в PostgreSQL
    │
    ▼
Ответ: { answer, fragments, source, metadata }
```

---

## 2. Стек технологий

| Слой | Технология | Версия |
|---|---|---|
| API | FastAPI + Pydantic v2 | >=0.110 |
| Агент | LangGraph + LangChain | >=0.2 |
| LLM Inference | vLLM | >=0.5 |
| LLM Модель | Qwen2.5-32B-Instruct-AWQ | 4bit |
| Embedding Модель | deepvk/USER-bge-m3 | — |
| Vector Store | Qdrant | >=1.9 |
| База данных | PostgreSQL | >=16 |
| Миграции | Alembic + SQLAlchemy | — |
| Кэш / Очередь | Redis | >=7 |
| SQL Валидация | sqlglot | — |
| RAG Evaluation | RAGAS + RAGChecker + BERTScore | — |
| Нагрузочное тестирование | Locust | — |
| Контейнеризация | Docker + Docker Compose | — |
| Reverse Proxy | Nginx | — |

---

## 3. Архитектурные решения

### 3.1 Разделение источников знаний

**Проблема:** Названия специальностей семантически похожи ("Прикладная информатика" vs "Прикладная математика и информатика"), поэтому чистый векторный поиск даёт нерелевантные результаты для числовых/структурированных запросов.

**Решение:**
- **Qdrant** — неструктурированные данные: справочная информация, правила приёма, FAQ, описания. Один документ = один чанк (пункт правил, FAQ-запись и т.д.)
- **PostgreSQL** — структурированные данные: проходные баллы, вступительные испытания, квоты, специальности. 9 таблиц в CSV формате.

### 3.2 Text-to-SQL с сэмплингом данных

**Проблема:** LLM генерирует SQL не зная реальных значений в таблице → `WHERE name = 'Прикладная информатика'` вместо правильного `ILIKE '%Прикладная информатика%'`.

**Решение:** Перед генерацией SQL выполнять два запроса:
```sql
-- Живой срез данных (структура строк)
SELECT * FROM table_name ORDER BY RANDOM() LIMIT 10;

-- Уникальные значения name-колонок (для точного матчинга)
SELECT DISTINCT name_column FROM table_name LIMIT 30;
```
Результаты передаются в промпт вместе со схемой таблицы.

### 3.3 Промпты для таблиц

Для каждой из 9 таблиц — отдельное описание в `prompts/tables/schemas.json`:
```json
{
  "table_name": {
    "description": "Описание таблицы",
    "columns": { "col_name": "описание колонки" },
    "name_columns": ["col1"],
    "query_examples": ["SELECT ..."],
    "rules": ["Всегда использовать ILIKE для поиска по названию", "..."]
  }
}
```

### 3.4 История диалога

- Хранится в PostgreSQL, таблица `chat_history`
- На пользователя: **5 последних пар** вопрос-ответ
- При < 5 парах — передаётся весь контекст
- Приоритет при генерации ответа: retrieved fragments > история

### 3.5 Управление нагрузкой на vLLM

- **Max concurrent requests: 5** (семафор в Redis)
- Превышение лимита → запрос встаёт в очередь
- **Timeout per request: 60 секунд** (при предыдущем тестировании: avg 5s, max 9s)
- Per-node таймауты в LangGraph (см. раздел 6)

### 3.6 Circuit Breaker

Если vLLM не отвечает N раз подряд → не посылать новые запросы, сразу возвращать `503`. Реализован через `tenacity` + счётчик в Redis.

### 3.7 Два инстанса vLLM

| Инстанс | Модель | Порт | Устройство |
|---|---|---|---|
| LLM | Qwen2.5-32B-AWQ | 8000 | GPU (RTX 3090) |
| Embedder | deepvk/USER-bge-m3 | 8001 | CPU или GPU (через .env) |

Переключение устройства для embedder: `EMBEDDING_DEVICE=cpu|cuda` в `.env`.

### 3.8 Fallback chain

```
SQL ошибка → retry (попытка 1)
           → retry (попытка 2)
           → retry (попытка 3)
           → fallback: search_qdrant
           → если Qdrant тоже не нашёл: "Не могу найти ответ на ваш вопрос"
```

---

## 4. Структура проекта

```
magtu_consultant/
│
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── routes/
│   │       │   ├── chat.py              # POST /v1/chat — основной эндпоинт
│   │       │   ├── admin.py             # Admin endpoints (ingestion, управление)
│   │       │   └── health.py            # GET /health — проверка всех сервисов
│   │       └── schemas/
│   │           ├── chat.py              # ChatRequest, ChatResponse, Fragment
│   │           └── admin.py             # IngestRequest, AdminResponse
│   │
│   ├── core/
│   │   ├── config.py                    # pydantic-settings, все env vars
│   │   ├── logging.py                   # structured JSON logger с trace_id
│   │   ├── dependencies.py              # FastAPI DI (get_db, get_qdrant, get_redis)
│   │   └── exceptions.py               # кастомные исключения + handlers
│   │
│   ├── services/
│   │   ├── agent/
│   │   │   ├── graph.py                 # LangGraph граф — сборка и компиляция
│   │   │   ├── state.py                 # AgentState (TypedDict)
│   │   │   ├── nodes/
│   │   │   │   ├── router.py            # node: выбор источника (qdrant/postgres)
│   │   │   │   ├── qdrant_search.py     # node: векторный поиск
│   │   │   │   ├── table_selector.py    # node: выбор таблицы из 9 доступных
│   │   │   │   ├── sample_data.py       # node: SELECT RANDOM LIMIT 10 + DISTINCT
│   │   │   │   ├── sql_generator.py     # node: генерация SQL (только SELECT)
│   │   │   │   ├── sql_validator.py     # node: валидация через sqlglot
│   │   │   │   ├── sql_executor.py      # node: выполнение запроса в PostgreSQL
│   │   │   │   └── answer_generator.py  # node: финальный ответ пользователю
│   │   │   └── edges.py                 # условные переходы: retry logic, fallback
│   │   │
│   │   ├── llm/
│   │   │   ├── client.py                # vLLM OpenAI-compatible client (async)
│   │   │   └── circuit_breaker.py       # Circuit breaker через Redis + tenacity
│   │   │
│   │   ├── embedding/
│   │   │   └── client.py                # vLLM embedding client (async)
│   │   │
│   │   └── history/
│   │       └── service.py               # загрузка/сохранение 5 пар вопрос-ответ
│   │
│   ├── repositories/
│   │   ├── qdrant.py                    # все операции с Qdrant
│   │   └── postgres.py                  # история + выполнение SQL запросов
│   │
│   ├── models/
│   │   └── history.py                   # SQLAlchemy модель chat_history
│   │
│   ├── prompts/
│   │   ├── system.py                    # системный промпт (роль, стиль, правила продаж)
│   │   ├── router.py                    # промпт для выбора источника
│   │   ├── sql.py                       # промпт для генерации SQL
│   │   └── tables/
│   │       └── schemas.json             # описания 9 таблиц: колонки, примеры, нюансы
│   │
│   └── main.py                          # FastAPI app factory
│
├── infra/
│   ├── docker-compose.yml               # продакшн compose
│   ├── docker-compose.override.yml      # dev overrides (volumes, hot reload)
│   ├── nginx/
│   │   └── nginx.conf                   # reverse proxy конфиг
│   └── vllm/
│       ├── start_llm.sh                 # запуск LLM инстанса (порт 8000)
│       └── start_embedder.sh            # запуск embedder инстанса (порт 8001)
│
├── scripts/
│   ├── ingest_qdrant.py                 # загрузка JSON → Qdrant (с батчингом)
│   └── ingest_postgres.py               # загрузка CSV → PostgreSQL (9 таблиц)
│
├── alembic/
│   ├── env.py
│   └── versions/
│       └── 001_create_chat_history.py   # первая миграция
│
├── tests/
│   ├── unit/
│   │   ├── test_sql_validator.py        # тесты sqlglot валидации
│   │   └── test_agent_nodes.py          # тесты нод LangGraph (mock LLM)
│   │
│   ├── integration/
│   │   └── test_chat_endpoint.py        # тесты /v1/chat эндпоинта
│   │
│   ├── rag_eval/
│   │   ├── evaluator.py                 # оркестратор: прогон вопросов, сбор ответов
│   │   ├── metrics/
│   │   │   ├── ragas_metrics.py         # faithfulness, answer_relevancy, context_recall
│   │   │   ├── bertscore_metric.py      # BERTScore F1 (ответ vs эталон)
│   │   │   └── ragchecker_metric.py     # RAGChecker: claim-level проверка
│   │   ├── data/
│   │   │   └── test_questions.json      # вопрос + эталонный ответ (твой файл)
│   │   └── reports/                     # результаты прогонов (в .gitignore)
│   │
│   └── load/
│       ├── locustfile.py                # сценарии нагрузочного тестирования
│       └── config.py                    # параметры: users, spawn_rate, host
│
├── data/
│   ├── qdrant/
│   │   └── documents.json               # исходные чанки для Qdrant
│   └── postgres/
│       ├── specialties.csv
│       ├── passing_scores.csv
│       └── ...                          # остальные 7 таблиц
│
├── .env.example                         # шаблон переменных окружения
├── .gitignore
├── pyproject.toml                       # зависимости + инструменты
├── Makefile                             # команды: make up, make ingest, make eval
└── README.md                            # инструкция по запуску для администратора
```

---

## 5. Компоненты инфраструктуры

### docker-compose сервисы

| Сервис | Образ | Порт | Описание |
|---|---|---|---|
| `api` | `./` (build) | 8080 | FastAPI приложение |
| `vllm_llm` | `vllm/vllm-openai` | 8000 | LLM инстанс (Qwen) |
| `vllm_embedder` | `vllm/vllm-openai` | 8001 | Embedder инстанс (USER-bge-m3) |
| `qdrant` | `qdrant/qdrant` | 6333 | Векторная БД |
| `postgres` | `postgres:16` | 5432 | Реляционная БД |
| `redis` | `redis:7-alpine` | 6379 | Очередь + circuit breaker |
| `nginx` | `nginx:alpine` | 80/443 | Reverse proxy |

### Параметры vLLM LLM инстанса

```bash
--model Qwen/Qwen2.5-32B-Instruct-AWQ
--quantization awq
--gpu-memory-utilization 0.85
--max-model-len 16384
--max-num-seqs 5          # max concurrent requests
--port 8000
```

### Параметры vLLM Embedder инстанса

```bash
--model deepvk/USER-bge-m3
--task embed
--port 8001
# CPU: CUDA_VISIBLE_DEVICES="" (через .env EMBEDDING_DEVICE=cpu)
# GPU: без ограничений
```

---

## 6. Агентный граф (LangGraph)

### Состояние (AgentState)

```python
class AgentState(TypedDict):
    user_id: str
    question: str
    history: list[dict]           # 5 пар вопрос-ответ
    source: str | None            # "qdrant" | "postgres"
    selected_table: str | None
    table_sample: str | None      # результат RANDOM LIMIT 10
    sql_query: str | None
    sql_result: list[dict] | None
    fragments: list[Fragment]     # retrieved chunks
    answer: str | None
    retry_count: int              # счётчик retry для SQL
    error: str | None
    trace_id: str
```

### Ноды и таймауты

| Нода | Таймаут | Описание |
|---|---|---|
| `load_history` | 3s | Загрузка истории из PostgreSQL |
| `router` | 10s | LLM выбирает tool: qdrant/postgres |
| `table_selector` | 10s | LLM выбирает таблицу из schemas.json |
| `sample_data` | 5s | SELECT RANDOM LIMIT 10 + DISTINCT |
| `sql_generator` | 15s | LLM генерирует SQL запрос |
| `sql_validator` | 1s | sqlglot парсинг (только SELECT) |
| `sql_executor` | 5s | Выполнение SELECT в PostgreSQL |
| `qdrant_search` | 5s | Векторный поиск top-k с score |
| `answer_generator` | 20s | LLM генерирует финальный ответ |
| `save_history` | 3s | Сохранение пары вопрос-ответ |

### Условные переходы (edges)

```
router → "qdrant"   → qdrant_search → answer_generator
router → "postgres" → table_selector → sample_data → sql_generator
                                                          │
                                                    sql_validator
                                                          │
                                               [valid] sql_executor
                                                          │
                                           [success] → answer_generator
                                           [error, retry<3] → sql_generator (retry_count+1)
                                           [error, retry=3] → qdrant_search (fallback)

qdrant_search → [found] answer_generator
             → [empty] answer_generator (с флагом "не найдено")
```

---

## 7. Схема данных

### PostgreSQL: таблица chat_history

```sql
CREATE TABLE chat_history (
    id          BIGSERIAL PRIMARY KEY,
    user_id     VARCHAR(255) NOT NULL,
    question    TEXT NOT NULL,
    answer      TEXT NOT NULL,
    source      VARCHAR(50),          -- 'qdrant' | 'postgres'
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_chat_history_user_id ON chat_history(user_id);
CREATE INDEX idx_chat_history_created_at ON chat_history(created_at DESC);
```

### Qdrant: схема payload

```json
{
  "text": "1.1. Приём на обучение осуществляется...",
  "source": "https://magtu.ru/pk/pravila-priema",
  "group": "Правила приёма"
}
```

### Qdrant: коллекция

- Векторная размерность: 1024 (USER-bge-m3)
- Метрика: Cosine
- Индекс: HNSW (по умолчанию)

### JSON структура для Qdrant ingestion

```json
[
  {
    "text": "...",
    "source": "https://...",
    "group": "Правила приёма"
  }
]
```

### JSON структура schemas.json (промпты таблиц)

```json
{
  "specialties": {
    "description": "Таблица специальностей и направлений подготовки",
    "columns": {
      "id": "Идентификатор",
      "name": "Полное название специальности",
      "code": "Код направления (например, 09.03.03)",
      "faculty": "Факультет",
      "level": "Уровень: бакалавриат/магистратура/специалитет"
    },
    "name_columns": ["name"],
    "query_examples": [
      "SELECT * FROM specialties WHERE name ILIKE '%информатика%'",
      "SELECT * FROM specialties WHERE faculty = 'ИТ'"
    ],
    "rules": [
      "Всегда использовать ILIKE для поиска по названию специальности",
      "Только SELECT запросы"
    ]
  }
}
```

---

## 8. API контракт

### POST /v1/chat

**Request:**
```json
{
  "user_id": "123456789",
  "question": "Какой проходной балл на прикладную информатику?"
}
```

**Response:**
```json
{
  "answer": "Проходной балл на специальность 'Прикладная информатика в экономике' в 2024 году составил 189 баллов...",
  "source": "postgres",
  "fragments": [
    {
      "text": "...",
      "source": "https://magtu.ru/...",
      "group": "Проходные баллы",
      "score": 0.91
    }
  ],
  "trace_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### GET /health

**Response:**
```json
{
  "status": "ok",
  "services": {
    "vllm_llm": "ok",
    "vllm_embedder": "ok",
    "qdrant": "ok",
    "postgres": "ok",
    "redis": "ok"
  }
}
```

### Admin endpoints (POST /v1/admin/*)

| Endpoint | Описание |
|---|---|
| `POST /v1/admin/ingest/qdrant` | Запуск ingestion JSON → Qdrant |
| `POST /v1/admin/ingest/postgres` | Запуск ingestion CSV → PostgreSQL |
| `GET /v1/admin/stats` | Статистика запросов |

---

## 9. Конфигурация и переменные окружения

Полный список в `.env.example`:

```bash
# App
APP_ENV=production                        # development | production
LOG_LEVEL=INFO

# vLLM — LLM
VLLM_LLM_BASE_URL=http://vllm_llm:8000/v1
VLLM_LLM_MODEL=Qwen/Qwen2.5-32B-Instruct-AWQ
VLLM_LLM_MAX_TOKENS=16384
VLLM_LLM_MAX_CONCURRENT=5
VLLM_LLM_TIMEOUT=60

# vLLM — Embedder
VLLM_EMBEDDER_BASE_URL=http://vllm_embedder:8001/v1
VLLM_EMBEDDER_MODEL=deepvk/USER-bge-m3
EMBEDDING_DEVICE=cpu                      # cpu | cuda

# Qdrant
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_COLLECTION=magtu_documents

# PostgreSQL
POSTGRES_DSN=postgresql+asyncpg://user:password@postgres:5432/magtu_consultant

# Redis
REDIS_URL=redis://redis:6379/0

# Circuit Breaker
CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
CIRCUIT_BREAKER_RECOVERY_TIMEOUT=30

# Retrieval
QDRANT_TOP_K=5
SQL_MAX_RETRIES=3
SQL_SAMPLE_LIMIT=10
SQL_DISTINCT_LIMIT=30

# History
HISTORY_MAX_PAIRS=5
```

---

## 10. Тестирование

### Структура тестов

```
tests/
├── unit/            pytest — быстрые тесты без внешних зависимостей
├── integration/     pytest — требуют запущенных сервисов (PostgreSQL, Qdrant)
├── rag_eval/        RAG качество — RAGAS + RAGChecker + BERTScore
└── load/            Нагрузочное — Locust
```

### RAG метрики

| Метрика | Инструмент | Нужен эталон | Что проверяет |
|---|---|---|---|
| `faithfulness` | RAGAS | Нет | Нет ли галлюцинаций относительно chunks |
| `answer_relevancy` | RAGAS | Нет | Отвечает ли ответ на вопрос |
| `context_recall` | RAGAS | Да | Содержат ли chunks нужную информацию |
| `BERTScore F1` | bert-score | Да | Семантическое сходство с эталоном |
| Claim coverage | RAGChecker | Да | Покрытие claims из эталона в ответе |

RAGAS использует тот же vLLM инстанс для evaluation.

### Формат тестовых данных (test_questions.json)

```json
[
  {
    "question": "Какой проходной балл на прикладную информатику?",
    "expected_answer": "Проходной балл на специальность...",
    "expected_source": "postgres"
  }
]
```

### Нагрузочное тестирование (Locust)

- Целевые показатели: avg < 10s, max < 60s, error rate < 1%
- Сценарий: 5 concurrent users, постепенный ramp-up
- Эндпоинт: `POST /v1/chat`

---

## 11. Запуск и деплой

### Быстрый старт для администратора

```bash
# 1. Клонировать репозиторий
git clone <repo_url>
cd magtu_consultant

# 2. Настроить окружение
cp .env.example .env
# Отредактировать .env (пути к моделям, пароли БД)

# 3. Поднять инфраструктуру
docker compose up -d

# 4. Применить миграции
docker compose exec api alembic upgrade head

# 5. Загрузить данные
docker compose exec api python scripts/ingest_postgres.py
docker compose exec api python scripts/ingest_qdrant.py

# 6. Проверить здоровье системы
curl http://localhost/health
```

### Команды Makefile

```makefile
make up          # docker compose up -d
make down        # docker compose down
make logs        # docker compose logs -f api
make migrate     # alembic upgrade head
make ingest      # ingest_postgres + ingest_qdrant
make test        # pytest tests/unit tests/integration
make eval        # python tests/rag_eval/evaluator.py
make load-test   # locust -f tests/load/locustfile.py
```

---

## 12. Узкие места и решения

| Проблема | Решение |
|---|---|
| Похожие названия специальностей в векторном поиске | PostgreSQL для структурированных запросов |
| LLM не знает реальных значений в таблице | SELECT RANDOM LIMIT 10 + DISTINCT перед SQL генерацией |
| Некорректный SQL от LLM | sqlglot валидация + retry x3 + fallback Qdrant |
| Перегрузка vLLM | Redis семафор, max 5 concurrent, очередь |
| vLLM недоступен | Circuit breaker (tenacity + Redis счётчик) |
| Зависание запроса | Per-node таймауты в LangGraph |
| Модель отвечает на провокационные вопросы | Системный промпт с явными запретами + роль |
| Галлюцинации | Приоритет retrieved fragments над историей в промпте |

---

*Документ обновляется по мере разработки.*