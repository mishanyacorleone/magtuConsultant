# MAGTU Consultant

Система автоматизированного консультирования абитуриентов МГТУ им. Г.И. Носова.

---

## Требования

- Docker Engine >= 24 + Docker Compose >= 2.24
- NVIDIA GPU (RTX 3090, 24GB VRAM) + NVIDIA Container Toolkit
- CUDA >= 12.1
- ~50GB свободного места на диске (модели + данные)

### Установка NVIDIA Container Toolkit (Ubuntu/Debian)

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## Быстрый старт

```bash
# 1. Клонировать репозиторий
git clone <repo_url> && cd magtu_consultant

# 2. Настроить .env
cp .env.example .env
# Открыть .env и заполнить: POSTGRES_PASSWORD, HF_CACHE_DIR

# 3. Разместить данные
#    data/qdrant/documents.json  — неструктурированные документы
#    data/postgres/*.csv         — таблицы (9 файлов)

# 4. Запустить всё одной командой
make setup
```

> При первом запуске vLLM скачает модели с HuggingFace (~30GB). Нужно стабильное соединение.

---

## Проверка

```bash
curl http://localhost/health
```

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

---

## Использование API

```bash
curl -X POST http://localhost/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "123456789", "question": "Какой проходной балл на прикладную информатику?"}'
```

**Ответ:**
```json
{
  "answer": "Проходной балл на специальность 'Прикладная информатика в экономике' составил 189 баллов...",
  "source": "postgres",
  "fragments": [
    {"text": "...", "source": "https://magtu.ru/...", "group": "Проходные баллы", "score": 0.91}
  ],
  "trace_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

---

## Управление данными

```bash
make ingest-qdrant    # обновить документы в Qdrant (после замены data/qdrant/documents.json)
make ingest-postgres  # обновить таблицы в PostgreSQL (после замены CSV)
make restart          # перезапустить API (после изменения промптов)
```

---

## Конфигурация embedder: CPU vs GPU

По умолчанию embedder работает на CPU. Для перевода на GPU в `.env`:

```bash
EMBEDDING_DEVICE=cuda
EMBEDDING_CUDA_VISIBLE_DEVICES=0
```

Затем: `docker compose restart vllm_embedder`

---

## Все команды

```
make help           — список команд
make up             — запустить
make down           — остановить
make logs           — логи API
make migrate        — применить миграции
make ingest         — загрузить все данные
make test           — unit + integration тесты
make eval           — RAG quality evaluation
make load-test      — нагрузочное тестирование
make shell          — shell внутри контейнера
```

---

## Архитектура

Подробная документация: [ARCHITECTURE.md](ARCHITECTURE.md)