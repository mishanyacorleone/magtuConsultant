.PHONY: help up down logs migrate ingest ingest-qdrant ingest-postgres \
        test test-unit test-integration eval load-test lint shell setup

GREEN  := \033[0;32m
YELLOW := \033[0;33m
RESET  := \033[0m

help:
	@echo ""
	@echo "$(GREEN)MAGTU Consultant — доступные команды:$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""

up: ## Поднять все сервисы
	docker compose up -d
	@echo "$(GREEN)Сервисы запущены. API: http://localhost/$(RESET)"

down: ## Остановить все сервисы
	docker compose down

down-volumes: ## Остановить и удалить volumes (удаляет данные!)
	docker compose down -v

logs: ## Логи API в реальном времени
	docker compose logs -f api

logs-all: ## Логи всех сервисов
	docker compose logs -f

restart: ## Перезапустить API
	docker compose restart api

migrate: ## Применить миграции БД
	docker compose exec api alembic upgrade head

migrate-create: ## Создать миграцию: make migrate-create MSG="описание"
	docker compose exec api alembic revision --autogenerate -m "$(MSG)"

migrate-down: ## Откатить последнюю миграцию
	docker compose exec api alembic downgrade -1

ingest: ingest-postgres ingest-qdrant ## Загрузить все данные

ingest-postgres: ## Загрузить CSV → PostgreSQL
	docker compose exec api python scripts/ingest_postgres.py

ingest-qdrant: ## Загрузить JSON → Qdrant
	docker compose exec api python scripts/ingest_qdrant.py

test: test-unit test-integration ## Запустить все тесты

test-unit: ## Unit тесты
	docker compose exec api pytest tests/unit -v

test-integration: ## Integration тесты
	docker compose exec api pytest tests/integration -v

eval: ## RAG evaluation (RAGAS + BERTScore + RAGChecker)
	docker compose exec api python tests/rag_eval/evaluator.py

load-test: ## Нагрузочное тестирование (Locust UI на :8089)
	locust -f tests/load/locustfile.py --host http://localhost

load-test-headless: ## Нагрузочное тестирование без UI (5 users, 60s)
	locust -f tests/load/locustfile.py \
		--host http://localhost \
		--users 5 --spawn-rate 1 --run-time 60s --headless

lint: ## Проверить код через ruff
	ruff check app/ scripts/ tests/

lint-fix: ## Исправить ошибки ruff
	ruff check --fix app/ scripts/ tests/

shell: ## Shell внутри контейнера API
	docker compose exec api bash

setup: ## Полная первоначальная настройка
	@test -f .env || (cp .env.example .env && echo "$(YELLOW)Создан .env — заполни POSTGRES_PASSWORD и пути к моделям!$(RESET)" && exit 1)
	$(MAKE) up
	@echo "$(YELLOW)Ожидание готовности сервисов (90s)...$(RESET)"
	@sleep 90
	$(MAKE) migrate
	$(MAKE) ingest
	@echo "$(GREEN)Система готова! API: http://localhost/health$(RESET)"