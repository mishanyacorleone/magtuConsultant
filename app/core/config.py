from functools import lru_cache
from typing import Literal

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_env: Literal["development", "production"] = "production"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # vLLM — LLM
    vllm_llm_port: int = 8000
    vllm_llm_model: str = "Qwen/Qwen2.5-32B-Instruct-AWQ"
    vllm_llm_max_tokens: int = 2048 # Максимальная длина ответа
    vllm_llm_max_model_len: int = 16384 # Максимальная длина контекста модели
    vllm_llm_max_concurrent: int = 5
    vllm_llm_timeout: int = 60

    @computed_field
    @property
    def vllm_llm_base_url(self) -> str:
        return f"http://vllm_llm:{self.vllm_llm_port}/v1"

    # vLLM — Embedder
    vllm_embedder_port: int = 8001
    vllm_embedder_model: str = "deepvk/USER-bge-m3"
    embedding_device: Literal["cpu", "cuda"] = "cpu"

    @computed_field
    @property
    def vllm_embedder_base_url(self) -> str:
        return f"http://vllm_embedder:{self.vllm_embedder_port}/v1"

    # Qdrant
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_collection: str = "magtu_documents"
    qdrant_top_k: int = 5

    # PostgreSQL
    postgres_user: str = "magtu"
    postgres_password: str
    postgres_db: str = "magtu_consultant"
    postgres_host: str = "postgres"
    postgres_port: int = 5432

    @computed_field
    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # Redis
    redis_url: str = "redis://redis:6379/0"

    # Circuit Breaker
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 30

    # Agent / RAG
    sql_max_retries: int = 3
    sql_sample_limit: int = 10
    sql_distinct_limit: int = 30
    history_max_pairs: int = 5

    # Node timeouts (секунды)
    timeout_load_history: int = 3
    timeout_router: int = 10
    timeout_table_selector: int = 10
    timeout_sample_data: int = 5
    timeout_sql_generator: int = 15
    timeout_sql_validator: int = 1
    timeout_sql_executor: int = 5
    timeout_qdrant_search: int = 5
    timeout_answer_generator: int = 20
    timeout_save_history: int = 3


@lru_cache
def get_settings() -> Settings:
    """Возвращает закешированный экземпляр настроек."""
    return Settings()