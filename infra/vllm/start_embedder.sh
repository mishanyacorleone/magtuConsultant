#!/usr/bin/env bash
# Запуск infinity-emb для embedding модели (USER-bge-m3)
# Используется внутри docker-compose.
#
# ВАЖНО: vllm/vllm-openai НЕ поддерживает CPU режим — падает при старте
# если нет CUDA. Поэтому для embedder используем infinity-emb,
# который корректно работает на CPU и GPU.
#
# Управление устройством через .env:
#   EMBEDDING_DEVICE=cpu   — CPU режим (по умолчанию)
#   EMBEDDING_DEVICE=cuda  — GPU режим

set -euo pipefail

: "${VLLM_EMBEDDER_MODEL:=deepvk/USER-bge-m3}"
: "${VLLM_EMBEDDER_PORT:=8001}"
: "${EMBEDDING_DEVICE:=cpu}"

echo "Starting infinity-emb | model=${VLLM_EMBEDDER_MODEL} device=${EMBEDDING_DEVICE} port=${VLLM_EMBEDDER_PORT}"

infinity_emb v2 \
    --model-name-or-path "${VLLM_EMBEDDER_MODEL}" \
    --port "${VLLM_EMBEDDER_PORT}" \
    --device "${EMBEDDING_DEVICE}"