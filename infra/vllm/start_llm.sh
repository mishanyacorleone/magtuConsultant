#!/usr/bin/env bash
# Запуск vLLM LLM инстанса (Qwen2.5-32B-AWQ)
set -euo pipefail

: "${VLLM_LLM_MODEL:=Qwen/Qwen2.5-32B-Instruct-AWQ}"
: "${VLLM_LLM_MAX_TOKENS:=16384}"
: "${VLLM_LLM_MAX_CONCURRENT:=5}"

echo "Starting vLLM LLM | model=${VLLM_LLM_MODEL} max_tokens=${VLLM_LLM_MAX_TOKENS} concurrent=${VLLM_LLM_MAX_CONCURRENT}"

python -m vllm.entrypoints.openai.api_server \
    --model "${VLLM_LLM_MODEL}" \
    --quantization awq \
    --gpu-memory-utilization 0.85 \
    --max-model-len "${VLLM_LLM_MAX_TOKENS}" \
    --max-num-seqs "${VLLM_LLM_MAX_CONCURRENT}" \
    --port 8000 \
    --served-model-name "${VLLM_LLM_MODEL}"