#!/usr/bin/env bash
set -euo pipefail

MODEL="${VLLM_MODEL:-shuyuej/Llama-3.3-70B-Instruct-GPTQ}"
API_KEY="${VLLM_API_KEY:-token-abc123}"
TP="${VLLM_TP:-1}"

exec vllm serve "${MODEL}" \
  --quantization gptq \
  --trust-remote-code \
  --dtype float16 \
  --max-model-len 4096 \
  --tensor-parallel-size "${TP}" \
  --gpu-memory-utilization 0.95 \
  --api-key "${API_KEY}"
