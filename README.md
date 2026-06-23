# LoCoMo vLLM Jasper Benchmark

This repository contains code to run LoCoMo against a local vLLM server with Mem0 retrieval backed by a vector database such as Jasper or Qdrant. On Runpod, runtime files are kept under `/workspace` so model downloads, caches, temp files, and results persist on the hosted partition.

## 1. Configure

From a Runpod shell:

```bash
cd /workspace
git clone https://github.com/saltsystemslab/benchmark-jasper.git
cd benchmark-jasper
git checkout runpod/6-22-2026
cp .env.example .env
```

Edit `.env` for your session. The common values are:

- `BENCHMARK_RUNTIME_ROOT=/workspace`
- `CUDA_MODULE=` for Runpod containers without environment modules
- `VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct`
- `VLLM_API_KEY=token-abc123`
- `OPENAI_API_KEY=...`
- `HF_TOKEN=...` (optional)

Load the environment in any shell that will run project commands:

```bash
source scripts/load_env.sh
```

## 2. Install

```bash
bash scripts/setup_remote.sh
```

## 3. Start vLLM

Use tmux so the server keeps running while the benchmark runs in another window:

```bash
tmux new -s locomo
```

In window 1:

```bash
source .venv/bin/activate
bash scripts/serve_vllm.sh
```

Create window 2 with `Ctrl-b c`, then check the server:

```bash
source .venv/bin/activate

curl --noproxy '*' \
  -H "Authorization: Bearer ${VLLM_API_KEY}" \
  "${VLLM_BASE_URL}/models"
```

## 4. Data

Place LoCoMo at `data/locomo10.json`:

```bash
mkdir -p data
curl -L \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json \
  -o data/locomo10.json
```

## 5. Precompute Embeddings

Embeddings are cached under `${BENCHMARK_CACHE_ROOT}/embeddings`, keyed by model, purpose, and exact text. Precompute before timed runs so OpenAI embedding calls are outside the measured benchmark:

```bash
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)

locomo-jasper-bench \
  --dataset data/locomo10.json \
  --results-dir "${BENCHMARK_RESULTS_ROOT}" \
  --max-samples 20 \
  --preembed-only \
  --run-id preembed-20samples-${RUN_STAMP}
```

Timed runs read from that cache and fail if an embedding is missing.

## 6. Compare Qdrant And Jasper

```bash
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)

locomo-jasper-bench \
  --dataset data/locomo10.json \
  --results-dir "${BENCHMARK_RESULTS_ROOT}" \
  --vector-backend qdrant \
  --max-samples 20 \
  --stream \
  --log-every "${LOCOMO_LOG_EVERY:-10}" \
  --run-id qdrant-20samples-${RUN_STAMP}

locomo-jasper-bench \
  --dataset data/locomo10.json \
  --results-dir "${BENCHMARK_RESULTS_ROOT}" \
  --vector-backend jasper \
  --jasper-alpha 1.0 \
  --max-samples 20 \
  --stream \
  --log-every "${LOCOMO_LOG_EVERY:-10}" \
  --run-id jasper-20samples-${RUN_STAMP}
```

## 7. GPU KV Injection

The `ai-memory-code` submodule can be used through an opt-in in-process vLLM backend. This mode keeps the current Mem0/Jasper top-k retrieval step, then composes the retrieved turns as chunked-RoPE KV tensors on GPU and injects them through a GPU connector. It forces vLLM V1 multiprocessing off so the connector can share the benchmark process's GPU memory registry.

```bash
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_ID=kv-strict-smoke-${RUN_STAMP}

locomo-jasper-bench \
  --dataset data/locomo10.json \
  --results-dir "${BENCHMARK_RESULTS_ROOT}" \
  --answer-backend vllm-kv \
  --vector-backend jasper \
  --top-k 50 \
  --kv-sample-window 1 \
  --kv-gpu-memory-utilization 0.30 \
  --kv-max-model-len 32768 \
  --kv-max-position 32768 \
  --max-samples 5 \
  --log-every 1 \
  --skip-judge \
  --measure-ttft \
  --run-id "${RUN_ID}"
```

## 8. Deferred Judging

Use deferred judging when the answer model and judge model cannot run at the same time. This works for both normal OpenAI-compatible answer runs and GPU KV runs.

After inference finishes, stop the answer model if needed, start the judge model, then judge the existing run:

```bash
locomo-jasper-bench \
  --results-dir "${BENCHMARK_RESULTS_ROOT}" \
  --run-id "${RUN_ID}" \
  --judge-only \
  --judge-base-url "${JUDGE_BASE_URL:-${VLLM_BASE_URL}}" \
  --judge-api-key "${JUDGE_API_KEY:-${VLLM_API_KEY}}" \
  --judge-model "${JUDGE_MODEL:-${VLLM_MODEL}}"
```

`--judge-only` reads `${BENCHMARK_RESULTS_ROOT}/${RUN_ID}/predictions.jsonl`, fills only rows that are still unjudged, preserves any rows already judged, and regenerates `summary.json`.

## 9. Results

Each run writes to `${BENCHMARK_RESULTS_ROOT}/<run-id>/`:

- `config.json`
- `system.json`
- `predictions.jsonl`
- `summary.json`

Read summaries:

```bash
cat "${BENCHMARK_RESULTS_ROOT}/qdrant-20samples-${RUN_STAMP}/summary.json"
cat "${BENCHMARK_RESULTS_ROOT}/jasper-20samples-${RUN_STAMP}/summary.json"
```

Primary metrics:

- `metrics.accuracy`: judged answer quality.
- `metrics.time_to_first_token_ms`: time to first answer token; populated by non-KV runs with `--stream` and KV runs with `--measure-ttft`.
- `metrics.kv_engine_time_to_first_token_ms`: vLLM engine time for the one-token KV TTFT probe; populated by KV runs with `--measure-ttft`.
- `metrics.vector_db_query_time_ms`: raw backend vector query time.
- `metrics.vector_db_queries_per_sec`: vector DB query throughput.
