# LoCoMo KV Cache Benchmarks

This repository runs a LoCoMo benchmark comparison between in-process vLLM answer backends.
On Runpod, runtime files are kept under `/workspace` so model downloads, caches, temp files, and results persist on the hosted partition.

- `vllm-kv`: retrieved memories are encoded with the package's chunked-RoPE helpers, then injected through the top-level GPU KV connector.
- `vllm-prefix`: the same retrieved memory tokens are included as a normal vLLM prompt prefix.

## 1. Configure

```bash
cp .env.example .env
```

Edit `.env` for your session. The common values are:

- `MODEL_LLAMA=meta-llama/Llama-3.1-8B-Instruct`
- `MODEL_MISTRAL=mistralai/Mistral-7B-Instruct-v0.3`
- `MODEL_QWEN=Qwen/Qwen2.5-7B-Instruct`
- `MODEL_QWEN3_14B=Qwen/Qwen3-14B`
- `LOCOMO_VLLM_MODEL=llama`
- `BENCHMARK_RUNTIME_ROOT=/workspace`
- `JUDGE_PROVIDER=vllm`
- `JUDGE_MODEL=google/gemma-2-9b-it`
- `OPENAI_JUDGE_MODEL=gpt-5.4`
- `CUDA_MODULE=` for Runpod containers without environment modules
- `OPENAI_API_KEY=...` for embeddings and OpenAI judging
- `HF_TOKEN=...` if the model is gated

By default, runtime storage is rooted at `${BENCHMARK_RUNTIME_ROOT:-/workspace}` on Runpod:

- `${BENCHMARK_RUNTIME_ROOT}/.cache` for embeddings, Mem0/Jasper files, model downloads, and build caches.
- `${BENCHMARK_RUNTIME_ROOT}/results` for benchmark outputs.
- `${BENCHMARK_RUNTIME_ROOT}/tmp` for temporary files.

`source scripts/load_env.sh` prepares those directories and points the repo `.cache` entry at the runtime cache.

```bash
export BENCHMARK_CACHE_ROOT=/path/to/cache
export BENCHMARK_RESULTS_ROOT=/path/to/results
```

Load the environment in each shell that will run project commands:

```bash
source scripts/load_env.sh
```

The answer-model CLI accepts a Hugging Face id, a local model path, or one of
the configured aliases: `llama`, `mistral`, `qwen`, `qwen3-14b`. The `qwen`
alias resolves to `Qwen/Qwen2.5-7B-Instruct`; `qwen3-14b` resolves to
`Qwen/Qwen3-14B`.

## 2. Install

```bash
bash scripts/setup_remote.sh
```

The setup script initializes Jasper, installs the benchmark, Jasper, vLLM, and CUDA wheel constraints, downloads LoCoMo to `data/locomo10.json` when needed, and precomputes embeddings into `${BENCHMARK_CACHE_ROOT}/embeddings`.
Pre-embedding is required and setup fails if embedding credentials or network access are missing.

Activate the environment before running benchmark commands:

```bash
source .venv/bin/activate
```

Timed runs read from that cache and fail if an embedding is missing.

## 3. Run Jasper KV And Qdrant Prefix

Run answer generation with judging skipped.
This keeps the GPU focused on the in-process answer backend; judge result files afterward.

The `w5`, `w20`, and `w50` variants condition the retained target-turn KV on the immediately preceding 5, 20, or 50 turns, so they are KV turn-context ablations rather than information-equivalent prefix comparisons.
`--jasper-beam-width` is a minimum search width, and the benchmark automatically uses at least `top_k` candidates.

```bash
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ANSWER_MODEL="${ANSWER_MODEL:-llama}"
KV_WINDOW="${KV_WINDOW:-0}"
KV_RUN_ID="${ANSWER_MODEL}-kv-gpu-jasper10-k50-w${KV_WINDOW}-${RUN_STAMP}"
QDRANT_PREFIX_RUN_ID="${ANSWER_MODEL}-prefix-qdrant10-k50-${RUN_STAMP}"

locomo-jasper-bench \
  --dataset data/locomo10.json \
  --results-dir "${BENCHMARK_RESULTS_ROOT}" \
  --answer-model "${ANSWER_MODEL}" \
  --answer-backend vllm-kv \
  --vector-backend jasper \
  --top-k 50 \
  --context-window "${KV_WINDOW}" \
  --kv-gpu-memory-utilization 0.38 \
  --max-samples 10 \
  --log-every 1 \
  --skip-judge \
  --run-id "${KV_RUN_ID}"
```

```bash
locomo-jasper-bench \
  --dataset data/locomo10.json \
  --results-dir "${BENCHMARK_RESULTS_ROOT}" \
  --answer-model "${ANSWER_MODEL}" \
  --answer-backend vllm-prefix \
  --vector-backend qdrant \
  --top-k 50 \
  --context-window 0 \
  --kv-gpu-memory-utilization 0.38 \
  --max-samples 10 \
  --log-every 1 \
  --skip-judge \
  --run-id "${QDRANT_PREFIX_RUN_ID}"
```

Both runs write query-start-to-answer-complete latency as `metrics.query_to_answer_ms`.
This is a single stopwatch around query embedding, vector retrieval, prompt/KV composition, and answer generation.
It does not include memory storage/index construction, KV precompute, vLLM startup, or judging.

TTFT metrics come from vLLM request timing on the real answer `generate()` call.
They are populated only when vLLM returns usable per-request metrics on `RequestOutput.metrics`; no one-token probe or synthetic fallback is used.

The standard sweep runs four Jasper-KV windows and one Qdrant-prefix baseline for every model and top-k pair, for `4 models x 5 top-k values x 5 variants = 100 runs`.

```bash
DRY_RUN=1 BENCHMARK_RESULTS_ROOT="${BENCHMARK_RESULTS_ROOT}" bash scripts/full_run.sh
BENCHMARK_RESULTS_ROOT="${BENCHMARK_RESULTS_ROOT}" bash scripts/full_run.sh
```


## 4. Judge Accuracy

The CLI selects a judge with `--judge vllm|openai|none`.
`--skip-judge` is still accepted as an alias for `--judge none`.

For local Gemma/vLLM judging on the same GPU, start the judge after answer runs finish:

```bash
source .venv/bin/activate
bash scripts/serve_vllm.sh
```

Then judge each run from another shell:

```bash
locomo-jasper-bench \
  --results-dir "${BENCHMARK_RESULTS_ROOT}" \
  --run-id "${RUN_ID}" \
  --judge-only \
  --judge vllm \
  --judge-base-url "${JUDGE_BASE_URL}" \
  --judge-api-key "${JUDGE_API_KEY}" \
  --judge-model "${JUDGE_MODEL}"
```

OpenAI judging uses `OPENAI_API_KEY` and `OPENAI_JUDGE_MODEL`, runs through the OpenAI Batch API, and does not require `scripts/serve_vllm.sh`.

```bash
locomo-jasper-bench \
  --results-dir "${BENCHMARK_RESULTS_ROOT}" \
  --run-id "${RUN_ID}" \
  --judge-only \
  --judge openai \
  --judge-model "${OPENAI_JUDGE_MODEL:-gpt-5.4}"
```

`--judge-only` fills only rows that are still unjudged, preserves already judged rows, and regenerates `summary.json`.
Add `--rejudge` with `--judge-only` to replace existing judge results for every row in the run.
OpenAI batch judging persists `openai_judge_batch_input.jsonl`, `openai_judge_batch_output.jsonl`, `openai_judge_batch_errors.jsonl`, and `openai_judge_batch.json` in the run directory.

## 5. Compare Results

Each run writes to `${BENCHMARK_RESULTS_ROOT}/<run-id>/`:

- `config.json`
- `system.json`
- `predictions.jsonl`
- `summary.json`
- `query_metrics.csv`
- `sample_setup_metrics.csv`
- `plots/tokens_vs_ttft.png`
- `plots/tokens_vs_query_to_first_token.png`
- `plots/tokens_vs_query_to_answer.png`
- `plots/tokens_vs_accuracy_binned.png`
- `plots/tokens_vs_accuracy_binned.csv`

`query_metrics.csv` is derived from `predictions.jsonl` after normal generation and after `--judge-only`.
It uses `metrics.kv_memory_tokens` as the retrieved memory token count and computes total input prompt tokens as `memory + query tokens`.

Read the primary metrics:

```bash
cat "${BENCHMARK_RESULTS_ROOT}/${KV_RUN_ID}/summary.json"
cat "${BENCHMARK_RESULTS_ROOT}/${QDRANT_PREFIX_RUN_ID}/summary.json"
```

Primary summary metrics:

- `metrics.accuracy`: judged answer quality.
- `metrics.time_to_first_token_ms`: in-process vLLM time to first token from the real answer generation, when vLLM exposes request timing metrics.
- `metrics.query_to_first_token_ms`: query-start-to-generate-start wall time plus vLLM time to first token, when vLLM exposes request timing metrics.
- `metrics.query_to_answer_ms`: query embedding, retrieval, prompt/KV composition, and full answer generation measured with one stopwatch.
- `metrics.sample_setup_time_ms`: per-sample setup before the first query, including memory/index construction, KV precompute when applicable, and sample activation.
- `metrics.vector_db_query_time_ms`: raw backend vector search latency.
- Jasper graph and embedding metrics in `summary.json` and `sample_setup_metrics.csv`: `jasper_graph_gpu_mb` is the packed serialized graph size matching Jasper `total_file_size`, while `jasper_embedding_matrix_gpu_logical_mb` and `jasper_embedding_matrix_cpu_mb` describe embedding matrix storage.
- Llama KV chunk metrics in `summary.json` and `sample_setup_metrics.csv`: `llama_kv_total_tensor_gpu_mb`, `llama_kv_chunk_tensor_gpu_mb`, `llama_kv_prefix_tensor_gpu_mb`, and `llama_kv_chunk_map_cpu_mb`.
