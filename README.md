# LoCoMo KV Cache Benchmarks

This repository runs a LoCoMo benchmark comparison between in-process vLLM answer backends:

- `vllm-kv`: retrieved memories are encoded with the package's chunked-RoPE helpers, then injected through the top-level GPU KV connector.
- `vllm-prefix`: the same retrieved memory tokens are included as a normal vLLM prompt prefix.

## 1. Configure

```bash
cp .env.example .env
```

Edit `.env` for your session. The common values are:

- `LOCOMO_VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct`
- `JUDGE_MODEL=Gemma-2-9B-Instruct`
- `CUDA_MODULE=` for Runpod containers without environment modules
- `OPENAI_API_KEY=...`
- `HF_TOKEN=...` if the model is gated

By default, runtime storage is scratch-backed under `${SCRATCH_ROOT:-/scratch/$USER/benchmark-jasper}`:

- `${SCRATCH_ROOT}/cache` for embeddings, Mem0/Jasper files, model downloads, and build caches.
- `${SCRATCH_ROOT}/results` for benchmark outputs.
- `${SCRATCH_ROOT}/tmp` for temporary files.

`source scripts/load_env.sh` prepares those directories and points the repo `.cache` entry at scratch. Set `BENCHMARK_USE_SCRATCH=0` only when you intentionally want project-local storage. Override individual paths directly if needed:

```bash
export BENCHMARK_CACHE_ROOT=/path/to/cache
export BENCHMARK_RESULTS_ROOT=/path/to/results
```

Load the environment in each shell that will run project commands:

```bash
source scripts/load_env.sh
```

## 2. Install

```bash
bash scripts/setup_remote.sh
```

Activate the environment before running benchmark commands:

```bash
source .venv/bin/activate
```

## 3. Data

Place LoCoMo at `data/locomo10.json`:

```bash
mkdir -p data
curl -L \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json \
  -o data/locomo10.json
```

## 4. Precompute Embeddings

Embeddings are cached under `${BENCHMARK_CACHE_ROOT}/embeddings`, keyed by model, purpose, and exact text. Precompute before timed runs so embedding calls are outside the benchmark:

```bash
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)

locomo-jasper-bench \
  --dataset data/locomo10.json \
  --results-dir "${BENCHMARK_RESULTS_ROOT}" \
  --max-samples 10 \
  --preembed-only \
  --run-id "preembed-10samples-${RUN_STAMP}"
```

Timed runs read from that cache and fail if an embedding is missing.

## 5. Run KV And Prefix

Run answer generation with judging skipped. This keeps the GPU focused on the in-process answer backend; judge result files afterward.

```bash
KV_RUN_ID="kv-gpu-jasper10-${RUN_STAMP}"
PREFIX_RUN_ID="prefix-gpu-jasper10-${RUN_STAMP}"
QDRANT_PREFIX_RUN_ID="prefix-qdrant10-${RUN_STAMP}"

locomo-jasper-bench \
  --dataset data/locomo10.json \
  --results-dir "${BENCHMARK_RESULTS_ROOT}" \
  --answer-backend vllm-kv \
  --vector-backend jasper \
  --top-k 50 \
  --context-window 3 \
  --kv-gpu-memory-utilization 0.52 \
  --kv-max-model-len 32768 \
  --kv-max-position 32768 \
  --max-samples 10 \
  --log-every 1 \
  --skip-judge \
  --run-id "${KV_RUN_ID}"
```

```bash
locomo-jasper-bench \
  --dataset data/locomo10.json \
  --results-dir "${BENCHMARK_RESULTS_ROOT}" \
  --answer-backend vllm-prefix \
  --vector-backend jasper \
  --top-k 50 \
  --kv-gpu-memory-utilization 0.52 \
  --kv-max-model-len 32768 \
  --kv-max-position 32768 \
  --max-samples 10 \
  --log-every 1 \
  --skip-judge \
  --run-id "${PREFIX_RUN_ID}"
```

```bash
locomo-jasper-bench \
  --dataset data/locomo10.json \
  --results-dir "${BENCHMARK_RESULTS_ROOT}" \
  --answer-backend vllm-prefix \
  --vector-backend qdrant \
  --top-k 50 \
  --kv-gpu-memory-utilization 0.52 \
  --kv-max-model-len 32768 \
  --kv-max-position 32768 \
  --max-samples 10 \
  --log-every 1 \
  --skip-judge \
  --run-id "${QDRANT_PREFIX_RUN_ID}"
```

All three runs write query-start-to-answer-complete latency as `metrics.query_to_answer_ms`. This is a single stopwatch around query embedding, vector retrieval, prompt/KV composition, and answer generation. It does not include memory storage/index construction, KV precompute, vLLM startup, or judging.

TTFT metrics come from vLLM request timing on the real answer `generate()` call. They are populated only when vLLM returns usable per-request metrics on `RequestOutput.metrics`; no one-token probe or synthetic fallback is used.

## 6. Judge Accuracy

If the judge will run on the same GPU, start it after both answer runs finish:

```bash
source .venv/bin/activate
bash scripts/serve_vllm.sh
```

Then judge each run from another shell (or use tmux: `tmux new -s locomo` and create a new window with `Ctrl-b c`):

```bash
locomo-jasper-bench \
  --results-dir "${BENCHMARK_RESULTS_ROOT}" \
  --run-id "${RUN_ID}" \
  --judge-only \
  --judge-base-url "${JUDGE_BASE_URL}" \
  --judge-api-key "${JUDGE_API_KEY}" \
  --judge-model "${JUDGE_MODEL}"
```

`--judge-only` fills only rows that are still unjudged, preserves already judged rows, and regenerates `summary.json`.

## 7. Compare Results

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

`query_metrics.csv` is derived from `predictions.jsonl` after normal generation and after `--judge-only`. It uses `metrics.kv_memory_tokens` as the retrieved memory token count and computes total input prompt tokens as `memory + query tokens`.

Read the primary metrics:

```bash
cat "${BENCHMARK_RESULTS_ROOT}/${KV_RUN_ID}/summary.json"
cat "${BENCHMARK_RESULTS_ROOT}/${PREFIX_RUN_ID}/summary.json"
cat "${BENCHMARK_RESULTS_ROOT}/${QDRANT_PREFIX_RUN_ID}/summary.json"
```

Primary summary metrics:

- `metrics.accuracy`: judged answer quality.
- `metrics.time_to_first_token_ms`: in-process vLLM time to first token from the real answer generation, when vLLM exposes request timing metrics.
- `metrics.query_to_first_token_ms`: query-start-to-generate-start wall time plus vLLM time to first token, when vLLM exposes request timing metrics.
- `metrics.query_to_answer_ms`: query embedding, retrieval, prompt/KV composition, and full answer generation measured with one stopwatch.
- `metrics.sample_setup_time_ms`: per-sample setup before the first query, including memory/index construction, KV precompute when applicable, and sample activation.
- `metrics.vector_db_query_time_ms`: raw backend vector search latency.
