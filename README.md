# LoCoMo vLLM Jasper Benchmark

This repository contains code to run LoCoMo against a local vLLM server with Mem) retrieval backed by a vector database (e.g. Jasper or Qdrant). All files are kept  under `/scratch/$USER/benchmark-jasper` so the repo/project doesn't run out of memory.

## 1. Setup

Run from the repo root after allocating GPU, you may need to load `cmake/4.2.3`, `python/3.13.5`, and `cuda/12.8.0`.

```bash
cd /projects/SaltSystemsLab/<PATH_TO_REPO>/benchmark-jasper

export SCRATCH_ROOT=/scratch/$USER/benchmark-jasper
FRESH_REMOTE_BUILD=1 bash scripts/setup_remote.sh

source .venv/bin/activate
source scripts/scratch_env.sh
```

`FRESH_REMOTE_BUILD=1` removes the old `.venv`, `.cache`, `tmp`, `jasperpy/build`, and installed Jasper `.so` files before reinstalling. Omit it to reuse the existing environment.

`scripts/scratch_env.sh` creates scratch-backed directories and exports:

- `BENCHMARK_CACHE_ROOT=${SCRATCH_ROOT}/cache`
- `BENCHMARK_RESULTS_ROOT=${SCRATCH_ROOT}/results`
- `MEM0_DIR=${BENCHMARK_CACHE_ROOT}/mem0`
- `TMPDIR=${SCRATCH_ROOT}/tmp`
- `PIP_CACHE_DIR=${SCRATCH_ROOT}/pip`
- `XDG_CACHE_HOME=${BENCHMARK_CACHE_ROOT}/xdg`

It also refreshes `.cache` as a symlink to `${SCRATCH_ROOT}/cache`.

## 2. Start vLLM

NOTE: Official Meta Llama models may require Hugging Face access approval and login or `HF_TOKEN`.

Use tmux so that the server stays running while the benchmark runs in another window:

```bash
tmux new -s locomo
```

In tmux window 1:

```bash
cd /projects/SaltSystemsLab/<PATH_TO_REPO>/benchmark-jasper

export CUDA_MODULE=cuda/12.8
export VLLM_TP=1
export VLLM_GPU_MEMORY_UTILIZATION=0.80
export VLLM_MAX_MODEL_LEN=32768
export VLLM_DTYPE=auto
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_API_KEY=token-abc123
export VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct

bash scripts/serve_vllm.sh
```

Create a second tmux window for benchmark commands:

```bash
Ctrl-b c
```

Double check the vllm server is running correctly from window 2:

```bash
curl --noproxy '*' \
  -H "Authorization: Bearer ${VLLM_API_KEY:-token-abc123}" \
  http://127.0.0.1:8000/v1/models
```

## 3. Data

Place LoCoMo at `data/locomo10.json`:

```bash
cd /projects/SaltSystemsLab/<PATH_TO_REPO>/benchmark-jasper
mkdir -p data
curl -L \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json \
  -o data/locomo10.json
```

## 4. Shared Benchmark Environment

Run this in the benchmark tmux window before any benchmark command:

```bash

export VLLM_BASE_URL=http://127.0.0.1:8000/v1
export JUDGE_BASE_URL=http://127.0.0.1:8000/v1
export VLLM_API_KEY=token-abc123
export JUDGE_API_KEY=token-abc123
export OPENAI_API_KEY=<your-openai-key>
export NO_PROXY=localhost,127.0.0.1,::1
export no_proxy="${NO_PROXY}"
```

`OPENAI_API_KEY` is required for Mem0 embeddings. The benchmark caches embeddings under `${BENCHMARK_CACHE_ROOT}/embeddings`, keyed by model, purpose, and exact text.

Embeddings are passed to Jasper and Qdrant as raw vectors.

Before timed benchmark runs, precompute the LoCoMo turn and question embeddings:

```bash
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)

locomo-jasper-bench \
  --dataset data/locomo10.json \
  --results-dir "${SCRATCH_ROOT}/results" \
  --max-samples 20 \
  --preembed-only \
  --run-id preembed-20samples-${RUN_STAMP}
```

Timed benchmark runs read embeddings from the cache and fail if an embedding is missing.

## 7. Jasper vs Qdrant

Run back to back with the same dataset, model, `top_k`, sample count, and streaming setting.

For 20 samples:

```bash
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)

locomo-jasper-bench \
  --dataset data/locomo10.json \
  --results-dir "${SCRATCH_ROOT}/results" \
  --vector-backend qdrant \
  --max-samples 20 \
  --stream \
  --log-every 10 \
  --run-id qdrant-20samples-${RUN_STAMP}

locomo-jasper-bench \
  --dataset data/locomo10.json \
  --results-dir "${SCRATCH_ROOT}/results" \
  --vector-backend jasper \
  --jasper-alpha 1.0 \
  --max-samples 20 \
  --stream \
  --log-every 10 \
  --run-id jasper-20samples-${RUN_STAMP}
```

For a small comparison, replace `--max-samples 20` with:

```bash
--max-samples 1 --max-questions 3 --log-every 1
```

## 8. Read Results

Each run writes:

- `config.json`
- `system.json`
- `predictions.jsonl`
- `summary.json`

Open summaries:

```bash
cat "${SCRATCH_ROOT}/results/qdrant-20samples-${RUN_STAMP}/summary.json"
cat "${SCRATCH_ROOT}/results/jasper-20samples-${RUN_STAMP}/summary.json"
```

Important fields:

- `metrics.accuracy`: judged answer quality.
- `metrics.time_to_first_token_ms`: time from initial question handling through first answer token; only populated with `--stream`.
- `metrics.vector_db_query_time_ms`: raw backend vector DB query time; payload lookup and result formatting are excluded.
- `metrics.vector_db_query_count`: number of measured vector DB queries.
- `metrics.vector_db_query_time_total_ms`: total measured vector DB query time.
- `metrics.vector_db_queries_per_sec`: vector DB throughput, computed as measured queries divided by total measured query time.
