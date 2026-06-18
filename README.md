# LoCoMo vLLM Jasper Benchmark

This repository contains code to run LoCoMo against a local vLLM server with Mem0 retrieval backed by a vector database such as Jasper or Qdrant. On Runpod, keep the repo and runtime files under `/workspace` so caches, model downloads, temp files, and results persist on workspace storage.

## 1. Configure

From the repo root on the remote machine:

```bash
cd /workspace
cp .env.example .env
```

Edit `.env` for your session. The common values are:

- `BENCHMARK_RUNTIME_ROOT=/workspace`
- `CUDA_MODULE=` on Runpod, or `CUDA_MODULE=cuda/12.8` on module-based clusters
- `VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct`
- `VLLM_API_KEY=token-abc123`
- `OPENAI_API_KEY=...`
- `HF_TOKEN=...` (optional)

Load the environment in any shell that will run project commands:

```bash
source scripts/load_env.sh
```

`scripts/load_env.sh` reads `.env` and prepares cache/result/temp directories under `${BENCHMARK_RUNTIME_ROOT}`. With `BENCHMARK_RUNTIME_ROOT=/workspace`, caches live under `/workspace/.cache`, results under `/workspace/results`, and temp files under `/workspace/tmp`. Legacy scratch clusters can set `SCRATCH_ROOT=/scratch/$USER/benchmark-jasper`; set `BENCHMARK_USE_SCRATCH=0` only if you intentionally want project-local cache/result directories.

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

If `vllm serve` returns `500 Internal Server Error` and the server log contains `AttributeError: '_IncludedRouter' object has no attribute 'path'`, reinstall the pinned FastAPI version in the venv and restart vLLM:

```bash
source .venv/bin/activate
python -m pip install -c constraints-cu128.txt "fastapi[standard]==0.115.14"
```

That error comes from vLLM 0.10.2's Prometheus middleware with newer FastAPI route objects, before the request reaches model generation.

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

Run both back to back with the same sample count, model, `top_k`, and streaming setting:

If the answer model and judge model cannot run at the same time, add `--skip-judge` to each command below and use [Deferred Judging](#8-deferred-judging) after inference finishes.

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

For a quick smoke comparison, add this to each command:

```bash
--max-samples 1 --max-questions 3 --log-every 1
```

## 7. Strict GPU KV Injection

The `ai-memory-code` submodule can be used through an opt-in in-process vLLM backend. This mode keeps the current Mem0/Jasper top-k retrieval step, then composes the retrieved turns as chunked-RoPE KV tensors on GPU and injects them through a strict GPU connector. It does not use `memory_path`, safetensors loading, `CPUMemoryStore`, vLLM CPU swap, or vLLM CPU offload.

This path runs vLLM inside the benchmark process, so do not start `scripts/serve_vllm.sh` for the answer model. If the inference model and judge model cannot fit at the same time, run this step with `--skip-judge`, then run deferred judging after the inference process exits.

```bash
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_ID=kv-strict-smoke-${RUN_STAMP}

locomo-jasper-bench \
  --dataset data/locomo10.json \
  --results-dir "${BENCHMARK_RESULTS_ROOT}" \
  --answer-backend vllm-kv \
  --vector-backend jasper \
  --top-k 20 \
  --kv-gpu-memory-utilization 0.55 \
  --kv-max-model-len 32768 \
  --kv-max-position 32768 \
  --max-samples 1 \
  --max-questions 3 \
  --log-every 1 \
  --skip-judge \
  --run-id "${RUN_ID}"
```

Use `--kv-gpu-memory-utilization` conservatively because the retrieved chunk KV tensors remain GPU-resident while vLLM is loaded.

## 8. Deferred Judging

Use deferred judging when the answer model and judge model cannot run at the same time. This works for both normal OpenAI-compatible answer runs and strict GPU KV runs.

First, run inference with a stable run id and `--skip-judge`:

```bash
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_ID=jasper-20samples-${RUN_STAMP}

locomo-jasper-bench \
  --dataset data/locomo10.json \
  --results-dir "${BENCHMARK_RESULTS_ROOT}" \
  --vector-backend jasper \
  --max-samples 20 \
  --stream \
  --skip-judge \
  --run-id "${RUN_ID}"
```

That writes predictions immediately. Until judging runs, `summary.json` has `judged_count` set to `0` and `metrics.accuracy` set to `null`.

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

To inspect what is in `predictions.jsonl` before or after judging:

```bash
locomo-jasper-bench \
  --results-dir "${BENCHMARK_RESULTS_ROOT}" \
  --run-id "${RUN_ID}" \
  --inspect-run \
  --inspect-limit 5
```

This prints judge status counts plus a few question, reference answer, predicted answer, judge reason, and top retrieved memory snippets. Use it to distinguish bad predictions from bad judge parsing.

If a judge run produced bad labels, for example `accuracy=0.0000` because the judge model returned malformed or overly strict judgments, rejudge the existing predictions without rerunning inference:

```bash
locomo-jasper-bench \
  --results-dir "${BENCHMARK_RESULTS_ROOT}" \
  --run-id "${RUN_ID}" \
  --judge-only \
  --judge-rejudge \
  --judge-base-url "${JUDGE_BASE_URL:-${VLLM_BASE_URL}}" \
  --judge-api-key "${JUDGE_API_KEY:-${VLLM_API_KEY}}" \
  --judge-model "${JUDGE_MODEL:-${VLLM_MODEL}}"
```

If the judge server returns an error such as `openai.InternalServerError`, check the judge server logs first. The benchmark saves completed judgments back to `predictions.jsonl`, marks the failed row with `judge.status = "error"` while leaving it unjudged, and can be resumed by rerunning the same `--judge-only` command.

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
- `metrics.time_to_first_token_ms`: time to first answer token; populated when using `--stream`.
- `metrics.vector_db_query_time_ms`: raw backend vector query time.
- `metrics.vector_db_queries_per_sec`: vector DB query throughput.
