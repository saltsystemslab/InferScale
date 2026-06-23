# LoCoMo KV Cache Benchmarks

This repository runs a focused LoCoMo comparison between two in-process vLLM answer backends:

- `vllm-kv`: retrieved memories are composed as chunked-RoPE KV tensors and injected through the `ai-memory-code` GPU connector.
- `vllm-prefix`: the same retrieved memory tokens are included as a normal vLLM prompt prefix.

## 1. Configure

```bash
cp .env.example .env
```

Edit `.env` for your session. The common values are:

- `VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct`
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

The setup script initializes the `ai-memory-code` submodule and installs the benchmark, Jasper, vLLM, and CUDA wheel constraints.

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
  --max-samples 20 \
  --preembed-only \
  --run-id "preembed-20samples-${RUN_STAMP}"
```

Timed runs read from that cache and fail if an embedding is missing.

## 5. Run KV And Prefix

Run answer generation with judging skipped. This keeps the GPU focused on the in-process answer backend; judge both result files afterward.

```bash
KV_RUN_ID="kv-gpu-jasper5-${RUN_STAMP}"
PREFIX_RUN_ID="prefix-gpu-jasper5-${RUN_STAMP}"

locomo-jasper-bench \
  --dataset data/locomo10.json \
  --results-dir "${BENCHMARK_RESULTS_ROOT}" \
  --answer-backend vllm-kv \
  --top-k 50 \
  --context-window 3 \
  --kv-gpu-memory-utilization 0.30 \
  --kv-max-model-len 32768 \
  --kv-max-position 32768 \
  --max-samples 5 \
  --log-every 1 \
  --skip-judge \
  --run-id "${KV_RUN_ID}"

locomo-jasper-bench \
  --dataset data/locomo10.json \
  --results-dir "${BENCHMARK_RESULTS_ROOT}" \
  --answer-backend vllm-prefix \
  --top-k 50 \
  --kv-gpu-memory-utilization 0.30 \
  --kv-max-model-len 32768 \
  --kv-max-position 32768 \
  --max-samples 5 \
  --log-every 1 \
  --skip-judge \
  --run-id "${PREFIX_RUN_ID}"
```

Both runs write a one-token vLLM probe as `metrics.time_to_first_token_ms`.

## 6. Judge Accuracy

If the judge will run on the same GPU, start it after both answer runs finish:

```bash
source .venv/bin/activate
bash scripts/serve_vllm.sh
```

Then judge each run from another shell (or use tmux: `tmux new -s locomo` and create a new window with `Ctrl-b c`):

```bash
for RUN_ID in "${KV_RUN_ID}" "${PREFIX_RUN_ID}"; do
  locomo-jasper-bench \
    --results-dir "${BENCHMARK_RESULTS_ROOT}" \
    --run-id "${RUN_ID}" \
    --judge-only \
    --judge-base-url "${JUDGE_BASE_URL}" \
    --judge-api-key "${JUDGE_API_KEY}" \
    --judge-model "${JUDGE_MODEL}"
done
```

`--judge-only` fills only rows that are still unjudged, preserves already judged rows, and regenerates `summary.json`.

## 7. Compare Results

Each run writes to `${BENCHMARK_RESULTS_ROOT}/<run-id>/`:

- `config.json`
- `system.json`
- `predictions.jsonl`
- `summary.json`

Read the primary metrics:

```bash
cat "${BENCHMARK_RESULTS_ROOT}/${KV_RUN_ID}/summary.json"
cat "${BENCHMARK_RESULTS_ROOT}/${PREFIX_RUN_ID}/summary.json"
```

Primary summary metrics:

- `metrics.accuracy`: judged answer quality.
- `metrics.time_to_first_token_ms`: in-process vLLM one-token probe latency for the answer backend.
