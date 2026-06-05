# LoCoMo vLLM + Jasper Runbook

This runbook shows how to run LoCoMo against a local vLLM server with Mem0 retrieval backed by Jasper or Qdrant. It keeps caches, temp files, and outputs under `/scratch/$USER/benchmark-jasper` so the repo/project directory stays small.

Jasper is the default backend. Its default graph alpha is `1.0`.

## 1. Setup

Run from the repo root on the GPU machine:

```bash
cd /projects/SaltSystemsLab/peter/benchmark-jasper

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

Verify the install:

```bash
python - <<'PY'
import torch, vllm, transformers
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("vllm:", vllm.__version__)
print("transformers:", transformers.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY

python scripts/smoke_jasper.py
```

## 2. Start vLLM

Use tmux so the server stays running while the benchmark runs in another window:

```bash
tmux new -s locomo
```

In tmux window 1:

```bash
cd /projects/SaltSystemsLab/peter/benchmark-jasper
source .venv/bin/activate
source scripts/scratch_env.sh

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

Smoke check from window 2:

```bash
curl --noproxy '*' \
  -H "Authorization: Bearer ${VLLM_API_KEY:-token-abc123}" \
  http://127.0.0.1:8000/v1/models
```

If this returns a proxy error, set `NO_PROXY`/`no_proxy` as shown below and use `127.0.0.1`. If it says connection refused, confirm both tmux windows are on the same compute node with `hostname`.

## 3. Data

Place LoCoMo at `data/locomo10.json`:

```bash
cd /projects/SaltSystemsLab/peter/benchmark-jasper
mkdir -p data
curl -L \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json \
  -o data/locomo10.json
```

## 4. Shared Benchmark Environment

Run this in the benchmark tmux window before any benchmark command:

```bash
cd /projects/SaltSystemsLab/peter/benchmark-jasper
source .venv/bin/activate
source scripts/scratch_env.sh

export VLLM_BASE_URL=http://127.0.0.1:8000/v1
export JUDGE_BASE_URL=http://127.0.0.1:8000/v1
export VLLM_API_KEY=token-abc123
export JUDGE_API_KEY=token-abc123
export OPENAI_API_KEY=<your-openai-key>
export NO_PROXY=localhost,127.0.0.1,::1
export no_proxy="${NO_PROXY}"
```

Mem0 uses OpenAI embeddings by default. The benchmark caches embeddings under `${BENCHMARK_CACHE_ROOT}/embeddings`, keyed by model, purpose, and exact text. Use `--no-embedding-cache` only when you intentionally want to re-embed everything.

Embeddings are passed to Jasper and Qdrant as raw vectors.

## 5. Small Jasper Check

Use this for a quick end-to-end run:

```bash
locomo-jasper-bench \
  --dataset data/locomo10.json \
  --results-dir "${SCRATCH_ROOT}/results" \
  --vector-backend jasper \
  --jasper-alpha 1.0 \
  --max-samples 1 \
  --max-questions 3 \
  --stream \
  --log-every 1 \
  --run-id jasper-small-$(date -u +%Y%m%dT%H%M%SZ)
```

Use `--stream` for TTFT. Without it, `metrics.time_to_first_token_ms` is `null`.

## 6. Full Jasper Baseline

```bash
locomo-jasper-bench \
  --dataset data/locomo10.json \
  --results-dir "${SCRATCH_ROOT}/results" \
  --vector-backend jasper \
  --jasper-alpha 1.0 \
  --stream \
  --log-every 10 \
  --run-id jasper-full-$(date -u +%Y%m%dT%H%M%SZ)
```

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
- `metrics.time_to_first_token_ms`: time to first answer token, only populated with `--stream`.
- `metrics.vector_db_query_time_ms`: raw backend vector DB query time; payload lookup and result formatting are excluded.
- `metrics.throughput_tokens_per_sec`: answer generation throughput.

Jasper and Qdrant both return backend-ordered results over raw embeddings.

## 9. Scratch Checks

Run after setup or a benchmark:

```bash
ls -ld .cache
df -h /projects/SaltSystemsLab/peter/benchmark-jasper
du -sh .cache/
du -sh "${SCRATCH_ROOT}/results"
```

`.cache` should point to `/scratch/$USER/benchmark-jasper/cache`, and large model/cache/result files should grow under scratch.
