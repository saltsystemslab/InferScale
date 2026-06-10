# LoCoMo vLLM Jasper Benchmark

This repository contains code to run LoCoMo against a local vLLM server with Mem0 retrieval backed by a vector database such as Jasper or Qdrant. Runtime files are kept under `/scratch/$USER/benchmark-jasper` so the repo does not run out of space.

## 1. Configure

From the repo root on the remote machine:

```bash
cd /projects/SaltSystemsLab/<PATH_TO_REPO>/benchmark-jasper
cp .env.example .env
```

Edit `.env` for your session. The common values are:

- `SCRATCH_ROOT=/scratch/$USER/benchmark-jasper`
- `CUDA_MODULE=cuda/12.8`
- `VLLM_MODEL=google/gemma-3-12b-it`
- `VLLM_API_KEY=token-abc123`
- `OPENAI_API_KEY=...`
- `HF_TOKEN=...` (optional)

Load the environment in any shell that will run project commands:

```bash
source scripts/load_env.sh
```

`scripts/load_env.sh` reads `.env`, prepares scratch-backed cache/result/temp directories, and refreshes `.cache` as a symlink to `${SCRATCH_ROOT}/cache`. Set `BENCHMARK_USE_SCRATCH=0` in `.env` only if you intentionally want local repo cache/result directories.

## 2. Install

```bash
bash scripts/setup_remote.sh
```

## 3. Start vLLM

Use tmux so the server keeps running while the benchmark runs in another window:

```bash
tmux new -s locomo
```

In window 1, start the Gemma server:

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

The server defaults to `google/gemma-3-12b-it` on port `8000`. If Hugging Face gates the model, make sure `HF_TOKEN` is set before starting vLLM.

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

For full-dataset ablations, precompute without `--max-samples`:

```bash
locomo-jasper-bench \
  --dataset data/locomo10.json \
  --results-dir "${BENCHMARK_RESULTS_ROOT}" \
  --preembed-only \
  --run-id preembed-full-${RUN_STAMP}
```

## 6. Compare Qdrant And Jasper

Run both back to back with the same sample count, model, `top_k`, and streaming setting:

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

## 7. Full-Dataset Jasper Ablations

To test whether beam width `128` or L2 vector normalization helps, run the full dataset with the Gemma model:

```bash
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)

COMMON_ARGS=(
  --dataset data/locomo10.json
  --results-dir "${BENCHMARK_RESULTS_ROOT}"
  --vector-backend jasper
  --top-k 20
  --retrieval-diagnostic-k 128
  --stream
)

locomo-jasper-bench "${COMMON_ARGS[@]}" \
  --jasper-beam-width 64 \
  --run-id jasper-bw64-raw-${RUN_STAMP}

locomo-jasper-bench "${COMMON_ARGS[@]}" \
  --jasper-beam-width 128 \
  --run-id jasper-bw128-raw-${RUN_STAMP}

locomo-jasper-bench "${COMMON_ARGS[@]}" \
  --jasper-beam-width 64 \
  --vector-normalize \
  --run-id jasper-bw64-norm-${RUN_STAMP}

locomo-jasper-bench "${COMMON_ARGS[@]}" \
  --jasper-beam-width 128 \
  --vector-normalize \
  --run-id jasper-bw128-norm-${RUN_STAMP}
```

Compare the run summaries:

```bash
locomo-compare-runs \
  "${BENCHMARK_RESULTS_ROOT}/jasper-bw64-raw-${RUN_STAMP}" \
  "${BENCHMARK_RESULTS_ROOT}/jasper-bw128-raw-${RUN_STAMP}" \
  "${BENCHMARK_RESULTS_ROOT}/jasper-bw64-norm-${RUN_STAMP}" \
  "${BENCHMARK_RESULTS_ROOT}/jasper-bw128-norm-${RUN_STAMP}" \
  --json-output "${BENCHMARK_RESULTS_ROOT}/jasper-ablation-${RUN_STAMP}.json"
```

Treat a setting as helpful only if it improves `metrics.accuracy` under the same Gemma setup; use vector query time and exact recall as secondary tradeoff metrics.

## 8. Results

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
- `metrics.exact_top_k_answer_accuracy`: judged answer quality when Jasper exact top-k answer diagnostics are enabled.

## 9. Diagnose Jasper Exact Top-K Answer Quality

To compare Jasper's approximate-retrieval answer accuracy against exact top-k retrieval over the same in-memory vectors, run Jasper with:

```bash
locomo-jasper-bench \
  --dataset data/locomo10.json \
  --results-dir "${BENCHMARK_RESULTS_ROOT}" \
  --vector-backend jasper \
  --exact-answer-baseline \
  --top-k 20 \
  --max-samples 20 \
  --stream \
  --run-id jasper-exact-answer-20samples-${RUN_STAMP}
```

The normal Jasper answer remains in each `predictions.jsonl` row. The exact top-k answer is added under `exact_top_k_answer`, including its retrieved memories, predicted answer, judge result, and diagnostic metrics. The summary includes `metrics.exact_top_k_answer_accuracy`, `metrics.answer_accuracy_delta_exact_minus_jasper`, and paired disagreement counts.

This mode runs an extra exact search, answer call, and judge call per question, so use it for answer-quality diagnosis rather than latency benchmarking.

## 10. Diagnose Retrieval Quality

To compare completed Jasper and Qdrant runs against LoCoMo evidence turns, run:

```bash
locomo-retrieval-diagnostics \
  --dataset data/locomo10.json \
  --jasper-run "${BENCHMARK_RESULTS_ROOT}/jasper-20samples-${RUN_STAMP}" \
  --qdrant-run "${BENCHMARK_RESULTS_ROOT}/qdrant-20samples-${RUN_STAMP}" \
  --top-k 20 \
  --output-dir "${BENCHMARK_RESULTS_ROOT}/retrieval-diagnostics-${RUN_STAMP}"
```

This writes `summary.json`, `jasper_retrieval.jsonl`, and, when provided, `qdrant_retrieval.jsonl`. The summary reports evidence hit rate, evidence item recall, MRR, answer accuracy, and pairwise examples where Qdrant found evidence but Jasper missed it.

For a future Jasper run, enable exact-vector diagnostics to compare Jasper's approximate results against exact nearest neighbors over the same in-memory vectors:

```bash
locomo-jasper-bench \
  --dataset data/locomo10.json \
  --results-dir "${BENCHMARK_RESULTS_ROOT}" \
  --vector-backend jasper \
  --retrieval-diagnostic-k 64 \
  --max-samples 20 \
  --stream \
  --run-id jasper-diagnostic-20samples-${RUN_STAMP}
```

Each prediction record will include `retrieval_diagnostics`, including exact recall at the requested `top_k`, exact top-k items missing from Jasper's top-k, and exact top-k items found lower in the larger Jasper candidate list.

Because this mode runs extra candidate search and exact CPU scoring, use it for retrieval diagnosis rather than latency comparisons.
