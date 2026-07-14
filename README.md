# LoCoMo KV Cache Benchmarks

This repository runs a LoCoMo benchmark comparison between in-process vLLM answer backends.
On Runpod, runtime files are kept under `/workspace` so model downloads, caches, temp files, and results persist on the hosted partition.

- `vllm-kv`: retrieved Mem0 facts are encoded with the package's chunked-RoPE helpers, then injected through the top-level GPU KV connector.
- `vllm-prefix`: the same type of retrieved Mem0 facts are included as a normal vLLM prompt prefix.

## 1. Configure

```bash
cp .env.example .env
```

Edit `.env` for your session.
The common values are:

- `MODEL_LLAMA=meta-llama/Llama-3.1-8B-Instruct`
- `MODEL_MISTRAL=mistralai/Mistral-7B-Instruct-v0.3`
- `MODEL_QWEN=Qwen/Qwen2.5-7B-Instruct`
- `MODEL_QWEN3_14B=Qwen/Qwen3-14B`
- `LOCOMO_VLLM_MODEL=llama`
- `BENCHMARK_RUNTIME_ROOT=/workspace`
- `JUDGE_PROVIDER=vllm`
- `JUDGE_MODEL=google/gemma-2-9b-it`
- `MEM0_LLM_BASE_URL=http://localhost:8000/v1`
- `LOCOMO_KV_CONTEXT_WINDOW=0`
- `CUDA_MODULE=` for Runpod containers without environment modules
- `OPENAI_API_KEY=...` for embeddings and Mem0 inference
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

## 2. Install

```bash
bash scripts/setup_remote.sh
```

Activate the environment before running benchmark commands:

```bash
source .venv/bin/activate
```

## Extract the fact catalogs

Timed runs require the immutable fact catalogs and embedding-cache entries produced by `--preembed-only`.
Extraction is not part of `scripts/setup_remote.sh`; run it as a separate step after setup.

With three pods sharing the results network volume, extract the three primary models in parallel, one model per pod:

```bash
# Pod 1
bash scripts/individual/extract_llama.sh

# Pod 2
bash scripts/individual/extract_qwen.sh

# Pod 3
bash scripts/individual/extract_mistral.sh
```

Each script serves its own model on the pod's GPU and writes only that model's catalogs, so the runs never conflict on the shared volume.
The three runs share the embedding cache, which is safe: identical conversations produce identical cache entries regardless of which run writes them first.
The `qwen3-14b` catalogs are not covered by these scripts; extract them afterward on whichever pod frees up first with `EXTRACTION_MODELS="qwen3-14b" bash scripts/extract_facts.sh`.
To extract every model serially on one pod instead, run `bash scripts/extract_facts.sh` with no arguments.

`scripts/extract_facts.sh` runs a fixed bounded extraction protocol for each answer model.
The extraction vLLM server uses a 16,384-token context and each request allows at most 4,096 output tokens, 20 facts, and 600 characters per fact.
The benchmark replaces Mem0's unbounded JSON-object request with a strict JSON schema, validates every cached and fresh response, and aborts instead of caching malformed JSON or silently dropping facts.
The extraction protocol is part of inference-cache and fact-catalog identity, so artifacts from older extraction limits are ignored automatically and must be regenerated.
The embedding cache remains compatible and reusable.

Mem0 stores the facts produced by inference as searchable memory records.
Accordingly, `--top-k` counts inferred records rather than raw conversation turns.
Both `vllm-prefix` and `vllm-kv` retrieve and answer from these inferred fact texts.
Each inferred record retains source metadata that the KV path uses to identify its LoCoMo session context.

## 3. Run Mem0 Fact Benchmarks

Run answer generation with judging skipped.
This keeps the GPU focused on the in-process answer backend; judge result files afterward.

`--jasper-beam-width` is a minimum search width, and the benchmark automatically uses at least `top_k` candidates.

```bash
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ANSWER_MODEL="${ANSWER_MODEL:-llama}"
CONTEXT_WINDOW="${CONTEXT_WINDOW:-0}"
KV_RUN_ID="${ANSWER_MODEL}-kv-mem0-jasper10-k50-s${CONTEXT_WINDOW}-${RUN_STAMP}"
PREFIX_RUN_ID="${ANSWER_MODEL}-prefix-mem0-qdrant10-k50-s0-${RUN_STAMP}"

locomo-jasper-bench \
  --dataset data/locomo10.json \
  --results-dir "${BENCHMARK_RESULTS_ROOT}" \
  --answer-model "${ANSWER_MODEL}" \
  --answer-backend vllm-kv \
  --vector-backend jasper \
  --top-k 50 \
  --context-window "${CONTEXT_WINDOW}" \
  --kv-gpu-memory-utilization 0.30 \
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
  --kv-gpu-memory-utilization 0.30 \
  --max-samples 10 \
  --log-every 1 \
  --skip-judge \
  --run-id "${PREFIX_RUN_ID}"
```

Both runs write query-start-to-answer-complete latency as `metrics.query_to_answer_ms`.
This is a single stopwatch around query embedding, vector retrieval, prompt/KV composition, and answer generation.
It does not include memory storage/index construction, KV precompute, vLLM startup, or judging.

TTFT metrics come from vLLM request timing on the real answer `generate()` call.

```bash
DRY_RUN=1 BENCHMARK_RESULTS_ROOT="${BENCHMARK_RESULTS_ROOT}" bash scripts/full_run.sh
BENCHMARK_RESULTS_ROOT="${BENCHMARK_RESULTS_ROOT}" bash scripts/full_run.sh
```

`scripts/full_run_cpu_store.sh` repeats only the vllm-kv+jasper grid with the pinned-host KV store (`--kv-store-backend cpu-pinned`).
Its run ids use the `kvcpu` marker (e.g. `llama-kvcpu-mem0-jasper10-k50-s0-<stamp>`), so they never collide with the standard sweep and remain judge-discoverable.
The vllm-prefix baselines never touch the KV store, so compare cpu-store runs against the standard sweep's rows.
Tune the staging pool with `KV_STAGING_SLOTS` (default 4).


## 4. Judge Accuracy


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

`--judge-only` fills only rows that are still unjudged, preserves already judged rows, and regenerates `summary.json`.
Add `--rejudge` with `--judge-only` to replace existing judge results for every row in the run.

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

Read the primary metrics:

```bash
cat "${BENCHMARK_RESULTS_ROOT}/${KV_RUN_ID}/summary.json"
cat "${BENCHMARK_RESULTS_ROOT}/${PREFIX_RUN_ID}/summary.json"
```

Primary summary metrics:

- `metrics.accuracy`: judged answer quality.
- `metrics.time_to_first_token_ms`: in-process vLLM time to first token from the real answer generation, when vLLM exposes request timing metrics.
- `metrics.query_to_first_token_ms`: query-start-to-generate-start wall time plus vLLM time to first token, when vLLM exposes request timing metrics.
- `metrics.query_to_answer_ms`: query embedding, retrieval, prompt/KV composition, and full answer generation measured with one stopwatch.
- `metrics.sample_setup_time_ms`: per-sample setup before the first query, including memory/index construction, KV precompute when applicable, and sample activation.
- Mem0 fact identity fields in `query_metrics.csv`: `memory_retrieved_fact_ids` and `memory_retrieved_fact_text_hashes` identify the exact retrieved records and content without duplicating fact text.
- Context-window fields: `memory_context_window`, `memory_context_turn_ids`, `memory_context_turn_count`, `memory_context_encoding_tokens_total`, `memory_context_encoding_tokens_max`, `memory_context_encoding_truncated_tokens`, and `memory_context_text_tokens` expose the preceding turns used for KV fact encoding or rendered in the prefix prompt.
- Mem0 setup metrics in `summary.json` and `sample_setup_metrics.csv`: `memory_input_turn_count`, `memory_inferred_record_count`, and `memory_fact_catalog_loaded` record immutable catalog replay; `preembedding.json` records extraction-cache hits and misses.
- `metrics.vector_db_query_time_ms`: raw backend vector search latency.
- Jasper graph and embedding metrics in `summary.json` and `sample_setup_metrics.csv`: `jasper_graph_gpu_mb` is the packed serialized graph size matching Jasper `total_file_size`, while `jasper_embedding_matrix_gpu_logical_mb` and `jasper_embedding_matrix_cpu_mb` describe embedding matrix storage.
- Llama KV chunk metrics in `summary.json` and `sample_setup_metrics.csv`: `llama_kv_total_tensor_gpu_mb`, `llama_kv_chunk_tensor_gpu_mb`, `llama_kv_prefix_tensor_gpu_mb`, and `llama_kv_chunk_map_cpu_mb`.


## 6. Run Multi-User Throughput

The throughput benchmark measures multi-user serving performance over the LoCoMo dataset across four conditions:

- `no_memory`: question-only prompts, the upper-bound baseline.
- `mem0_qdrant`: per-request query embedding + Qdrant top-k retrieval over the sample's Mem0-extracted facts, injected as prompt text.
- `mem0_jasper`: the same retrieval pipeline on the Jasper vector store, prompt injection.
- `kv_injection`: the identical Jasper retrieval per request, but the retrieved facts' pre-encoded chunked-RoPE KV is composed and injected through the GPU connector instead of prompt tokens.

Run all configured model aliases:

```bash
MODELS="llama mistral qwen" bash scripts/full_throughput.sh
```

`scripts/full_throughput_cpu_store.sh` repeats only the `kv_injection` condition with the pinned-host KV store (`--kv-store-backend cpu-pinned`), since the other conditions never touch the KV store.
Its run ids are prefixed `throughput-cpu-<stamp>`; compare the rows against the standard sweep's `kv_injection` rows via the `kv_store_backend` and `kv_h2d_*` columns.
Tune the staging pool with `KV_STAGING_SLOTS` (default 4).
