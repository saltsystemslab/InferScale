# InferScale: GPU-Native KV Injection for Personalized LLM Serving

This repository runs a LoCoMo benchmark comparison between in-process vLLM answer backends.

- `vllm-kv`: retrieved Mem0 facts are encoded with the package's chunked-RoPE implementation, then injected directly into the KV cache.
- `vllm-prefix`: the same retrieved Mem0 facts are included as a normal prompt injection.

## 1. Requirements

Benchmark runs target a Linux GPU host; the reference environment is a Runpod container with the persistent `/workspace` partition.

- GPU: one NVIDIA GPU with CUDA >=12.8.
- Python >=3.10,<3.14.
- CMake and the CUDA toolkit, used to build the `jasperpy` submodule.
- Hugging Face API key (`HF_TOKEN` for gated models such as Llama 3.1) and an OpenAI API key for `text-embedding-3-small` embedding calls.

We configure all default parameters to run on an RTX Pro 6000 GPU with 96 GB of VRAM.

## 2. Setup

```bash
cp .env.example .env
```

Edit `.env` for your session.
The common values are:

- `BENCHMARK_RUNTIME_ROOT=/workspace`
- `CUDA_MODULE=` for Runpod containers without environment modules
- `OPENAI_API_KEY=...` for embeddings and Mem0 inference
- `HF_TOKEN=...` if the model is gated

By default, runtime storage is rooted at `${BENCHMARK_RUNTIME_ROOT:-/workspace}` on Runpod:

- `${BENCHMARK_RUNTIME_ROOT}/.cache` for embeddings, Mem0/Jasper files, model downloads, and build caches.
- `${BENCHMARK_RUNTIME_ROOT}/results` for benchmark outputs.
- `${BENCHMARK_RUNTIME_ROOT}/tmp` for temporary files.

`source scripts/load_env.sh` prepares those directories and points the repo `.cache` entry at the runtime cache.

Load the environment in each shell that will run project commands:

```bash
source scripts/load_env.sh
```

## 3. Optimize Jasper

There is a minor optimization we can make to Jasper. First initialize and update the Jasper submodule:

```bash
git submodule update --init --recursive jasperpy
```

To apply the optimization, edit `jasperpy/include/jasper/index/graph.cuh` and change line 70 to:

```cpp
static constexpr index_t vectors_per_segment = 1u << 12;
```

## 4. Install

```bash
bash scripts/setup_remote.sh
```

`scripts/setup_remote.sh` initializes the `jasperpy` submodule, downloads the LoCoMo dataset when missing, installs the Python environment, builds the Jasper library, and extracts the Mem0 facts for every answer model.
Fact extraction serves each answer model on a vLLM server; set `SKIP_EXTRACTION=1` to defer it and run `bash scripts/extract_facts.sh` separately.

Activate the environment before running benchmark commands:

```bash
source .venv/bin/activate
```

## 5. Run Experiments

Now we are ready to run answer generation.

```bash
bash scripts/full_run.sh
```

To repeat the KV injection grid with the CPU KV store, run:

```bash
bash scripts/full_run_cpu_store.sh
```

To run the throughput experiments:

```bash
bash scripts/full_throughput.sh
```

To opt in to the GPU-resident Jasper result-ID to packed-KV selection path:

```bash
bash scripts/full_throughput.sh --jasper-device-kv-selection
```

To repeat the `kv_injection` condition with the CPU KV store, run:

```bash
bash scripts/full_throughput_cpu_store.sh
```

## 6. Judge Accuracy

For local Gemma/vLLM judging on the same GPU, start the judge after answer runs finish:

```bash
source .venv/bin/activate
bash scripts/serve_vllm.sh
```

Then judge each run from another shell that has sourced `scripts/load_env.sh`:

```bash
STAMP=<stamp> bash scripts/judge.sh
```

`STAMP` is the sweep stamp printed by `scripts/full_run.sh`, also visible in the `sweep-logs-<stamp>` directory name.

## 7. Compare Results

Each run writes to `${BENCHMARK_RESULTS_ROOT}/<run-id>/`, where the run id encodes the swept axes:
`<model>-kv-mem0-jasper10-k<topk>-s<window>-<stamp>` for KV runs and `<model>-prefix-mem0-<vector>10-k<topk>-s0-<stamp>` for the prompt baselines.

```bash
ls "${BENCHMARK_RESULTS_ROOT}"
cat "${BENCHMARK_RESULTS_ROOT}/<run-id>/summary.json"
```

Primary summary metrics:

- `metrics.accuracy`: judged answer quality.
- `metrics.time_to_first_token_ms`: in-process vLLM time to first token from the real answer generation.
- `metrics.query_to_first_token_ms`: query-start-to-generate-start wall time plus vLLM time to first token.
- `metrics.query_to_answer_ms`: query embedding, retrieval, prompt/KV composition, and full answer generation.
- `metrics.sample_setup_time_ms`: per-sample setup before the first query, including memory/index construction, KV precompute when applicable, and sample activation.

## 8. MultiHop-RAG Benchmark (standalone RAG)

`rag-jasper-bench` evaluates the core InferScale pipeline on standard RAG benchmarks without Mem0 fact extraction, starting with MultiHop-RAG (609 news articles, 2,556 multi-hop queries).
Each document is chunked into 1024-token chunks, every chunk and query is embedded with `text-embedding-3-small`, and retrieval is top-k (default k=15) over one shared Jasper index.
Each chunk's KV is precomputed once with an encoding-only prefix of its 5 preceding same-document chunks into a per-chunk disk cache, and composed at query time with chunked-RoPE repositioning; `vllm-prefix` runs the identical chunk token ids as a plain prompt baseline.
At answer time the full corpus chunk KV is loaded from that cache into host RAM.
The code lives in `src/rag_bench/` and reuses the LoCoMo pipeline's KV encoder, injection connector, Jasper store, and embedding cache.

Run the stages in order after sections 1 to 4 (the same `.venv` provides `rag-jasper-bench`):

```bash
bash scripts/rag/setup_data.sh
rag-jasper-bench --estimate-only --answer-model llama
bash scripts/rag/preembed.sh
bash scripts/rag/precompute_kv.sh
bash scripts/rag/full_run.sh
```

Preembedding needs `OPENAI_API_KEY`; answer runs read the embedding cache and make no embedding API calls.
The KV precompute is resumable per chunk; interrupt and rerun freely.
The sweep defaults to `MODELS="llama"` and `TOPKS="15"` and runs both `vllm-kv` and `vllm-prefix` per cell with `--skip-judge`; override with `MODELS`, `TOPKS`, `RAG_WINDOW`, or `RAG_CHUNK_SIZE`.

Judge with the same local Gemma server as the LoCoMo runs, using the RAG-specific judge script:

```bash
bash scripts/serve_vllm.sh
STAMP=<stamp> bash scripts/rag/judge.sh
```

## License

This repository is released under the BSD 3-Clause License; see [LICENSE](LICENSE).
