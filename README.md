# InferScale: GPU-Native KV Injection for Personalized LLM Serving

The main benchmark uses 1,540 answerable LoCoMo questions on a single NVIDIA RTX PRO 6000 with 96 GB of memory, vLLM 0.19.1, and prefix caching enabled.

### Serving latency

![Serving latency for InferScale and Mem0 across three models](figures/paper/serving-latency.png)

InferScale keeps TTFT nearly flat as more memory is retrieved, while Mem0's prefill latency grows with `k`.

### End-to-end memory QA accuracy

Judged LoCoMo accuracy (%), micro-averaged over the 1,540 answerable questions.

| Model | Method | `k=5` | `k=10` | `k=20` | `k=50` |
| --- | --- | ---: | ---: | ---: | ---: |
| Llama-3.1-8B | InferScale (`w=0`) | 62.21 | 62.14 | 57.66 | 53.77 |
| Llama-3.1-8B | InferScale (`w=5`) | 60.00 | 59.81 | 59.87 | 56.82 |
| Llama-3.1-8B | InferScale (`w=20`) | 60.13 | 62.08 | 61.62 | 59.22 |
| Llama-3.1-8B | InferScale (`w=50`) | 60.06 | 61.36 | 62.66 | 60.26 |
| Llama-3.1-8B | Mem0 | 56.95 | 59.29 | 61.49 | 63.25 |
| Mistral-7B | InferScale (`w=0`) | 54.03 | 55.13 | 52.84 | 37.22 |
| Mistral-7B | InferScale (`w=5`) | 56.04 | 57.60 | 58.09 | 56.45 |
| Mistral-7B | InferScale (`w=20`) | 53.96 | 55.97 | 56.95 | 58.24 |
| Mistral-7B | InferScale (`w=50`) | 55.78 | 56.49 | 58.70 | 58.30 |
| Mistral-7B | Mem0 | 61.30 | 63.96 | 65.00 | 64.35 |
| Qwen2.5-7B | InferScale (`w=0`) | 59.74 | 56.75 | 54.22 | 50.45 |
| Qwen2.5-7B | InferScale (`w=5`) | 59.29 | 57.53 | 58.05 | 57.40 |
| Qwen2.5-7B | InferScale (`w=20`) | 60.00 | 59.68 | 58.25 | 58.64 |
| Qwen2.5-7B | InferScale (`w=50`) | 58.83 | 58.25 | 58.18 | 58.77 |
| Qwen2.5-7B | Mem0 | 60.06 | 63.64 | 65.13 | 64.61 |

 Encoding each fact in isolation (w=0) trails Mem0 and degrades as more facts are retrieved, from 62.2% to 53.8% on Llama and, most steeply, 54.0% to 37.2% on Mistral as `k` grows from 5 to 50. A context window reverses this: with w>=20, accuracy is flat-to-rising in `k` and comes within a few points of Mem0 (on Llama, 60.3% vs. 63.3% at k=50) while often exceeding it at small `k`.

### Serving throughput

![Serving throughput for InferScale and Mem0 across three models](figures/paper/serving-throughput.png)

InferScale’s throughput scales nearlinearly with the number of concurrent users, while Mem0 saturates early. At 100 user, InferScale reaches 100/104/135 QPS on Llama/Mistral/Qwen versus Mem0’s 27/23/32 QPS, a 3.7–4.5x speedup, and the gap widens with concurrency (Mem0 gains only ∼2x from 10 to 100 users, whereas InferScale gains ~4x).

### Jasper retrieval ablation

Retrieval backend's effect on end-to-end query-to-first-token latency in milliseconds.

| Model | Backend | `k=5` | `k=10` | `k=20` | `k=50` |
| --- | --- | ---: | ---: | ---: | ---: |
| Llama-3.1-8B | Jasper GPU | 47.92 | 53.13 | 60.86 | 86.56 |
| Llama-3.1-8B | Qdrant CPU | 186.15 | 190.22 | 206.79 | 236.48 |
| Mistral-7B | Jasper GPU | 39.72 | 45.78 | 56.56 | 88.41 |
| Mistral-7B | Qdrant CPU | 129.37 | 147.41 | 162.01 | 198.07 |
| Qwen2.5-7B | Jasper GPU | 51.30 | 54.87 | 63.36 | 85.24 |
| Qwen2.5-7B | Qdrant CPU | 178.81 | 182.25 | 190.83 | 224.76 |

The two backends return comparable results; the backend’s large effect is on retrieval latency.

### Memory footprint and CPU offload

Average per-conversation storage footprint in decimal MB.

| Model | Jasper GPU | Fact-ID map CPU | Fact KVs GPU |
| --- | ---: | ---: | ---: |
| Llama-3.1-8B | 13.50 | 7.97 | 4,796.67 |
| Mistral-7B v0.3 | 13.50 | 3.02 | 2,257.44 |
| Qwen2.5-7B | 13.50 | 6.48 | 1,780.10 |

The retrieval structures are negligible in size. The Jasper graph and the map together cost under 25 MB per conversation on every model. The footprint is dominated by the KV store (1.8-4.8 GB per conversation), which is also the only term that scales with model architecture, growing with the number of layers, KV heads, and head dimension.

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

## License

This repository is released under the BSD 3-Clause License; see [LICENSE](LICENSE).
