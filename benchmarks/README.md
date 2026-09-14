# InferScale Benchmarks

This directory runs memory and RAG baseline comparisons on InferScale between an array of model families and sizes (Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct, and Qwen3-14B).

- `kv-injection`: retrieved Mem0 facts are encoded with the package's chunked-RoPE implementation, then injected directly into the KV cache.
- `prompt-injection`: the same retrieved Mem0 facts are included as a normal prompt injection.

## 1. Requirements

Benchmark runs target a Linux GPU host; the reference environment is a Runpod container with the persistent `/workspace` partition.

- GPU: one NVIDIA GPU with CUDA >=12.8.
- Python >=3.10,<3.14.
- CMake and the CUDA toolkit, used to build the `jasperpy` submodule.
- Hugging Face API key (`HF_TOKEN` for gated models such as Llama 3.1) and an OpenAI API key for `text-embedding-3-small` embedding calls.
- A separate Runpod CPU Pod running Qdrant for memory benchmarks and fact extraction.

We configure all default parameters to run on an RTX Pro 6000 GPU with 96 GB of VRAM.

## 2. Setup

```bash
cp .env.example .env
```

Put API keys in `.env`: `OPENAI_API_KEY` for embeddings, `JUDGE_LLM_API_KEY` and optional `EXTRACTION_LLM_API_KEY` for the configured servers, and `HF_TOKEN` for gated models.
Set `QDRANT_URL` and `QDRANT_API_KEY` for the Qdrant Pod as described below.

Load credentials and export runtime paths in each shell:

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

## 4. Configure Runpod CPU Qdrant

Deploy a separate CPU Pod for Qdrant before fact extraction or memory benchmark runs.
Use these Runpod template settings:

| Field | Value |
| --- | --- |
| Template type | Pod |
| Compute type | CPU |
| Public template | Off |
| Container image | `qdrant/qdrant:v1.17.0` |
| Start command | Leave blank |
| Container disk | 10 GB |
| Persistent storage | Volume disk, 50 GB as a starting point |
| Persistent storage mount path | `/workspace` |
| HTTP port | Label `Qdrant`, port `6333` |
| TCP ports | Leave blank |
| UDP support | Off |

Add these environment variables to the Qdrant Pod:

| Name | Value |
| --- | --- |
| `QDRANT__STORAGE__STORAGE_PATH` | `/workspace/qdrant/storage` |
| `QDRANT__STORAGE__SNAPSHOTS_PATH` | `/workspace/qdrant/snapshots` |
| `QDRANT__SERVICE__API_KEY` | A long, random secret |
| `QDRANT__SERVICE__ENABLE_TLS` | `false` |

Generate the secret with `openssl rand -hex 32` and use the same value as `QDRANT_API_KEY` in the benchmark host's `.env`.

On the benchmark GPU host, set these entries in `.env`:

```dotenv
QDRANT_URL=https://YOUR_POD_ID-6333.proxy.runpod.net
QDRANT_API_KEY=YOUR_SECRET
```

Replace `YOUR_POD_ID` with the CPU Pod's ID and `YOUR_SECRET` with its configured API key.

Verify the connection before installation or fact extraction:

```bash
source scripts/load_env.sh
bash scripts/qdrant.sh check
```

## 5. Install

```bash
bash scripts/setup_remote.sh
```

Load the environment and activate the installed dependencies before benchmark commands:

```bash
source scripts/load_env.sh
source "${VENV_DIR}/bin/activate"
```

## 6. Run Experiments

```bash
bash scripts/full_run.sh
```

To run the throughput experiments:

```bash
bash scripts/full_throughput.sh
```

## 7. Judge Accuracy

For local Gemma/vLLM judging on the same GPU, start the judge after answer runs finish:

```bash
source .venv/bin/activate
bash scripts/serve_vllm.sh
```


```bash
bash scripts/judge.sh
```

## 8. Compare Results

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

## 9. RAG Benchmark

Run the stages in order after sections 1 to 5.

```bash
bash scripts/rag/setup_data.sh
bash scripts/rag/estimate.sh
bash scripts/rag/preembed.sh
bash scripts/rag/precompute_kv.sh
bash scripts/rag/full_run.sh
```

Judge with the same local Gemma server as the LoCoMo runs, using the RAG-specific judge script:

```bash
bash scripts/serve_vllm.sh
bash scripts/rag/judge.sh
```

## License

This repository is released under the BSD 3-Clause License; see [LICENSE](LICENSE).
