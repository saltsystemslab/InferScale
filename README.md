# GPU-Native KV Injection for Personalized LLM Serving

This repository runs a LoCoMo benchmark comparison between in-process vLLM answer backends.

- `vllm-kv`: retrieved Mem0 facts are encoded with the package's chunked-RoPE implementation, then injected through KV Injection.
- `vllm-prefix`: the same type of retrieved Mem0 facts are included as a normal vLLM prompt injection.

## 1. Requirements

Benchmark runs target a Linux GPU host; the reference environment is a Runpod container with the persistent `/workspace` partition.

- GPU: one NVIDIA GPU with CUDA >=12.8
- Python >=3.10,<3.14
- Hugging Face API key (`HF_TOKEN` for gated models such as Llama 3.1) and an OpenAI API key for `text-embedding-3-small` embedding calls.

We configure all default parameters to run on a RTX Pro 6000 GPU with 96 GB of VRAM.

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

## 3. Install

```bash
bash scripts/setup_remote.sh
```

`scripts/setup_remote.sh` initializes the `jasperpy` submodule, downloads the LoCoMo dataset when missing, installs the pinned Python stack, 
builds the Jasper native library, and extracts all Mem0 facts.

Activate the environment before running benchmark commands:

```bash
source .venv/bin/activate
```

## 4. Run Experiments

Now we are ready to run answer generation.

```bash
bash scripts/full_run.sh
```

To run the KV injection grid with the CPU KV store run:
```bash
bash scripts/full_run_cpu_store.sh
```

## 5. Judge Accuracy

For local Gemma/vLLM judging on the same GPU, start the judge after answer runs finish:

```bash
source .venv/bin/activate
bash scripts/serve_vllm.sh
```

Then judge each run from another shell:

```bash
STAMP=<stamp> bash scripts/judge.sh
```

## 6. Compare Results

```bash
cat "${BENCHMARK_RESULTS_ROOT}/${KV_RUN_ID}/summary.json"
cat "${BENCHMARK_RESULTS_ROOT}/${PREFIX_RUN_ID}/summary.json"
```

Primary summary metrics:

- `metrics.accuracy`: judged answer quality.
- `metrics.time_to_first_token_ms`: in-process vLLM time to first token from the real answer generation.
- `metrics.query_to_first_token_ms`: query-start-to-generate-start wall time plus vLLM time to first token.
- `metrics.query_to_answer_ms`: query embedding, retrieval, prompt/KV composition, and full answer generation.
- `metrics.sample_setup_time_ms`: per-sample setup before the first query, including memory/index construction, KV precompute when applicable, and sample activation.

## 7. Run Multi-User Throughput

The throughput benchmark measures multi-user serving performance over the LoCoMo dataset:

```bash
bash scripts/full_throughput.sh
```

To run the KV injection grid with the CPU KV store run:
```bash
bash scripts/full_throughput_cpu_store.sh
```

## License

This repository is released under the MIT License; see [LICENSE](LICENSE).
