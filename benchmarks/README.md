# InferScale Benchmarks

This directory runs the memory and RAG benchmarks on InferScale using Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct, and Qwen3-14B.

- `kv-injection`: retrieved Mem0 facts are encoded with chunked RoPE, then injected directly into the KV cache
- `prompt-injection`: retrieved Mem0 facts are injected as prompt text

## 1. Installation

Follow the installation steps in the [top-level README](../README.md).

## 2. Run Experiments

```bash
bash scripts/full_run.sh
```

Multi-user throughput:

```bash
bash scripts/full_throughput.sh
```

Supplemental experiments with the CPU KV store:

```bash
bash scripts/full_run_cpu_store.sh
bash scripts/full_throughput_cpu_store.sh
```

## 3. Judge Accuracy

Judging uses a local Gemma judge served by vLLM on the same GPU, so start it after the answer runs finish:

```bash
bash scripts/serve_vllm.sh
```

Judge the results:

```bash
bash scripts/judge.sh
```

## 4. Compare Results

Each run writes to `${BENCHMARK_RESULTS_ROOT}/<run-id>/`, where the run id encodes the swept axes:
`<model>-kv-mem0-jasper10-k<topk>-s<window>-<stamp>` for KV runs and `<model>-prompt-injection-mem0-<vector>10-k<topk>-s0-<stamp>` for the prompt-injection baselines.

```bash
ls "${BENCHMARK_RESULTS_ROOT}"
cat "${BENCHMARK_RESULTS_ROOT}/<run-id>/summary.json"
```

## 5. RAG Benchmark

Run in order after installation:

```bash
bash scripts/rag/setup_data.sh
bash scripts/rag/estimate.sh
bash scripts/rag/preembed.sh
bash scripts/rag/precompute_kv.sh
bash scripts/rag/full_run.sh
```

Judge the results:

```bash
bash scripts/serve_vllm.sh
bash scripts/rag/judge.sh
```

For Qwen3-Embedding-0.6B latency and throughput on the memory benchmark's cached facts, see the [fact embedding benchmark](embedding/README.md).

## License

This repository is released under the BSD 3-Clause License; see [LICENSE](../LICENSE).
