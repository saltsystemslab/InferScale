# InferScale Benchmarks

This directory runs the memory and RAG benchmarks on InferScale using Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct, and Qwen3-14B.

- `kv-injection`: each Mem0 fact is encoded with chunked RoPE behind the facts from the `w` conversation turns before its source turn, and the KV of the retrieved facts is injected directly into the KV cache
- `prompt-injection`: retrieved Mem0 facts are injected as prompt text
- `lmcache-blend`: retrieved Mem0 facts are served through LMCache CacheBlend, which reuses each fact's cached KV at its position in the prompt and recomputes a fraction of the tokens

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

The LMCache CacheBlend baseline, described in section 6:

```bash
bash scripts/full_run_lmcache.sh
bash scripts/full_throughput_lmcache.sh
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
`<model>-kv-mem0-jasper10-k<topk>-s<window>-<stamp>` for KV runs, `<model>-prompt-injection-mem0-<vector>10-k<topk>-s0-<stamp>` for the prompt-injection baselines, and `<model>-lmcache-blend-mem0-qdrant10-k<topk>-s0-<stamp>` for the LMCache CacheBlend baseline.

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

The LMCache CacheBlend baseline on the same corpora:

```bash
bash scripts/rag/full_run_lmcache.sh
```

Judge the results:

```bash
bash scripts/serve_vllm.sh
bash scripts/rag/judge.sh
```

For Qwen3-Embedding-0.6B latency and throughput on the memory benchmark's cached facts, see the [fact embedding benchmark](embedding/README.md).

## 6. LMCache CacheBlend Baseline

The `lmcache-blend` arm serves the retrieved units (Mem0 facts, or RAG chunks) through [LMCache](https://github.com/LMCache/LMCache) CacheBlend, which stores each unit's KV in a pinned-CPU cache and reuses it at any position in the prompt, recomputing a fraction of the tokens to repair cross-attention.

The arm runs LMCache's in-process vLLM connector through `BlendConnector` in `benchmarks/common/lmcache/connector.py`, which is loaded by module path like InferScale's own connector.
The subclass defers the creation of LMCache's blender until the model is loaded, because vLLM builds connectors before loading the model (upstream LMCache issue 4131), and it records each request's matched token count and load time.

vLLM prefix caching is off for this arm, because LMCache's blending replaces it.

Before timing, the arm warms the cache: it clears LMCache, then sends the scaffold header and every unit alone as a one-token prompt, after checking that the units and the per-request tails fit `max_local_cpu_size_gb`.

## License

This repository is released under the BSD 3-Clause License; see [LICENSE](../LICENSE).
