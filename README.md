# LoCoMo vLLM Plugin Benchmark

This repository contains a self-contained LoCoMo benchmark harness for:

- vLLM OpenAI-compatible chat completions for baseline and plugin servers.
- Real Mem0 memory search by default, with Jasper, NumPy, or local Qdrant vector search.
- A plain baseline vLLM judge for both baseline and plugin outputs.
- Accuracy and answer API latency summaries for comparing separate result directories.

The default model target is `meta-llama/Llama-3.1-8B-Instruct`. Local development is CPU-only for tests; remote GPU machines should run vLLM and Jasper. Official Meta Llama models may require Hugging Face access approval and login or `HF_TOKEN`. LoCoMo turns are imported into Mem0 with `infer=False` for deterministic raw storage. Jasper is the default vector backend and caps graph search requests to the beam width/vector count so Mem0 retrieval does not repeat the same top memory or return empty results from invalid overfetch sizes. The Mem0 adapter also normalizes `user_id` metadata for per-sample stores; this is Python-side and does not require a Jasper rebuild. Use `--vector-backend qdrant` to benchmark local qdrant-client retrieval against Jasper, and use `--stream` when you need TTFT metrics. Use `--context-mode full` for no-memory full-transcript baseline runs. `--context-mode retrieval` is accepted as a deprecated alias for `mem0`.

`OPENAI_API_KEY` is required for Mem0 embeddings unless you run `--context-mode full`.

Quick local test:

```bash
python -m pip install -e ".[dev]"
pytest
```

Remote setup, serving, and full benchmark commands are in [docs/locomo_vllm_jasper.md](docs/locomo_vllm_jasper.md). The remote flow uses `/scratch/$USER/benchmark-jasper` for caches, temp files, and results, with `.cache` in the repo as a symlink.
