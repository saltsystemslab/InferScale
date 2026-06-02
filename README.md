# LoCoMo Jasper Benchmark

This repository contains a self-contained LoCoMo benchmark harness for:

- vLLM OpenAI-compatible chat completions as the baseline LLM.
- A Mem0-style memory wrapper that stores embeddings in a Jasper-backed vector index.
- OpenAI `text-embedding-3-small` embeddings by default.
- A plain baseline vLLM judge for both baseline and plugin runs.

The default model target is `shuyuej/Llama-3.3-70B-Instruct-GPTQ`. Local development is CPU-only; remote GPU machines should build `jasperpy` and run vLLM.

Quick local test:

```bash
python -m pip install -e ".[dev]"
pytest
```

Remote setup, serving, and full benchmark commands are in [docs/locomo_vllm_jasper.md](docs/locomo_vllm_jasper.md).
