# LoCoMo vLLM Plugin Benchmark

This repository contains a self-contained LoCoMo benchmark harness for:

- vLLM OpenAI-compatible chat completions for baseline and plugin servers.
- Full-conversation LoCoMo prompts by default, shared across baseline and plugin runs.
- A plain baseline vLLM judge for both baseline and plugin outputs.
- Accuracy and answer API latency summaries for comparing separate result directories.

The default model target is `shuyuej/Llama-3.3-70B-Instruct-GPTQ`. Local development is CPU-only; remote GPU machines should run vLLM. The old Jasper-backed retrieval path remains available with `--context-mode retrieval`.

Quick local test:

```bash
python -m pip install -e ".[dev]"
pytest
```

Remote setup, serving, and full benchmark commands are in [docs/locomo_vllm_jasper.md](docs/locomo_vllm_jasper.md).
