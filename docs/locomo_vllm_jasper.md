# LoCoMo vLLM + Jasper Benchmark

This harness runs LoCoMo question answering with a Mem0-style memory layer backed by Jasper. The LLM and judge use OpenAI-compatible vLLM endpoints. The default embedding provider is OpenAI `text-embedding-3-small`.

## 1. Create the Remote Environment

Run from the repository root on the GPU machine:

```bash
bash scripts/setup_remote.sh
source .venv/bin/activate
```

If the Jasper build succeeds, the `jasper` Python package will be installed from `jasperpy/python`.

## 2. Serve the Baseline vLLM Model

In a separate shell:

```bash
export CUDA_MODULE=cuda/12.9
export VLLM_TP=1
export VLLM_GPU_MEMORY_UTILIZATION=0.80
export VLLM_API_KEY=token-abc123
bash scripts/serve_vllm.sh
```

The script expands to:

```bash
vllm serve shuyuej/Llama-3.3-70B-Instruct-GPTQ \
  --quantization gptq \
  --trust-remote-code \
  --dtype float16 \
  --max-model-len 4096 \
  --tensor-parallel-size ${VLLM_TP:-1} \
  --gpu-memory-utilization ${VLLM_GPU_MEMORY_UTILIZATION:-0.80} \
  --api-key ${VLLM_API_KEY:-token-abc123}
```

The script keeps Hugging Face, vLLM, Torch, FlashInfer, and temp files under `.cache/` and `tmp/` in the repository by default. FlashInfer 0.6.x reads `FLASHINFER_WORKSPACE_BASE`, so the script sets that to the repository root and FlashInfer writes under `.cache/flashinfer/`. Use a larger `VLLM_TP` only if the allocation needs multiple GPUs or latency improves.

Smoke check:

```bash
curl -H "Authorization: Bearer ${VLLM_API_KEY:-token-abc123}" http://localhost:8000/v1/models
python scripts/smoke_jasper.py
```

## 3. Get LoCoMo Data

Place `locomo10.json` at `data/locomo10.json`. The loader expects the public LoCoMo shape: each sample has session turns under `conversation.session_N` and question records under `qa`.

## 4. Run a Small Baseline

```bash
export OPENAI_API_KEY=...
locomo-jasper-bench \
  --mode baseline \
  --dataset data/locomo10.json \
  --results-dir results \
  --max-samples 1 \
  --max-questions 3 \
  --vllm-command "bash scripts/serve_vllm.sh"
```

Outputs are written under `results/<run_id>/`:

- `config.json`
- `system.json`
- `predictions.jsonl`
- `summary.json`
- `indexes/<sample_id>/vectors.npy`, `payloads.sqlite`, and `jasper.graph`

## 5. Full Baseline

```bash
locomo-jasper-bench \
  --mode baseline \
  --dataset data/locomo10.json \
  --results-dir results \
  --run-id baseline-vllm-$(date -u +%Y%m%dT%H%M%SZ) \
  --vllm-command "bash scripts/serve_vllm.sh"
```

Add `--skip-judge` if you want to write predictions first and judge later.

## 6. Re-Judge Saved Predictions

The judge should always be the plain baseline vLLM server, even when evaluating plugin outputs.

```bash
locomo-jasper-bench \
  --mode evaluate-only \
  --predictions results/<run_id>/predictions.jsonl \
  --results-dir results \
  --run-id judge-$(date -u +%Y%m%dT%H%M%SZ)
```

## 7. Plugin Variant Later

Start the plugin-enabled vLLM server separately, then keep embeddings, prompts, LoCoMo data, Jasper settings, and judge settings unchanged:

```bash
locomo-jasper-bench \
  --mode plugin \
  --dataset data/locomo10.json \
  --llm-base-url http://localhost:8000/v1 \
  --judge-base-url http://baseline-host:8000/v1 \
  --results-dir results \
  --run-id plugin-$(date -u +%Y%m%dT%H%M%SZ)
```

If the plugin requires request-body options, pass them to answer generation only:

```bash
--llm-extra-body-json '{"your_plugin_option": true}'
```

## 8. CPU-Only Dry Run

For local harness checks without CUDA, vLLM, or OpenAI embeddings, use the NumPy vector backend and hash embeddings with mocked tests:

```bash
python -m pip install -e ".[dev]"
pytest
```
