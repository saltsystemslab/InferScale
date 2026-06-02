# LoCoMo vLLM Plugin Benchmark

This harness runs LoCoMo question answering against OpenAI-compatible vLLM endpoints to compare a plain baseline server with a plugin-enabled server. By default, LoCoMo turns are imported into real Mem0 with `infer=False` and Mem0 uses Jasper as its vector store; answers receive Mem0-retrieved context plus the question. Accuracy is judged by the plain baseline vLLM server, and answer API latency remains the primary latency metric.

## 1. Create the Remote Environment

Run from the repository root on the GPU machine:

```bash
bash scripts/setup_remote.sh
source .venv/bin/activate
```

The setup script also builds Jasper retrieval support. The default Mem0 path uses Mem0 embeddings and Jasper indexes. The `jasper` Python package is installed from `jasperpy/python`.
By default, `scripts/setup_remote.sh` loads `cuda/12.8`, installs PyTorch CUDA 12.8 wheels from the versions pinned in `constraints-cu128.txt`, and builds Jasper with the same toolkit. Override `CUDA_MODULE`, `PYTORCH_INDEX`, or `CONSTRAINTS_FILE` only if you intentionally want a different compatible stack.

## 2. Serve the Baseline vLLM Model

In a separate shell:

```bash
export CUDA_MODULE=cuda/12.8
export VLLM_TP=1
export VLLM_GPU_MEMORY_UTILIZATION=0.80
export VLLM_MAX_MODEL_LEN=32768
export VLLM_DTYPE=auto
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_API_KEY=token-abc123
bash scripts/serve_vllm.sh
```

The script expands to:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --trust-remote-code \
  --dtype ${VLLM_DTYPE:-auto} \
  --max-model-len ${VLLM_MAX_MODEL_LEN:-32768} \
  --tensor-parallel-size ${VLLM_TP:-1} \
  --gpu-memory-utilization ${VLLM_GPU_MEMORY_UTILIZATION:-0.80} \
  --api-key ${VLLM_API_KEY:-token-abc123}
```

The script keeps Hugging Face, vLLM, Torch, Triton, TorchInductor, PyTorch extension builds, CUDA JIT, FlashInfer, and temp files under `.cache/` and `tmp/` in the repository by default. It also disables vLLM usage stats by default to avoid writing `~/.config/vllm/usage_stats.json` on quota-limited home directories. FlashInfer 0.6.x reads `FLASHINFER_WORKSPACE_BASE`, so the script sets that to the repository root and FlashInfer writes under `.cache/flashinfer/`. The script defaults `VLLM_USE_FLASHINFER_SAMPLER=0` because the vLLM 0.10.2 FlashInfer sampler can reject RTX Blackwell/SM 12.x under the CUDA 12.8 pin; vLLM falls back to the PyTorch-native sampler. Use a larger `VLLM_TP` only if the allocation needs multiple GPUs or latency improves. Increase `VLLM_MAX_MODEL_LEN` if full conversation prompts are rejected as too long. Official Meta Llama models may require Hugging Face access approval and login or `HF_TOKEN`. If you override `VLLM_MODEL` with a quantized model, set `VLLM_QUANTIZATION` to the matching vLLM quantization mode, such as `gptq`. Keep `CUDA_MODULE=cuda/12.8` for the pinned setup unless you rebuild both Jasper and the vLLM/PyTorch stack for another CUDA version.

Smoke check:

```bash
curl --noproxy '*' \
  -H "Authorization: Bearer ${VLLM_API_KEY:-token-abc123}" \
  http://127.0.0.1:8000/v1/models
```

If the smoke check returns a Squid proxy page with `ERR_ACCESS_DENIED`, the request was sent through the cluster proxy instead of directly to the local vLLM server. Use `--noproxy '*'` with `curl`, set `NO_PROXY`/`no_proxy` for Python clients, and prefer `127.0.0.1` for local vLLM URLs. If the no-proxy request says `connection refused`, verify the benchmark shell and vLLM shell are on the same compute node with `hostname`.

## 3. Get LoCoMo Data

Place `locomo10.json` at `data/locomo10.json`. The loader expects the public LoCoMo shape: each sample has session turns under `conversation.session_N` and question records under `qa`.

Download the dataset on the data transfer node, then run the benchmark from the GPU node:

```bash
cd /path/to/benchmark-jasper
mkdir -p data
curl -L \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json \
  -o data/locomo10.json
```

## 4. Run a Small Baseline

```bash
export VLLM_BASE_URL=http://127.0.0.1:8000/v1
export JUDGE_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=<your-openai-key>
export BENCHMARK_CACHE_ROOT=/projects/SaltSystemsLab/peter/benchmark-jasper/.cache
export MEM0_DIR="${BENCHMARK_CACHE_ROOT}/mem0"
export NO_PROXY=localhost,127.0.0.1,::1
export no_proxy="${NO_PROXY}"
locomo-jasper-bench \
  --mode baseline \
  --dataset data/locomo10.json \
  --results-dir results \
  --max-samples 1 \
  --max-questions 3 \
  --log-every 1 \
  --vllm-command "bash scripts/serve_vllm.sh"
```

The benchmark logs progress with Loguru. By default it logs every 5 questions, plus sample updates. Use `--log-every 1` for a small smoke run, increase it for full runs, or set `LOCOMO_LOG_EVERY`. Set `LOCOMO_LOG_LEVEL=DEBUG` or `LOCOMO_LOG_LEVEL=WARNING` to adjust verbosity. `locomo-jasper-bench` defaults `MEM0_DIR` to `${BENCHMARK_CACHE_ROOT:-<repo>/.cache}/mem0` so Mem0 does not write `~/.mem0` on quota-limited home directories.

Outputs are written under `results/<run_id>/`:

- `config.json`
- `system.json`
- `predictions.jsonl`
- `summary.json`

The default `--context-mode mem0` creates one Mem0 instance per LoCoMo sample under `results/<run_id>/mem0/<sample_id>/`. It adds each formatted turn with `infer=False`, preserving sample, session, turn, speaker, and timestamp metadata, then finalizes the Jasper graph before questions are answered. `latency_ms.memory_search_ms` measures Mem0 search wall time, while `latency_ms.answer_generation_ms` remains the baseline-vs-plugin latency metric. Mem0 index build and add time are recorded in index metadata, not answer API latency.

Mem0 embeddings use the OpenAI embedder by default, so `OPENAI_API_KEY` must be set for `--context-mode mem0`. Use `--context-mode full` for a no-memory full-transcript baseline that does not build embeddings or Jasper indexes. `--context-mode retrieval` is accepted as a deprecated alias for `mem0`.

If `--context-mode mem0` fails `python scripts/smoke_jasper.py` with `cudaErrorUnsupportedPtxVersion`, rebuild Jasper for the GPU's native architecture instead of relying on PTX JIT fallback:

```bash
source .venv/bin/activate
export JASPER_CUDA_ARCHITECTURES=native
cmake -S jasperpy -B jasperpy/build \
  -DJASPER_BUILD_FFI=ON \
  -DJASPER_BUILD_CMD=ON \
  -DCMAKE_CUDA_ARCHITECTURES="${JASPER_CUDA_ARCHITECTURES}"
cmake --build jasperpy/build --parallel
cmake --install jasperpy/build
python scripts/smoke_jasper.py
```

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

## 7. Plugin Variant

Start the plugin-enabled vLLM server separately, then run the same benchmark against that server. Keep the LoCoMo data, prompt mode, judge settings, generation settings, and max token settings unchanged so baseline and plugin runs are comparable:

```bash
locomo-jasper-bench \
  --mode plugin \
  --dataset data/locomo10.json \
  --llm-base-url http://plugin-host:8000/v1 \
  --judge-base-url http://baseline-host:8000/v1 \
  --results-dir results \
  --run-id plugin-$(date -u +%Y%m%dT%H%M%SZ)
```

If the plugin requires request-body options, pass them to answer generation only:

```bash
--llm-extra-body-json '{"your_plugin_option": true}'
```

Compare `summary.json` files from the baseline and plugin run directories. Primary latency is `latency_avg_ms.answer_generation_ms` and accuracy is `accuracy`.

## 8. CPU-Only Dry Run

For local harness checks without CUDA, vLLM, or OpenAI embeddings:

```bash
python -m pip install -e ".[dev]"
pytest
```
