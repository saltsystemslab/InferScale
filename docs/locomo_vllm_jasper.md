# LoCoMo vLLM Plugin Benchmark

This harness runs LoCoMo question answering against OpenAI-compatible vLLM endpoints to compare a plain baseline server with a plugin-enabled server. By default, LoCoMo turns are imported into real Mem0 with `infer=False` and Mem0 uses Jasper as its vector store; answers receive Mem0-retrieved context plus the question. Accuracy is judged by the plain baseline vLLM server, and answer API latency remains the primary latency metric.

The remote workflow keeps cache, temp, and result files under `/scratch/$USER/benchmark-jasper`. The repository should only keep lightweight source files plus a `.cache` symlink pointing at scratch.

## 1. Create the Remote Environment

Run from the repository root on the GPU machine:

```bash
cd /projects/SaltSystemsLab/peter/benchmark-jasper

export SCRATCH_ROOT=/scratch/$USER/benchmark-jasper
FRESH_REMOTE_BUILD=1 bash scripts/setup_remote.sh

source .venv/bin/activate
source scripts/scratch_env.sh
```

`FRESH_REMOTE_BUILD=1` removes the old local `.venv`, `.cache`, `tmp`, `jasperpy/build`, and installed Jasper `.so` files before rebuilding. Omit it if you want to keep an existing build. `scripts/scratch_env.sh` creates `${SCRATCH_ROOT}/cache`, `${SCRATCH_ROOT}/results`, `${SCRATCH_ROOT}/tmp`, and `${SCRATCH_ROOT}/pip`; exports `BENCHMARK_CACHE_ROOT`, `BENCHMARK_RESULTS_ROOT`, `MEM0_DIR`, `TMPDIR`, `PIP_CACHE_DIR`, and `XDG_CACHE_HOME`; and refreshes `.cache` as a symlink to `${SCRATCH_ROOT}/cache`.

The setup script also builds Jasper retrieval support. The default Mem0 path uses Mem0 embeddings and Jasper indexes. The `jasper` Python package is installed from `jasperpy/python`.

By default, `scripts/setup_remote.sh` loads `cuda/12.8`, installs PyTorch CUDA 12.8 wheels from the versions pinned in `constraints-cu128.txt`, and builds Jasper with the same toolkit. The pinned stack is PyTorch `2.8.0`, vLLM `0.10.2`, and Transformers `4.56.1`. Override `CUDA_MODULE`, `PYTORCH_INDEX`, or `CONSTRAINTS_FILE` only if you intentionally want a different compatible stack.

Verify the environment:

```bash
python - <<'PY'
import torch, vllm, transformers
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("vllm:", vllm.__version__)
print("transformers:", transformers.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY

python scripts/smoke_jasper.py
```

## 2. Serve the Baseline vLLM Model

In a separate tmux window on the same compute node:

```bash
cd /projects/SaltSystemsLab/peter/benchmark-jasper
source .venv/bin/activate
source scripts/scratch_env.sh

export CUDA_MODULE=cuda/12.8
export VLLM_TP=1
export VLLM_GPU_MEMORY_UTILIZATION=0.80
export VLLM_MAX_MODEL_LEN=32768
export VLLM_DTYPE=auto
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_API_KEY=token-abc123
export VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct

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

`scripts/serve_vllm.sh` sources `scripts/scratch_env.sh` by default, then keeps Hugging Face, vLLM, Torch, Triton, TorchInductor, PyTorch extension builds, CUDA JIT, FlashInfer, and temp files under scratch-backed cache paths. It also disables vLLM usage stats by default to avoid writing `~/.config/vllm/usage_stats.json` on quota-limited home directories. The script defaults `VLLM_USE_FLASHINFER_SAMPLER=0` because the vLLM 0.10.2 FlashInfer sampler can reject RTX Blackwell/SM 12.x under the CUDA 12.8 pin; vLLM falls back to the PyTorch-native sampler. Use a larger `VLLM_TP` only if the allocation needs multiple GPUs or latency improves. Increase `VLLM_MAX_MODEL_LEN` if full conversation prompts are rejected as too long. Official Meta Llama models may require Hugging Face access approval and login or `HF_TOKEN`.

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
cd /projects/SaltSystemsLab/peter/benchmark-jasper
mkdir -p data
curl -L \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json \
  -o data/locomo10.json
```

## 4. Run a Small Baseline

In a second tmux window on the same compute node:

```bash
cd /projects/SaltSystemsLab/peter/benchmark-jasper
source .venv/bin/activate
source scripts/scratch_env.sh

export VLLM_BASE_URL=http://127.0.0.1:8000/v1
export JUDGE_BASE_URL=http://127.0.0.1:8000/v1
export VLLM_API_KEY=token-abc123
export JUDGE_API_KEY=token-abc123
export OPENAI_API_KEY=<your-openai-key>
export NO_PROXY=localhost,127.0.0.1,::1
export no_proxy="${NO_PROXY}"

locomo-jasper-bench \
  --mode baseline \
  --dataset data/locomo10.json \
  --results-dir "${SCRATCH_ROOT}/results" \
  --vector-backend jasper \
  --max-samples 1 \
  --max-questions 3 \
  --stream \
  --log-every 1 \
  --vllm-command "bash scripts/serve_vllm.sh"
```

After `source scripts/scratch_env.sh`, `locomo-jasper-bench` also defaults `--results-dir` to `${BENCHMARK_RESULTS_ROOT}` and `MEM0_DIR` to `${BENCHMARK_CACHE_ROOT}/mem0`, so Mem0 does not write `~/.mem0` on quota-limited home directories. The explicit `--results-dir` above makes the scratch target visible in the command.

The benchmark logs progress with Loguru. By default it logs every 5 questions, plus sample updates. Use `--log-every 1` for a small smoke run, increase it for full runs, or set `LOCOMO_LOG_EVERY`. Set `LOCOMO_LOG_LEVEL=DEBUG` or `LOCOMO_LOG_LEVEL=WARNING` to adjust verbosity. Use `--stream` when you want TTFT metrics; without streaming, `vllm.answer.ttft_ms` is `null`.

Jasper stores vectors and payload metadata under `${SCRATCH_ROOT}/results/<run_id>/mem0/<sample_id>/`. The Jasper search path caps the graph search `k` to the available vector count and configured beam width, so filtered searches do not request an invalid overfetch size. This prevents the repeated top-hit behavior where a question returned the same memory UUID multiple times. The adapter reports inner-product scores as positive dot products, and if Jasper returns fewer payload-backed ordinals than requested, the benchmark fills missing neighbors from the exact normalized vectors already stored on disk. The Mem0 adapter also normalizes `user_id` metadata and treats the per-sample store path as satisfying `filters={"user_id": <sample_id>}`, so broad sample filters do not drop every retrieved hit when Mem0 nests metadata differently. During question answering, the benchmark embeds the query with Mem0's embedder and searches the adapter directly; this avoids Mem0's extra post-search semantic threshold from discarding all Jasper candidates before they reach the prompt. These are Python adapter/runner fixes; they do not require rebuilding Jasper CUDA/C++ artifacts. If SQLite reports `disk I/O error`, verify that `--results-dir` and `BENCHMARK_CACHE_ROOT` are on writable scratch storage with quota available.

Outputs are written under `${SCRATCH_ROOT}/results/<run_id>/`:

- `config.json`
- `system.json`
- `predictions.jsonl`
- `summary.json`

The default `--context-mode mem0` creates one Mem0 instance per LoCoMo sample under the run directory. It adds each formatted turn with `infer=False`, preserving sample, session, turn, speaker, and timestamp metadata, then finalizes the vector index before questions are answered. `latency_ms.memory_search_ms` measures Mem0 search wall time, while `latency_ms.answer_generation_ms` remains the baseline-vs-plugin latency metric. Index build and add time are recorded in index metadata, not answer API latency.

Mem0 embeddings use the OpenAI embedder by default, so `OPENAI_API_KEY` must be set for `--context-mode mem0`. The project installs `mem0ai[nlp]` because Mem0's local memory processing can require spaCy NLP dependencies. Use `--context-mode full` for a no-memory full-transcript baseline that does not build embeddings or Jasper indexes. `--context-mode retrieval` is accepted as a deprecated alias for `mem0`.

## 5. Acceptance Checks

Run these from the repository root after setup:

```bash
ls -ld .cache
df -h /projects/SaltSystemsLab/peter/benchmark-jasper
du -sh .cache/
du -sh "${SCRATCH_ROOT}/results"
```

`ls -ld .cache` should show a symlink to `/scratch/$USER/benchmark-jasper/cache`. The vLLM model cache and benchmark outputs should grow under scratch, not under the project quota.

If `--context-mode mem0` fails `python scripts/smoke_jasper.py` with `cudaErrorUnsupportedPtxVersion`, rebuild Jasper for the GPU's native architecture:

```bash
source .venv/bin/activate
source scripts/scratch_env.sh
export JASPER_CUDA_ARCHITECTURES=native
cmake -S jasperpy -B jasperpy/build \
  -DJASPER_BUILD_FFI=ON \
  -DJASPER_BUILD_CMD=ON \
  -DCMAKE_CUDA_ARCHITECTURES="${JASPER_CUDA_ARCHITECTURES}"
cmake --build jasperpy/build --parallel
cmake --install jasperpy/build
python scripts/smoke_jasper.py
```

## 6. Full Baseline

```bash
cd /projects/SaltSystemsLab/peter/benchmark-jasper
source .venv/bin/activate
source scripts/scratch_env.sh

export VLLM_BASE_URL=http://127.0.0.1:8000/v1
export JUDGE_BASE_URL=http://127.0.0.1:8000/v1
export VLLM_API_KEY=token-abc123
export JUDGE_API_KEY=token-abc123
export OPENAI_API_KEY=<your-openai-key>
export NO_PROXY=localhost,127.0.0.1,::1
export no_proxy="${NO_PROXY}"

locomo-jasper-bench \
  --mode baseline \
  --dataset data/locomo10.json \
  --results-dir "${SCRATCH_ROOT}/results" \
  --vector-backend jasper \
  --stream \
  --run-id baseline-vllm-$(date -u +%Y%m%dT%H%M%SZ) \
  --vllm-command "bash scripts/serve_vllm.sh"
```

Add `--skip-judge` if you want to write predictions first and judge later.

## 7. Jasper vs Qdrant

Run both commands from the benchmark tmux window while the baseline vLLM server is running. Keep the same `OPENAI_API_KEY`, model, dataset, `top_k`, question count, and `--stream` setting so TTFT, throughput, retrieval timing, and accuracy are comparable.

```bash
source .venv/bin/activate
source scripts/scratch_env.sh

export VLLM_BASE_URL=http://127.0.0.1:8000/v1
export JUDGE_BASE_URL=http://127.0.0.1:8000/v1
export VLLM_API_KEY=token-abc123
export JUDGE_API_KEY=token-abc123
export OPENAI_API_KEY=<your-openai-key>
export NO_PROXY=localhost,127.0.0.1,::1
export no_proxy="${NO_PROXY}"

locomo-jasper-bench \
  --mode baseline \
  --dataset data/locomo10.json \
  --results-dir "${SCRATCH_ROOT}/results" \
  --vector-backend jasper \
  --max-samples 1 \
  --max-questions 3 \
  --stream \
  --log-every 1 \
  --run-id jasper-k-cap-$(date -u +%Y%m%dT%H%M%SZ)

locomo-jasper-bench \
  --mode baseline \
  --dataset data/locomo10.json \
  --results-dir "${SCRATCH_ROOT}/results" \
  --vector-backend qdrant \
  --max-samples 1 \
  --max-questions 3 \
  --stream \
  --log-every 1 \
  --run-id qdrant-local-$(date -u +%Y%m%dT%H%M%SZ)
```

Compare each run's `summary.json`. Key fields are `accuracy`, `by_category`, `latency_ms`, `vllm.answer.ttft_ms`, `vllm.answer.output_tokens_per_sec`, `vector_store.search_time_ms`, and `retrieval.questions_with_duplicate_ids`. In `predictions.jsonl`, answerable questions should have non-empty `retrieved_memories`; `retrieval.questions_with_duplicate_ids` should stay at `0` unless there are genuinely duplicated memory IDs in the source data.

## 8. Re-Judge Saved Predictions

The judge should always be the plain baseline vLLM server, even when evaluating plugin outputs.

```bash
source .venv/bin/activate
source scripts/scratch_env.sh

locomo-jasper-bench \
  --mode evaluate-only \
  --predictions "${SCRATCH_ROOT}/results/<run_id>/predictions.jsonl" \
  --results-dir "${SCRATCH_ROOT}/results" \
  --run-id judge-$(date -u +%Y%m%dT%H%M%SZ)
```

## 9. Plugin Variant

Start the plugin-enabled vLLM server separately, then run the same benchmark against that server. Keep the LoCoMo data, prompt mode, judge settings, generation settings, and max token settings unchanged so baseline and plugin runs are comparable:

```bash
source .venv/bin/activate
source scripts/scratch_env.sh

locomo-jasper-bench \
  --mode plugin \
  --dataset data/locomo10.json \
  --llm-base-url http://plugin-host:8000/v1 \
  --judge-base-url http://baseline-host:8000/v1 \
  --results-dir "${SCRATCH_ROOT}/results" \
  --run-id plugin-$(date -u +%Y%m%dT%H%M%SZ)
```

If the plugin requires request-body options, pass them to answer generation only:

```bash
--llm-extra-body-json '{"your_plugin_option": true}'
```

Compare `summary.json` files from the baseline and plugin run directories. Primary latency is `latency_avg_ms.answer_generation_ms`, TTFT is under `vllm.answer.ttft_ms`, throughput is under `vllm.answer.output_tokens_per_sec`, and accuracy is `accuracy`.

## 10. CPU-Only Dry Run

For local harness checks without CUDA, vLLM, or OpenAI embeddings:

```bash
python -m pip install -e ".[dev]"
pytest
```
