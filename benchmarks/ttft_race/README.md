# TTFT race: Mem0 vs InferScale

Answer the same LoCoMo questions with Mem0 and with InferScale, then replay one question as a GIF: two panels on a shared clock, each showing its first-token time and its answer.

- `Mem0 (k=100)`: the 100 retrieved Mem0 facts are read as prompt text (`prompt-injection` with Qdrant retrieval).
- `InferScale (k=100, w=50)`: the KV precomputed for the 100 retrieved facts is injected (`kv-injection` with Jasper retrieval), each fact encoded with its 50 preceding turns.

## Run

Run from the repository root on the GPU host after sections 1 to 4 of the [benchmark README](../README.md).
The race reads the Mem0 fact catalogs and cached embeddings that `bash scripts/extract_facts.sh` produces for the answer model.

```bash
bash benchmarks/ttft_race/run.sh
```

The script measures Mem0 and then InferScale, each in its own process so that it owns the whole GPU, and writes `mem0.json`, `inferscale.json`, and `ttft_race.gif` to `${BENCHMARK_RESULTS_ROOT}/ttft-race/`.

Each measurement is the memory benchmark itself on the first 11 answerable questions of the first LoCoMo conversation.
The first 3 questions absorb the engine's one-off startup costs and are left out of the timeline, and the next 8 are timed.
The InferScale measurement first encodes the conversation's facts, which takes a few minutes, unless `bash scripts/precompute_kv_chunks.sh` already cached them.

## What the GIF shows

TTFT is the benchmark's `time_to_first_token_ms`: vLLM's own time from receiving the answer request to its first token.
The clock starts when the engine receives the request, so retrieval and KV composition happen before it.
Each measurement prints its benchmark run directory, whose `predictions.jsonl` keeps `query_to_first_token_ms` for the time from the question to the first token.
The engine generates offline and times only the first token and the finished answer, so the GIF spaces the words in between evenly.

## Steps by hand

```bash
source scripts/load_env.sh
source "${VENV_DIR}/bin/activate"
python -m benchmarks.ttft_race.run measure --mode mem0 --out mem0.json
python -m benchmarks.ttft_race.run measure --mode inferscale --out inferscale.json
python -m benchmarks.ttft_race.run gif inferscale.json mem0.json --out ttft_race.gif
```

`measure --config configs/memory/accuracy-latency/qwen.json` selects another model, and `measure --questions 16` times more questions.
`gif` prints both first-token times for every timed question and replays the one with the median speedup.
`gif --question 3` replays another question, and `gif --slowdown 10` changes the playback speed.
The retrieval depth and the context window are `TOP_K` and `CONTEXT_WINDOW` in [run.py](run.py).
The `gif` step needs only Pillow, so the two timelines can also be rendered on another machine.
