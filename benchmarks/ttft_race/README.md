# TTFT race: InferScale vs Mem0 vs LMCache CacheBlend

Answer the same LoCoMo questions with Mem0, with LMCache CacheBlend, and with InferScale, then replay one question as a GIF.

- `Mem0 (k=100)`: the 100 retrieved Mem0 facts are read as prompt text
- `LMCache CacheBlend (k=100)`: the 100 retrieved Mem0 facts are laid out as LMCache CacheBlend segments, each fact's KV precomputed in isolation and blended at prefill
- `InferScale (k=100, w=50)`: the KV precomputed for the 100 retrieved facts is injected, each fact encoded with its 50 preceding turns.

## Run

Run from the repository root on the GPU host after the installation in the [top-level README](../../README.md).
The LMCache side needs `lmcache==0.4.4`, which the installation pins.

```bash
bash benchmarks/ttft_race/run.sh
```

The script measures each side in its own process and writes the GIF, one panel per side, under `${BENCHMARK_RESULTS_ROOT}/ttft-race/`.
Run `python -m benchmarks.ttft_race.run --help` to time more questions, use another memory benchmark JSON, or replay a different question.
