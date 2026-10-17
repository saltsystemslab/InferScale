# TTFT race: Mem0 vs InferScale

Answer the same LoCoMo questions with Mem0 and with InferScale, then replay one question as a GIF.

- `Mem0 (k=100)`: the 100 retrieved Mem0 facts are read as prompt text
- `InferScale (k=100, w=50)`: the KV precomputed for the 100 retrieved facts is injected, each fact encoded with its 50 preceding turns.

## Run

Run from the repository root on the GPU host after sections 1 to 4 of the [benchmark README](../README.md).

```bash
bash benchmarks/ttft_race/run.sh
```
