# Qwen3-14B Five-Pod Sweep

These five scripts divide the Qwen3-14B sweep by top-k so each pod runs four Jasper KV windows and one Qdrant prefix baseline.
All pods may write to the same network-mounted `BENCHMARK_RESULTS_ROOT` because every run ID includes its assigned top-k.

Choose one shared stamp before launching the pods and use that exact value everywhere:

```bash
export RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
export BENCHMARK_RESULTS_ROOT=/shared/results
```

Run one command on each pod:

```bash
bash scripts/qwen3-14b-5pod/run_k5.sh
bash scripts/qwen3-14b-5pod/run_k10.sh
bash scripts/qwen3-14b-5pod/run_k20.sh
bash scripts/qwen3-14b-5pod/run_k50.sh
bash scripts/qwen3-14b-5pod/run_k100.sh
```

Preview any partition by setting `DRY_RUN=1`.
The scripts preserve the existing `WINDOWS`, `DATASET`, and other environment overrides supported by `scripts/full_run.sh`.
