# Fact embedding benchmark

Measure the latency and throughput of Qwen3-Embedding-0.6B, the model every benchmark retrieves with, on the Mem0 facts of the memory benchmark.

## Setup and corpus selection

Run from the repository root on the GPU host after the memory fact extraction.

```bash
python -m pip install -e '.[embedding]'
```

The fact catalogs live under the configured cache root, `${BENCHMARK_CACHE_ROOT}/mem0-inference`, which `source scripts/load_env.sh` exports.

```bash
find "${BENCHMARK_CACHE_ROOT}/mem0-inference/fact-catalogs" -type f -name '*.json' -exec dirname {} \; | sort -u
```

Choose one leaf directory matching the extraction model, endpoint, and embedding model of the memory run being compared.

```bash
CATALOG_DIR=/path/to/the/selected/catalog/leaf
python -m benchmarks.embedding.run --catalog-dir "$CATALOG_DIR" --inspect
```

## Run

```bash
python -m benchmarks.embedding.run \
  --catalog-dir "$CATALOG_DIR" \
  --device cuda \
  --batch-sizes 1 8 32 64 128 \
  --warmup 5 \
  --repeats 3 \
  --output results/embedding/qwen3-0.6b.json
```
