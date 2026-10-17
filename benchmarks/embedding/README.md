# Fact embedding benchmark

Measure local Qwen3-Embedding-0.6B latency and throughput using the Mem0 facts embedded with OpenAI during the memory benchmarks.

## Setup and corpus selection

Run from the repository root on the GPU host after the memory fact extraction.

```bash
python -m pip install -e '.[embedding]'
```
The default remote cache is `/workspace/.cache/mem0-inference`; use the configured cache root if it differs.

```bash
find /workspace/.cache/mem0-inference/fact-catalogs -type f -name '*.json' -exec dirname {} \; | sort -u
```

Choose one leaf directory matching the extraction model, endpoint, and OpenAI embedding configuration of the memory run being compared.

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
