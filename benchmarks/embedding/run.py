from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.embedding.corpus import load_corpus


def percentiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    result = {"mean": statistics.mean(ordered)}
    for name, fraction in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99)):
        index = (len(ordered) - 1) * fraction
        low = int(index)
        high = min(low + 1, len(ordered) - 1)
        result[name] = ordered[low] + (ordered[high] - ordered[low]) * (index - low)
    return result


def measure(embedder: Any, texts: list[str], lengths: list[int], *, batch_size: int,
            warmup: int, repeats: int, clock=time.perf_counter) -> dict[str, Any]:
    if not texts or len(texts) != len(lengths):
        raise ValueError("Expected nonempty texts with matching token lengths")
    if min(batch_size, warmup, repeats) < 1:
        raise ValueError("batch_size, warmup, and repeats must be positive")
    starts = list(range(0, len(texts), batch_size))
    for step in range(warmup):
        start = starts[step % len(starts)]
        batch = texts[start:start + batch_size]
        vectors = embedder.encode(batch)
        embedder.synchronize()
        embedder.validate(vectors, len(batch))
        del vectors
    embedder.reset_peak_memory()
    batches = []
    for repeat in range(repeats):
        for start in starts:
            batch = texts[start:start + batch_size]
            embedder.synchronize()
            began = clock()
            vectors = embedder.encode(batch)
            embedder.synchronize()
            elapsed = clock() - began
            embedder.validate(vectors, len(batch))
            del vectors
            batches.append({"repeat": repeat, "start": start, "facts": len(batch),
                            "tokens": sum(lengths[start:start + batch_size]),
                            "latency_ms": elapsed * 1000})
    seconds = sum(row["latency_ms"] for row in batches) / 1000
    facts = sum(row["facts"] for row in batches)
    tokens = sum(row["tokens"] for row in batches)
    return {
        "batch_size": batch_size, "effective_max_batch_size": min(batch_size, len(texts)),
        "warmup_batches": warmup, "repeats": repeats, "measured_batches": len(batches),
        "total_facts": facts, "total_tokens": tokens, "embedding_seconds": seconds,
        "facts_per_second": facts / seconds, "tokens_per_second": tokens / seconds,
        "amortized_ms_per_fact": seconds * 1000 / facts,
        "batch_latency_ms": percentiles([row["latency_ms"] for row in batches]),
        "peak_cuda_allocated_bytes": embedder.peak_memory_bytes(), "batches": batches,
    }


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Qwen embeddings of memory catalog facts")
    parser.add_argument("--catalog-dir", type=Path, required=True,
                        help="One leaf directory of Mem0 fact catalog JSON files (not recursive)")
    parser.add_argument("--inspect", action="store_true", help="Inspect corpus without loading a model")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--attention", choices=["sdpa", "eager", "flash_attention_2"], default="sdpa")
    parser.add_argument("--max-length", type=positive_int, default=32768)
    parser.add_argument("--batch-sizes", type=positive_int, nargs="+", default=[1, 8, 32, 64, 128])
    parser.add_argument("--warmup", type=positive_int, default=5)
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--max-facts", type=positive_int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_length > 32768:
        parser.error("--max-length must be <= 32768 for Qwen3-Embedding-0.6B")
    texts, corpus = load_corpus(args.catalog_dir, seed=args.seed, max_facts=args.max_facts)
    if args.inspect:
        print(json.dumps(corpus, indent=2))
        return
    from benchmarks.embedding.model import QwenEmbedder

    output = args.output or Path("results/embedding") / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        parser.error(f"Output already exists: {output}")
    print(f"Loading {args.model}; corpus contains {len(texts)} facts", flush=True)
    embedder = QwenEmbedder(model=args.model, revision=args.revision, device=args.device,
                            dtype=args.dtype, attention=args.attention, max_length=args.max_length)
    lengths = embedder.token_lengths(texts)
    corpus["token_lengths"] = {**percentiles(lengths), "min": min(lengths), "max": max(lengths)}
    report = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "model": embedder.metadata, "corpus": corpus,
        "timing_scope": "tokenization + host/device transfer + forward + pooling + normalization + CPU output",
        "scheduling": "sequential batches, fixed seeded shuffle, no length sorting or embedding cache",
        "requested_batch_sizes": list(dict.fromkeys(args.batch_sizes)), "complete": False,
        "results": [],
    }
    print("batch  facts/s   tokens/s   p50 batch ms   p95 batch ms   amortized ms/fact", flush=True)
    for size in dict.fromkeys(args.batch_sizes):
        row = measure(embedder, texts, lengths, batch_size=size, warmup=args.warmup, repeats=args.repeats)
        report["results"].append(row)
        report["complete"] = len(report["results"]) == len(report["requested_batch_sizes"])
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output)
        print(f"{size:5d}  {row['facts_per_second']:8.1f}  {row['tokens_per_second']:9.1f}  "
              f"{row['batch_latency_ms']['p50']:13.3f}  {row['batch_latency_ms']['p95']:13.3f}  "
              f"{row['amortized_ms_per_fact']:17.3f}", flush=True)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
