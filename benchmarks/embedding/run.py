"""Benchmark a local Qwen embedding model on the facts of one Mem0 fact catalog directory."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from benchmarks.common.atomic_files import atomic_write_text
from benchmarks.common.stats import percentile
from benchmarks.embedding.corpus import load_corpus

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
QWEN3_EMBEDDING_MAX_LENGTH = 32768
DEFAULT_BATCH_SIZES = (1, 8, 32, 64, 128)
LATENCY_PERCENTILES = (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))
REPORT_SCHEMA_VERSION = 1


def distribution(values: list[float]) -> dict[str, float]:
    """Mean and the reported percentiles of a sample."""
    ordered = sorted(values)
    summary = {"mean": statistics.mean(ordered)}
    for name, fraction in LATENCY_PERCENTILES:
        summary[name] = percentile(ordered, fraction)
    return summary


def measure(
    embedder: Any,
    texts: list[str],
    lengths: list[int],
    *,
    batch_size: int,
    warmup: int,
    repeats: int,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Time every batch of ``texts`` ``repeats`` times after ``warmup`` untimed batches."""
    if not texts or len(texts) != len(lengths):
        raise ValueError("Expected nonempty texts with matching token lengths")
    if min(batch_size, warmup, repeats) < 1:
        raise ValueError("batch_size, warmup, and repeats must be positive")
    starts = list(range(0, len(texts), batch_size))
    _warm_up(embedder, texts, starts, batch_size=batch_size, warmup=warmup)
    embedder.reset_peak_memory()
    batches = [
        _timed_batch(
            embedder, texts, lengths, start, batch_size=batch_size, repeat=repeat, clock=clock
        )
        for repeat in range(repeats)
        for start in starts
    ]
    seconds = sum(row["latency_ms"] for row in batches) / 1000
    facts = sum(row["facts"] for row in batches)
    tokens = sum(row["tokens"] for row in batches)
    return {
        "batch_size": batch_size,
        "effective_max_batch_size": min(batch_size, len(texts)),
        "warmup_batches": warmup,
        "repeats": repeats,
        "measured_batches": len(batches),
        "total_facts": facts,
        "total_tokens": tokens,
        "embedding_seconds": seconds,
        "facts_per_second": facts / seconds,
        "tokens_per_second": tokens / seconds,
        "amortized_ms_per_fact": seconds * 1000 / facts,
        "batch_latency_ms": distribution([row["latency_ms"] for row in batches]),
        "peak_cuda_allocated_bytes": embedder.peak_memory_bytes(),
        "batches": batches,
    }


def _warm_up(
    embedder: Any, texts: list[str], starts: list[int], *, batch_size: int, warmup: int
) -> None:
    for step in range(warmup):
        start = starts[step % len(starts)]
        batch = texts[start : start + batch_size]
        vectors = embedder.encode(batch)
        embedder.synchronize()
        embedder.validate(vectors, len(batch))
        del vectors


def _timed_batch(
    embedder: Any,
    texts: list[str],
    lengths: list[int],
    start: int,
    *,
    batch_size: int,
    repeat: int,
    clock: Callable[[], float],
) -> dict[str, Any]:
    batch = texts[start : start + batch_size]
    embedder.synchronize()
    began = clock()
    vectors = embedder.encode(batch)
    embedder.synchronize()
    elapsed = clock() - began
    embedder.validate(vectors, len(batch))
    del vectors
    return {
        "repeat": repeat,
        "start": start,
        "facts": len(batch),
        "tokens": sum(lengths[start : start + batch_size]),
        "latency_ms": elapsed * 1000,
    }


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.max_length > QWEN3_EMBEDDING_MAX_LENGTH:
        parser.error(f"--max-length must be <= {QWEN3_EMBEDDING_MAX_LENGTH} for {DEFAULT_MODEL}")
    texts, corpus = load_corpus(args.catalog_dir, seed=args.seed, max_facts=args.max_facts)
    if args.inspect:
        print(json.dumps(corpus, indent=2))
        return
    output = _output_path(parser, args.output)
    logger.info("Loading {}; corpus contains {} facts", args.model, len(texts))
    from benchmarks.embedding.model import QwenEmbedder

    embedder = QwenEmbedder(
        model=args.model,
        revision=args.revision,
        device=args.device,
        dtype=args.dtype,
        attention=args.attention,
        max_length=args.max_length,
    )
    lengths = embedder.token_lengths(texts)
    corpus["token_lengths"] = {**distribution(lengths), "min": min(lengths), "max": max(lengths)}
    report = _report_header(args, embedder.metadata, corpus)
    print("batch  facts/s   tokens/s   p50 batch ms   p95 batch ms   amortized ms/fact")
    for size in report["requested_batch_sizes"]:
        row = measure(
            embedder, texts, lengths, batch_size=size, warmup=args.warmup, repeats=args.repeats
        )
        report["results"].append(row)
        report["complete"] = len(report["results"]) == len(report["requested_batch_sizes"])
        atomic_write_text(output, json.dumps(report, indent=2) + "\n")
        print(
            f"{size:5d}  {row['facts_per_second']:8.1f}  {row['tokens_per_second']:9.1f}  "
            f"{row['batch_latency_ms']['p50']:13.3f}  {row['batch_latency_ms']['p95']:13.3f}  "
            f"{row['amortized_ms_per_fact']:17.3f}"
        )
    print(f"Saved {output}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen embeddings of memory catalog facts"
    )
    parser.add_argument(
        "--catalog-dir",
        type=Path,
        required=True,
        help="One leaf directory of Mem0 fact catalog JSON files (not recursive)",
    )
    parser.add_argument(
        "--inspect", action="store_true", help="Inspect corpus without loading a model"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto"
    )
    parser.add_argument(
        "--attention", choices=["sdpa", "eager", "flash_attention_2"], default="sdpa"
    )
    parser.add_argument("--max-length", type=_positive_int, default=QWEN3_EMBEDDING_MAX_LENGTH)
    parser.add_argument(
        "--batch-sizes", type=_positive_int, nargs="+", default=list(DEFAULT_BATCH_SIZES)
    )
    parser.add_argument("--warmup", type=_positive_int, default=5)
    parser.add_argument("--repeats", type=_positive_int, default=3)
    parser.add_argument("--max-facts", type=_positive_int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _output_path(parser: argparse.ArgumentParser, requested: Path | None) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = requested or Path("results/embedding") / f"{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        parser.error(f"Output already exists: {output}")
    return output


def _report_header(
    args: argparse.Namespace, model_metadata: dict[str, Any], corpus: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "model": model_metadata,
        "corpus": corpus,
        "timing_scope": (
            "tokenization + host/device transfer + forward + pooling + normalization + CPU output"
        ),
        "scheduling": (
            "sequential batches, fixed seeded shuffle, no length sorting or embedding cache"
        ),
        "requested_batch_sizes": list(dict.fromkeys(args.batch_sizes)),
        "complete": False,
        "results": [],
    }


if __name__ == "__main__":
    main()
