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
from benchmarks.common.logging import configure_logging
from benchmarks.common.stats import percentile
from benchmarks.embedding.corpus import load_corpus

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
QWEN3_EMBEDDING_MAX_LENGTH = 32768
DEFAULT_BATCH_SIZES = (1, 8, 32, 64, 128)
LATENCY_PERCENTILES = (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))
REPORT_SCHEMA_VERSION = 1
MS_PER_SECOND = 1000
DEFAULT_OUTPUT_FOLDER = Path("results/embedding")
OUTPUT_STAMP_FORMAT = "%Y%m%dT%H%M%S%fZ"
TABLE_HEADER = "batch  facts/s   tokens/s   p50 batch ms   p95 batch ms   amortized ms/fact"
TABLE_ROW = "{:5d}  {:8.1f}  {:9.1f}  {:13.3f}  {:13.3f}  {:17.3f}"


def distribution(values: list[float]) -> dict[str, float]:
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
    if not texts or len(texts) != len(lengths):
        raise ValueError("Expected nonempty texts with matching token lengths")
    if min(batch_size, warmup, repeats) < 1:
        raise ValueError("batch_size, warmup, and repeats must be positive")
    batch_starts = list(range(0, len(texts), batch_size))
    _warm_up(embedder, texts, batch_starts, batch_size=batch_size, warmup=warmup)
    embedder.reset_peak_memory()
    batches = [
        _timed_batch(
            embedder, texts, lengths, start, batch_size=batch_size, repeat=repeat, clock=clock
        )
        for repeat in range(repeats)
        for start in batch_starts
    ]
    embedding_seconds = sum(batch["latency_ms"] for batch in batches) / MS_PER_SECOND
    total_facts = sum(batch["facts"] for batch in batches)
    total_tokens = sum(batch["tokens"] for batch in batches)
    return {
        "batch_size": batch_size,
        "effective_max_batch_size": min(batch_size, len(texts)),
        "warmup_batches": warmup,
        "repeats": repeats,
        "measured_batches": len(batches),
        "total_facts": total_facts,
        "total_tokens": total_tokens,
        "embedding_seconds": embedding_seconds,
        "facts_per_second": total_facts / embedding_seconds,
        "tokens_per_second": total_tokens / embedding_seconds,
        "amortized_ms_per_fact": embedding_seconds * MS_PER_SECOND / total_facts,
        "batch_latency_ms": distribution([batch["latency_ms"] for batch in batches]),
        "peak_cuda_allocated_bytes": embedder.peak_memory_bytes(),
        "batches": batches,
    }


def _warm_up(
    embedder: Any, texts: list[str], batch_starts: list[int], *, batch_size: int, warmup: int
) -> None:
    for step in range(warmup):
        start = batch_starts[step % len(batch_starts)]
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
        "latency_ms": elapsed * MS_PER_SECOND,
    }


def main() -> None:
    configure_logging()
    parser = _build_parser()
    args = parser.parse_args()
    if args.max_length > QWEN3_EMBEDDING_MAX_LENGTH:
        parser.error(f"--max-length must be <= {QWEN3_EMBEDDING_MAX_LENGTH} for {DEFAULT_MODEL}")
    texts, corpus_manifest = load_corpus(args.catalog_dir, seed=args.seed, max_facts=args.max_facts)
    if args.inspect:
        logger.info(json.dumps(corpus_manifest, indent=2))
        return
    output_path = _new_output_path(parser, args.output)
    logger.info("Loading {}; corpus contains {} facts", args.model, len(texts))
    embedder = _load_embedder(args)
    token_lengths = embedder.token_lengths(texts)
    corpus_manifest["token_lengths"] = _token_length_summary(token_lengths)
    report = _report_header(args, embedder.metadata, corpus_manifest)
    logger.info(TABLE_HEADER)
    for batch_size in report["requested_batch_sizes"]:
        result = measure(
            embedder,
            texts,
            token_lengths,
            batch_size=batch_size,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        report["results"].append(result)
        report["complete"] = len(report["results"]) == len(report["requested_batch_sizes"])
        atomic_write_text(output_path, json.dumps(report, indent=2) + "\n")
        _log_table_row(batch_size, result)
    logger.info("Saved {}", output_path)


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


def _positive_int(text: str) -> int:
    number = int(text)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _new_output_path(parser: argparse.ArgumentParser, requested: Path | None) -> Path:
    stamp = datetime.now(timezone.utc).strftime(OUTPUT_STAMP_FORMAT)
    output_path = requested or DEFAULT_OUTPUT_FOLDER / f"{stamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        parser.error(f"Output already exists: {output_path}")
    return output_path


def _load_embedder(args: argparse.Namespace) -> Any:
    from benchmarks.embedding.model import QwenEmbedder

    return QwenEmbedder(
        model=args.model,
        revision=args.revision,
        device=args.device,
        dtype=args.dtype,
        attention=args.attention,
        max_length=args.max_length,
    )


def _token_length_summary(token_lengths: list[int]) -> dict[str, float]:
    return {**distribution(token_lengths), "min": min(token_lengths), "max": max(token_lengths)}


def _report_header(
    args: argparse.Namespace, model_metadata: dict[str, Any], corpus_manifest: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "model": model_metadata,
        "corpus": corpus_manifest,
        "timing_scope": (
            "tokenization + host/device transfer + forward + pooling + normalization + CPU output"
        ),
        "scheduling": (
            "sequential batches, fixed seeded shuffle, no length sorting or embedding cache"
        ),
        "requested_batch_sizes": _without_repeats(args.batch_sizes),
        "complete": False,
        "results": [],
    }


def _without_repeats(batch_sizes: list[int]) -> list[int]:
    return list(dict.fromkeys(batch_sizes))


def _log_table_row(batch_size: int, result: dict[str, Any]) -> None:
    logger.info(
        TABLE_ROW,
        batch_size,
        result["facts_per_second"],
        result["tokens_per_second"],
        result["batch_latency_ms"]["p50"],
        result["batch_latency_ms"]["p95"],
        result["amortized_ms_per_fact"],
    )


if __name__ == "__main__":
    main()
