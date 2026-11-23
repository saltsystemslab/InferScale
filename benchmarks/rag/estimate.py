from __future__ import annotations

from typing import Any, NamedTuple

from loguru import logger

from benchmarks.rag.config import RagBenchConfig
from benchmarks.rag.corpus import RagCorpus, load_rag_corpus
from benchmarks.rag.data_types import RagChunk

GIB = 1024**3
KEYS_AND_VALUES = 2


class KvGeometry(NamedTuple):
    layers: int
    kv_heads: int
    head_dim: int


ANSWER_MODEL_GEOMETRY = {
    "meta-llama/Llama-3.1-8B-Instruct": KvGeometry(layers=32, kv_heads=8, head_dim=128),
    "mistralai/Mistral-7B-Instruct-v0.3": KvGeometry(layers=32, kv_heads=8, head_dim=128),
    "Qwen/Qwen2.5-7B-Instruct": KvGeometry(layers=28, kv_heads=4, head_dim=128),
    "Qwen/Qwen3-14B": KvGeometry(layers=40, kv_heads=8, head_dim=128),
}
DTYPE_BYTES = {
    "bf16": 2,
    "bfloat16": 2,
    "fp16": 2,
    "float16": 2,
    "half": 2,
    "fp32": 4,
    "float32": 4,
}


def run_estimate(config: RagBenchConfig) -> dict[str, Any]:
    corpus = load_rag_corpus(config, allow_transformers_fallback=True)
    chunk_token_counts = [chunk.token_count for chunk in corpus.chunks]
    kv_bytes_per_token = _kv_bytes_per_token(config)
    estimate = {
        "dataset": config.dataset_name,
        "model": config.model,
        "chunk_size": config.chunk_size,
        "context_window": config.context_window,
        "top_k": config.top_k,
        "doc_count": len(corpus.docs),
        "query_count": len(corpus.queries),
        "chunk_count": len(corpus.chunks),
        "corpus_total_tokens": corpus.total_tokens,
        "chunk_tokens_min": min(chunk_token_counts),
        "chunk_tokens_max": max(chunk_token_counts),
        "chunks_per_doc_max": _chunks_per_doc_max(corpus.chunks),
        "embedding_texts": len(corpus.chunks) + len(corpus.queries),
        "kv_bytes_per_token": kv_bytes_per_token,
        "projected_kv_cache_bytes": _kv_bytes(kv_bytes_per_token, corpus.total_tokens),
        "per_query_fetch_bytes": _kv_bytes(kv_bytes_per_token, config.top_k * config.chunk_size),
        "kv_encode_input_tokens": _encode_input_tokens(config, corpus),
        "corpus_fingerprint": corpus.fingerprint,
    }
    _log_estimate(estimate)
    return estimate


def _log_estimate(estimate: dict[str, Any]) -> None:
    logger.info(
        "Estimate dataset={} model={} docs={} queries={} chunks={} corpus_tokens={}",
        estimate["dataset"],
        estimate["model"],
        estimate["doc_count"],
        estimate["query_count"],
        estimate["chunk_count"],
        estimate["corpus_total_tokens"],
    )
    projected_bytes = estimate["projected_kv_cache_bytes"]
    if projected_bytes is None:
        logger.info("kv_bytes_per_token unknown: model not in the geometry table and no AutoConfig")
    else:
        logger.info(
            "Projected KV cache {:.1f} GiB (also the host RAM the cpu store needs), "
            "{:.2f} GiB fetched per query, {} encode input tokens",
            projected_bytes / GIB,
            estimate["per_query_fetch_bytes"] / GIB,
            estimate["kv_encode_input_tokens"],
        )


def _chunks_per_doc_max(chunks: list[RagChunk]) -> int:
    chunk_count_by_doc: dict[str, int] = {}
    for chunk in chunks:
        chunk_count_by_doc[chunk.doc_id] = chunk_count_by_doc.get(chunk.doc_id, 0) + 1
    return max(chunk_count_by_doc.values())


def _kv_bytes(bytes_per_token: int | None, tokens: int) -> int | None:
    return None if bytes_per_token is None else bytes_per_token * tokens


def _encode_input_tokens(config: RagBenchConfig, corpus: RagCorpus) -> int:
    return sum(_encode_input_tokens_of(chunk, config) for chunk in corpus.chunks)


def _encode_input_tokens_of(chunk: RagChunk, config: RagBenchConfig) -> int:
    context_chunks = min(config.context_window, chunk.chunk_index)
    context_and_chunk_tokens = context_chunks * config.chunk_size + chunk.token_count
    return min(context_and_chunk_tokens, config.kv_max_position)


def _kv_bytes_per_token(config: RagBenchConfig) -> int | None:
    dtype_bytes = DTYPE_BYTES.get(config.kv_dtype.lower())
    if dtype_bytes is None:
        return None
    geometry = ANSWER_MODEL_GEOMETRY.get(config.model) or _kv_geometry_from_autoconfig(config.model)
    if geometry is None:
        return None
    return geometry.layers * geometry.kv_heads * geometry.head_dim * KEYS_AND_VALUES * dtype_bytes


def _kv_geometry_from_autoconfig(model: str) -> KvGeometry | None:
    try:
        from transformers import AutoConfig

        architecture_config = AutoConfig.from_pretrained(model)
    except Exception:
        return None
    layers = getattr(architecture_config, "num_hidden_layers", None)
    kv_heads = getattr(architecture_config, "num_key_value_heads", None) or getattr(
        architecture_config, "num_attention_heads", None
    )
    head_dim = _head_dim(architecture_config)
    if not layers or not kv_heads or not head_dim:
        return None
    return KvGeometry(int(layers), int(kv_heads), int(head_dim))


def _head_dim(architecture_config: Any) -> Any:
    head_dim = getattr(architecture_config, "head_dim", None)
    if head_dim is not None:
        return head_dim
    hidden_size = getattr(architecture_config, "hidden_size", None)
    attention_heads = getattr(architecture_config, "num_attention_heads", None)
    if hidden_size and attention_heads:
        return hidden_size // attention_heads
    return None
