"""The estimate stage: a CPU-only projection of corpus, chunk, embedding, and KV sizes."""

from __future__ import annotations

from typing import Any

from loguru import logger

from benchmarks.rag.config import RagBenchConfig
from benchmarks.rag.corpus import RagCorpus, load_rag_corpus
from benchmarks.rag.data_types import RagChunk

GIB = 1024**3
# KV geometry (layers, kv_heads, head_dim) of the configured answer models.
# Other models are read from their transformers AutoConfig when available.
KV_GEOMETRY = {
    "meta-llama/Llama-3.1-8B-Instruct": (32, 8, 128),
    "mistralai/Mistral-7B-Instruct-v0.3": (32, 8, 128),
    "Qwen/Qwen2.5-7B-Instruct": (28, 4, 128),
    "Qwen/Qwen3-14B": (40, 8, 128),
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
    """Project the sizes a run would need, before any GPU work."""
    corpus = load_rag_corpus(config, allow_transformers_fallback=True)
    chunk_counts = [chunk.token_count for chunk in corpus.chunks]
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
        "chunk_tokens_min": min(chunk_counts),
        "chunk_tokens_max": max(chunk_counts),
        "chunks_per_doc_max": _chunks_per_doc_max(corpus.chunks),
        "embedding_texts": len(corpus.chunks) + len(corpus.queries),
        "kv_bytes_per_token": kv_bytes_per_token,
        "projected_kv_cache_bytes": _scaled(kv_bytes_per_token, corpus.total_tokens),
        "per_query_fetch_bytes": _scaled(kv_bytes_per_token, config.top_k * config.chunk_size),
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
    projected = estimate["projected_kv_cache_bytes"]
    if projected is None:
        logger.info("kv_bytes_per_token unknown: model not in the geometry table and no AutoConfig")
    else:
        logger.info(
            "Projected KV cache {:.1f} GiB (also the host RAM the cpu store needs), "
            "{:.2f} GiB fetched per query, {} encode input tokens",
            projected / GIB,
            estimate["per_query_fetch_bytes"] / GIB,
            estimate["kv_encode_input_tokens"],
        )


def _chunks_per_doc_max(chunks: list[RagChunk]) -> int:
    counts: dict[str, int] = {}
    for chunk in chunks:
        counts[chunk.doc_id] = counts.get(chunk.doc_id, 0) + 1
    return max(counts.values())


def _scaled(bytes_per_token: int | None, tokens: int) -> int | None:
    return None if bytes_per_token is None else bytes_per_token * tokens


def _encode_input_tokens(config: RagBenchConfig, corpus: RagCorpus) -> int:
    """Tokens the precompute forward passes see: each chunk with its preceding-chunk prefix."""
    return sum(
        min(
            min(config.context_window, chunk.chunk_index) * config.chunk_size + chunk.token_count,
            config.kv_max_position,
        )
        for chunk in corpus.chunks
    )


def _kv_bytes_per_token(config: RagBenchConfig) -> int | None:
    dtype_bytes = DTYPE_BYTES.get(config.kv_dtype.lower())
    if dtype_bytes is None:
        return None
    geometry = KV_GEOMETRY.get(config.model) or _kv_geometry_from_autoconfig(config.model)
    if geometry is None:
        return None
    layers, kv_heads, head_dim = geometry
    return layers * kv_heads * head_dim * 2 * dtype_bytes


def _kv_geometry_from_autoconfig(model: str) -> tuple[int, int, int] | None:
    try:
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(model)
    except Exception:
        return None
    # AutoConfig objects differ per architecture; these are the common attribute names.
    layers = getattr(hf_config, "num_hidden_layers", None)
    kv_heads = getattr(hf_config, "num_key_value_heads", None) or getattr(
        hf_config, "num_attention_heads", None
    )
    head_dim = getattr(hf_config, "head_dim", None)
    if head_dim is None:
        hidden_size = getattr(hf_config, "hidden_size", None)
        heads = getattr(hf_config, "num_attention_heads", None)
        if hidden_size and heads:
            head_dim = hidden_size // heads
    if not layers or not kv_heads or not head_dim:
        return None
    return int(layers), int(kv_heads), int(head_dim)
