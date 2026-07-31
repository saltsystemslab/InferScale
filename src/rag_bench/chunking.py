from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

from locomo_jasper_bench.kv.tokenization import encode_text_no_special

from .data_types import RagChunk, RagDocument, chunk_id_for


def render_document(doc: RagDocument) -> str:
    """Render a document once, with a metadata header before the body.

    Titles, sources, and publication dates carry signal for MultiHop-RAG's
    comparison and temporal queries, so they are part of the retrievable and
    injectable text. The header is prepended once and then the whole render
    is chunked, so only a document's first chunk carries it.
    """
    return (
        f"Title: {doc.title}\n"
        f"Source: {doc.source}\n"
        f"Published: {doc.published_at}\n"
        f"Category: {doc.category}\n\n"
        f"{doc.body.strip()}\n"
    )


def chunk_document(doc: RagDocument, *, tokenizer: Any, chunk_size: int) -> list[RagChunk]:
    """Split one rendered document into contiguous chunk_size token slices.

    The final remainder chunk is kept (it has at least one token). Chunk text
    is the decode of the exact token ids so downstream embedding and metrics
    can never drift from what the KV encoder and the prompt see.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1.")
    token_ids = encode_text_no_special(tokenizer, render_document(doc))
    if not token_ids:
        raise ValueError(f"Document {doc.doc_id} rendered to zero tokens.")
    chunks: list[RagChunk] = []
    for chunk_index, start in enumerate(range(0, len(token_ids), chunk_size)):
        slice_ids = token_ids[start : start + chunk_size]
        chunks.append(
            RagChunk(
                chunk_id=chunk_id_for(doc.doc_id, chunk_index),
                doc_id=doc.doc_id,
                chunk_index=chunk_index,
                token_ids=list(slice_ids),
                text=_decode(tokenizer, slice_ids),
            )
        )
    return chunks


def chunk_corpus(
    docs: Sequence[RagDocument],
    *,
    tokenizer: Any,
    chunk_size: int,
) -> list[RagChunk]:
    chunks: list[RagChunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, tokenizer=tokenizer, chunk_size=chunk_size))
    return chunks


def chunks_by_doc(chunks: Sequence[RagChunk]) -> dict[str, list[RagChunk]]:
    grouped: dict[str, list[RagChunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.doc_id, []).append(chunk)
    for doc_chunks in grouped.values():
        doc_chunks.sort(key=lambda chunk: chunk.chunk_index)
    return grouped


def corpus_fingerprint(docs: Sequence[RagDocument]) -> str:
    """Order-sensitive content hash over the rendered corpus.

    Tokenizer-independent on purpose: the KV cache directory key combines this
    fingerprint with the resolved model id, and per-chunk loads additionally
    validate exact token ids.
    """
    hasher = hashlib.sha256()
    for doc in docs:
        render_digest = hashlib.sha256(render_document(doc).encode("utf-8")).hexdigest()
        for part in (doc.doc_id, render_digest):
            hasher.update(part.encode("utf-8"))
            hasher.update(b"\x00")
    return hasher.hexdigest()[:20]


def _decode(tokenizer: Any, token_ids: list[int]) -> str:
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise RuntimeError("Tokenizer has no decode method.")
    return str(decode(token_ids))
