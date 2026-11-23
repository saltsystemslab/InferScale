from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

from benchmarks.rag.data_types import RagChunk, RagDocument, chunk_id_for
from inferscale.v1.kv.tokenization import encode_text_no_special


def render_document(doc: RagDocument) -> str:
    return (
        f"Title: {doc.title}\n"
        f"Source: {doc.source}\n"
        f"Published: {doc.published_at}\n"
        f"Category: {doc.category}\n\n"
        f"{doc.body.strip()}\n"
    )


def chunk_document(doc: RagDocument, *, tokenizer: Any, chunk_size: int) -> list[RagChunk]:
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
    hasher = hashlib.sha256()
    for doc in docs:
        render_digest = hashlib.sha256(render_document(doc).encode("utf-8")).hexdigest()
        for part in (doc.doc_id, render_digest):
            hasher.update(part.encode("utf-8"))
            hasher.update(b"\x00")
    return hasher.hexdigest()[:20]


def _decode(tokenizer: Any, token_ids: list[int]) -> str:
    # vLLM and transformers tokenizers share the decode call but no base class.
    return str(tokenizer.decode(token_ids))
