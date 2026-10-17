"""Load a dataset and chunk its documents the way every RAG stage must see them."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

from benchmarks.rag.chunking import chunk_corpus, corpus_fingerprint
from benchmarks.rag.config import RagBenchConfig
from benchmarks.rag.data_types import RagChunk, RagDatasetSpec, RagDocument, RagQuery
from benchmarks.rag.datasets import get_dataset
from benchmarks.rag.tokenizer import load_rag_tokenizer


@dataclass
class RagCorpus:
    """A dataset's documents and queries, the documents chunked by the answer model's tokenizer."""

    spec: RagDatasetSpec
    docs: list[RagDocument]
    queries: list[RagQuery]
    tokenizer: Any
    chunks: list[RagChunk]

    @cached_property
    def fingerprint(self) -> str:
        return corpus_fingerprint(self.docs)

    @cached_property
    def docs_by_id(self) -> dict[str, RagDocument]:
        return {doc.doc_id: doc for doc in self.docs}

    @cached_property
    def chunks_by_id(self) -> dict[str, RagChunk]:
        return {chunk.chunk_id: chunk for chunk in self.chunks}

    @cached_property
    def chunk_text_by_id(self) -> dict[str, str]:
        return {chunk.chunk_id: chunk.text for chunk in self.chunks}

    @property
    def total_tokens(self) -> int:
        return sum(chunk.token_count for chunk in self.chunks)


def load_rag_corpus(
    config: RagBenchConfig, *, allow_transformers_fallback: bool = False
) -> RagCorpus:
    """Load and chunk the configured dataset.

    The transformers tokenizer fallback is only for the estimate stage on
    hosts without vLLM; every other stage needs the engine's own tokenizer so
    chunk token ids match what the engine sees.
    """
    spec = get_dataset(config.dataset_name)
    docs, queries = spec.load(config.data_dir)
    tokenizer = load_rag_tokenizer(
        config.model, allow_transformers_fallback=allow_transformers_fallback
    )
    chunks = chunk_corpus(docs, tokenizer=tokenizer, chunk_size=config.chunk_size)
    return RagCorpus(spec=spec, docs=docs, queries=queries, tokenizer=tokenizer, chunks=chunks)
