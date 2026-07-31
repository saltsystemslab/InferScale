from __future__ import annotations

import hashlib
from dataclasses import dataclass


def doc_id_for_url(url: str) -> str:
    """Stable document id derived from the (unique) source URL.

    Prefixed so ids never look like bare hashes in logs, and independent of
    corpus file ordering so cache fingerprints survive reordering.
    """
    normalized = url.strip()
    if not normalized:
        raise ValueError("Document URL must be non-empty to derive a doc id.")
    return "d" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


@dataclass(slots=True, frozen=True)
class RagDocument:
    doc_id: str
    title: str
    source: str
    published_at: str
    category: str
    url: str
    body: str
    author: str | None = None


@dataclass(slots=True, frozen=True)
class EvidenceRef:
    """One gold evidence item, resolved to a corpus document at load time."""

    doc_id: str
    title: str
    url: str
    fact: str


@dataclass(slots=True, frozen=True)
class RagQuery:
    query_id: str
    question: str
    gold_answer: str
    question_type: str
    evidence: tuple[EvidenceRef, ...]

    @property
    def is_null(self) -> bool:
        return self.question_type == "null_query"


@dataclass(slots=True)
class RagChunk:
    """One fixed-token-size slice of a rendered document.

    token_ids are the single source of truth for both answer modes; text is
    the decode of token_ids and is used only for embedding and metrics.
    """

    chunk_id: str
    doc_id: str
    chunk_index: int
    token_ids: list[int]
    text: str

    @property
    def token_count(self) -> int:
        return len(self.token_ids)


def chunk_id_for(doc_id: str, chunk_index: int) -> str:
    return f"{doc_id}:{chunk_index}"
