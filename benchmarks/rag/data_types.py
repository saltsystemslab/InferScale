from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

NULL_QUERY_TYPE = "null_query"
UNANSWERABLE_TYPE = "unanswerable"
ABSTENTION_QUESTION_TYPES = frozenset({NULL_QUERY_TYPE, UNANSWERABLE_TYPE})


def doc_id_for_url(url: str) -> str:
    stripped_url = url.strip()
    if not stripped_url:
        raise ValueError("Document URL must be non-empty to derive a doc id.")
    return "d" + hashlib.sha256(stripped_url.encode("utf-8")).hexdigest()[:12]


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
    doc_id: str
    title: str
    url: str
    fact: str


@dataclass(slots=True, frozen=True)
class RagQuery:
    query_id: str
    question: str
    gold_answers: tuple[str, ...]
    question_type: str
    evidence: tuple[EvidenceRef, ...]

    def __post_init__(self) -> None:
        if not self.gold_answers:
            raise ValueError(f"RagQuery {self.query_id} requires at least one gold answer.")

    @property
    def gold_answer(self) -> str:
        return self.gold_answers[0]


@dataclass(slots=True, frozen=True)
class RagPromptProfile:
    system_prompt: str
    answer_instruction: str


@dataclass(slots=True)
class RagChunk:
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


@dataclass(slots=True, frozen=True)
class RagDatasetSpec:
    name: str
    corpus_filename: str
    queries_filename: str
    download_urls: dict[str, str]
    prompt_profile: RagPromptProfile
    load: Callable[[Path], tuple[list[RagDocument], list[RagQuery]]]
