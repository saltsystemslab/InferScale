from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.rag.data_types import (
    NULL_QUERY_TYPE,
    EvidenceRef,
    RagDatasetSpec,
    RagDocument,
    RagPromptProfile,
    RagQuery,
    doc_id_for_url,
)

INSUFFICIENT_ANSWER_TEXT = "Insufficient information"
MULTIHOP_PROMPT_PROFILE = RagPromptProfile(
    system_prompt=(
        "You are a helpful assistant that answers questions using retrieved excerpts "
        "from a corpus of news articles. "
        "The following are excerpts from news articles:\n\n"
    ),
    answer_instruction=(
        "Answer the question using only the news article excerpts above.\n"
        "Answer with a short phrase. If the excerpts do not contain the "
        f"information needed, answer exactly: {INSUFFICIENT_ANSWER_TEXT}\n"
    ),
)

CORPUS_FILENAME = "corpus.json"
QUERIES_FILENAME = "MultiHopRAG.json"
CORPUS_URL = "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/corpus.json"
QUERIES_URL = "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/MultiHopRAG.json"

MULTIHOP_QUESTION_TYPES = frozenset(
    {"inference_query", "comparison_query", "temporal_query", NULL_QUERY_TYPE}
)

_REQUIRED_DOC_FIELDS = ("title", "body", "category", "published_at", "source", "url")
_REQUIRED_QUERY_FIELDS = ("query", "answer", "question_type", "evidence_list")


def load_multihop_rag(data_dir: Path) -> tuple[list[RagDocument], list[RagQuery]]:
    corpus_path = Path(data_dir) / CORPUS_FILENAME
    queries_path = Path(data_dir) / QUERIES_FILENAME
    docs = _load_corpus(corpus_path)
    queries = _load_queries(queries_path, docs)
    return docs, queries


def _load_corpus(path: Path) -> list[RagDocument]:
    records = _load_json_list(path, source_url=CORPUS_URL)
    docs: list[RagDocument] = []
    record_index_by_url: dict[str, int] = {}
    for index, record in enumerate(records):
        _require_document_fields(path, index, record)
        url = record["url"].strip()
        first_index = record_index_by_url.get(url)
        if first_index is not None:
            raise ValueError(
                f"{path}: corpus records {first_index} and {index} share the url {url!r}; "
                "urls must be unique because they identify documents."
            )
        record_index_by_url[url] = index
        docs.append(_document(record, url))
    if not docs:
        raise ValueError(f"{path}: corpus contains no documents.")
    return docs


def _require_document_fields(path: Path, index: int, record: Any) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"{path}: corpus record {index} is not a JSON object.")
    for field in _REQUIRED_DOC_FIELDS:
        text = record.get(field)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}: corpus record {index} is missing required field {field!r}.")


def _document(record: dict[str, Any], url: str) -> RagDocument:
    author = record.get("author")
    return RagDocument(
        doc_id=doc_id_for_url(url),
        title=record["title"].strip(),
        source=record["source"].strip(),
        published_at=record["published_at"].strip(),
        category=record["category"].strip(),
        url=url,
        body=record["body"],
        author=str(author) if isinstance(author, str) and author.strip() else None,
    )


def _load_queries(path: Path, docs: list[RagDocument]) -> list[RagQuery]:
    records = _load_json_list(path, source_url=QUERIES_URL)
    docs_by_url = {doc.url: doc for doc in docs}
    docs_by_title = {doc.title: doc for doc in docs}
    queries = [
        _query(path, index, record, docs_by_url, docs_by_title)
        for index, record in enumerate(records)
    ]
    if not queries:
        raise ValueError(f"{path}: query file contains no queries.")
    return queries


def _query(
    path: Path,
    index: int,
    record: Any,
    docs_by_url: dict[str, RagDocument],
    docs_by_title: dict[str, RagDocument],
) -> RagQuery:
    _require_query_fields(path, index, record)
    question_type = _question_type(path, index, record)
    evidence = _evidence(path, index, record, docs_by_url, docs_by_title)
    _require_evidence_unless_null(path, index, question_type, evidence)
    return RagQuery(
        query_id=f"q{index:04d}",
        question=str(record["query"]).strip(),
        gold_answers=(str(record["answer"]).strip(),),
        question_type=question_type,
        evidence=evidence,
    )


def _require_query_fields(path: Path, index: int, record: Any) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"{path}: query record {index} is not a JSON object.")
    for field in _REQUIRED_QUERY_FIELDS:
        if field not in record:
            raise ValueError(f"{path}: query record {index} is missing required field {field!r}.")


def _question_type(path: Path, index: int, record: dict[str, Any]) -> str:
    question_type = str(record["question_type"]).strip()
    if question_type not in MULTIHOP_QUESTION_TYPES:
        raise ValueError(
            f"{path}: query record {index} has unknown question_type "
            f"{question_type!r}; expected one of {sorted(MULTIHOP_QUESTION_TYPES)}."
        )
    return question_type


def _evidence(
    path: Path,
    index: int,
    record: dict[str, Any],
    docs_by_url: dict[str, RagDocument],
    docs_by_title: dict[str, RagDocument],
) -> tuple[EvidenceRef, ...]:
    evidence_items = record["evidence_list"]
    if not isinstance(evidence_items, list):
        raise ValueError(f"{path}: query record {index} evidence_list is not a list.")
    return tuple(
        _resolve_evidence(path, index, evidence_item, docs_by_url, docs_by_title)
        for evidence_item in evidence_items
    )


def _require_evidence_unless_null(
    path: Path, index: int, question_type: str, evidence: tuple[EvidenceRef, ...]
) -> None:
    is_null_query = question_type == NULL_QUERY_TYPE
    if is_null_query and evidence:
        raise ValueError(f"{path}: query record {index} is a null_query but has evidence.")
    if not is_null_query and not evidence:
        raise ValueError(f"{path}: query record {index} ({question_type}) has no evidence.")


def _resolve_evidence(
    path: Path,
    query_index: int,
    evidence_item: object,
    docs_by_url: dict[str, RagDocument],
    docs_by_title: dict[str, RagDocument],
) -> EvidenceRef:
    if not isinstance(evidence_item, dict):
        raise ValueError(f"{path}: query record {query_index} evidence item is not a JSON object.")
    url = str(evidence_item.get("url") or "").strip()
    title = str(evidence_item.get("title") or "").strip()
    fact = str(evidence_item.get("fact") or "").strip()
    if not fact:
        raise ValueError(f"{path}: query record {query_index} evidence item has no fact text.")
    doc = docs_by_url.get(url) if url else None
    if doc is None and title:
        doc = docs_by_title.get(title)
    if doc is None:
        raise ValueError(
            f"{path}: query record {query_index} evidence does not resolve to any corpus "
            f"document (url={url!r} title={title!r})."
        )
    return EvidenceRef(doc_id=doc.doc_id, title=doc.title, url=doc.url, fact=fact)


def _load_json_list(path: Path, *, source_url: str) -> list[object]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run bash scripts/rag/setup_data.sh to download it "
            f"(source: {source_url})."
        )
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, list):
        raise ValueError(f"{path}: expected a top-level JSON list, got {type(parsed).__name__}.")
    return parsed


MULTIHOP_RAG_SPEC = RagDatasetSpec(
    name="multihoprag",
    corpus_filename=CORPUS_FILENAME,
    queries_filename=QUERIES_FILENAME,
    download_urls={CORPUS_FILENAME: CORPUS_URL, QUERIES_FILENAME: QUERIES_URL},
    prompt_profile=MULTIHOP_PROMPT_PROFILE,
    load=load_multihop_rag,
)
