"""MultiHop-RAG dataset loader (Tang & Yang, 2024).

Corpus: 609 news articles. Queries: 2,556 with types inference_query,
comparison_query, temporal_query, and null_query. Null queries have an empty
evidence_list and the gold answer "Insufficient information."; every non-null
evidence item names a corpus document (title and url are unique corpus keys).
"""

from __future__ import annotations

import json
from pathlib import Path

from ..data_types import EvidenceRef, RagDocument, RagQuery, doc_id_for_url

CORPUS_FILENAME = "corpus.json"
QUERIES_FILENAME = "MultiHopRAG.json"
CORPUS_URL = "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/corpus.json"
QUERIES_URL = "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/MultiHopRAG.json"

QUESTION_TYPES = frozenset(
    {"inference_query", "comparison_query", "temporal_query", "null_query"}
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
    records = _load_json_list(path, url=CORPUS_URL)
    docs: list[RagDocument] = []
    seen_urls: dict[str, int] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: corpus record {index} is not a JSON object.")
        for field in _REQUIRED_DOC_FIELDS:
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{path}: corpus record {index} is missing required field {field!r}."
                )
        url = record["url"].strip()
        previous = seen_urls.get(url)
        if previous is not None:
            raise ValueError(
                f"{path}: corpus records {previous} and {index} share the url {url!r}; "
                "urls must be unique because they identify documents."
            )
        seen_urls[url] = index
        author = record.get("author")
        docs.append(
            RagDocument(
                doc_id=doc_id_for_url(url),
                title=record["title"].strip(),
                source=record["source"].strip(),
                published_at=record["published_at"].strip(),
                category=record["category"].strip(),
                url=url,
                body=record["body"],
                author=str(author) if isinstance(author, str) and author.strip() else None,
            )
        )
    if not docs:
        raise ValueError(f"{path}: corpus contains no documents.")
    return docs


def _load_queries(path: Path, docs: list[RagDocument]) -> list[RagQuery]:
    records = _load_json_list(path, url=QUERIES_URL)
    docs_by_url = {doc.url: doc for doc in docs}
    docs_by_title = {doc.title: doc for doc in docs}
    queries: list[RagQuery] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: query record {index} is not a JSON object.")
        for field in _REQUIRED_QUERY_FIELDS:
            if field not in record:
                raise ValueError(
                    f"{path}: query record {index} is missing required field {field!r}."
                )
        question_type = str(record["question_type"]).strip()
        if question_type not in QUESTION_TYPES:
            raise ValueError(
                f"{path}: query record {index} has unknown question_type "
                f"{question_type!r}; expected one of {sorted(QUESTION_TYPES)}."
            )
        raw_evidence = record["evidence_list"]
        if not isinstance(raw_evidence, list):
            raise ValueError(f"{path}: query record {index} evidence_list is not a list.")
        evidence = tuple(
            _resolve_evidence(path, index, item, docs_by_url, docs_by_title)
            for item in raw_evidence
        )
        is_null = question_type == "null_query"
        if is_null and evidence:
            raise ValueError(
                f"{path}: query record {index} is a null_query but has evidence."
            )
        if not is_null and not evidence:
            raise ValueError(
                f"{path}: query record {index} ({question_type}) has no evidence."
            )
        queries.append(
            RagQuery(
                query_id=f"q{index:04d}",
                question=str(record["query"]).strip(),
                gold_answer=str(record["answer"]).strip(),
                question_type=question_type,
                evidence=evidence,
            )
        )
    if not queries:
        raise ValueError(f"{path}: query file contains no queries.")
    return queries


def _resolve_evidence(
    path: Path,
    query_index: int,
    item: object,
    docs_by_url: dict[str, RagDocument],
    docs_by_title: dict[str, RagDocument],
) -> EvidenceRef:
    if not isinstance(item, dict):
        raise ValueError(f"{path}: query record {query_index} evidence item is not a JSON object.")
    url = str(item.get("url") or "").strip()
    title = str(item.get("title") or "").strip()
    fact = str(item.get("fact") or "").strip()
    if not fact:
        raise ValueError(
            f"{path}: query record {query_index} evidence item has no fact text."
        )
    doc = docs_by_url.get(url) if url else None
    if doc is None and title:
        doc = docs_by_title.get(title)
    if doc is None:
        raise ValueError(
            f"{path}: query record {query_index} evidence does not resolve to any corpus "
            f"document (url={url!r} title={title!r})."
        )
    return EvidenceRef(doc_id=doc.doc_id, title=doc.title, url=doc.url, fact=fact)


def _load_json_list(path: Path, *, url: str) -> list[object]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Download it with scripts/rag/setup_data.sh "
            f"(source: {url})."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a top-level JSON list, got {type(data).__name__}.")
    return data


def _spec():
    from . import RagDatasetSpec

    return RagDatasetSpec(
        name="multihoprag",
        corpus_filename=CORPUS_FILENAME,
        queries_filename=QUERIES_FILENAME,
        download_urls={CORPUS_FILENAME: CORPUS_URL, QUERIES_FILENAME: QUERIES_URL},
        chunking="token-window",
        load=load_multihop_rag,
    )


MULTIHOP_RAG_SPEC = _spec()
