"""No-GPU smoke test wiring loader -> chunker -> fake retrieval -> records -> summary."""

from __future__ import annotations

import string
from typing import Any

import numpy as np

from benchmarks.common.clients import ChatResult
from benchmarks.common.files import write_csv
from benchmarks.common.files import JsonlWriter
from benchmarks.common.files import read_jsonl
from benchmarks.common.vector_types import RetrievalMetrics, SearchHit
from benchmarks.rag.chunking import chunk_corpus
from benchmarks.rag.config import RagBenchConfig
from benchmarks.rag.data_types import RagChunk
from benchmarks.rag.datasets.multihop_rag import load_multihop_rag
from benchmarks.rag.evaluation import build_query_record, final_answer_text
from benchmarks.common.judge import skipped_judge_payload
from benchmarks.rag.judging import judge_metadata
from benchmarks.rag.metrics import retrieval_metrics_for_query
from benchmarks.rag.results import (
    QUERY_METRICS_COLUMNS,
    build_query_metric_rows,
    summarize_rag_records,
)
from rag_fixtures import CharTokenizer, write_multihop_files


def _letter_vector(text: str) -> np.ndarray:
    counts = np.zeros(len(string.ascii_lowercase), dtype=np.float32)
    for character in text.lower():
        index = ord(character) - ord("a")
        if 0 <= index < len(counts):
            counts[index] += 1.0
    norm = float(np.linalg.norm(counts))
    return counts / norm if norm else counts


class FakeVectorStore:
    def __init__(self, chunks: list[RagChunk]) -> None:
        self._chunks = chunks
        self._matrix = np.vstack([_letter_vector(chunk.text) for chunk in chunks])

    def search(self, question: str, top_k: int) -> list[SearchHit]:
        scores = self._matrix @ _letter_vector(question)
        order = np.argsort(-scores, kind="stable")[:top_k]
        hits: list[SearchHit] = []
        for rank, ordinal in enumerate(order, start=1):
            chunk = self._chunks[int(ordinal)]
            hits.append(
                SearchHit(
                    id=chunk.chunk_id,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "chunk_index": chunk.chunk_index,
                        "token_count": chunk.token_count,
                    },
                    score=float(scores[int(ordinal)]),
                    distance=float(-scores[int(ordinal)]),
                    rank=rank,
                )
            )
        return hits


def test_pipeline_smoke_end_to_end(tmp_path) -> None:
    data_dir = write_multihop_files(tmp_path / "data")
    docs, queries = load_multihop_rag(data_dir)
    docs_by_id = {doc.doc_id: doc for doc in docs}
    tokenizer = CharTokenizer()
    chunks = chunk_corpus(docs, tokenizer=tokenizer, chunk_size=64)
    chunk_text_by_id = {chunk.chunk_id: chunk.text for chunk in chunks}
    store = FakeVectorStore(chunks)
    config = RagBenchConfig(
        results_dir=tmp_path / "results",
        run_id="smoke",
        answer_backend="vllm-prefix",
        skip_judge=True,
        top_k=4,
    )

    records: list[dict[str, Any]] = []
    for query in queries:
        hits = store.search(query.question, config.top_k)
        answer = ChatResult(
            content=f"ANSWER: {query.gold_answer}",
            ttft_ms=12.5,
            metrics={"kv_memory_tokens": 100, "answer_generate_time_ms": 3.0},
        )
        answer.content = final_answer_text(answer.content)
        records.append(
            build_query_record(
                config,
                query,
                hits,
                answer,
                retrieval_metrics=RetrievalMetrics(
                    embedding_time_ms=1.0,
                    search_time_ms=0.5,
                    total_time_ms=1.5,
                    vector_backend="fake",
                ),
                retrieval_quality=retrieval_metrics_for_query(query, hits, chunk_text_by_id),
                judge_payload=skipped_judge_payload(judge_metadata(config)),
                docs_by_id=docs_by_id,
            )
        )

    predictions_path = tmp_path / "results" / "smoke" / "predictions.jsonl"
    with JsonlWriter(predictions_path) as writer:
        for record in records:
            writer.write(record)
    round_tripped = read_jsonl(predictions_path)
    assert round_tripped == records

    summary = summarize_rag_records(
        records,
        run_id=config.run_id,
        mode=config.result_mode(),
        config=config.to_jsonable(),
        system_metadata={},
        setup_metrics={"chunk_count": len(chunks)},
    )
    metrics = summary["metrics"]
    assert summary["question_count"] == 4
    assert summary["judged_count"] == 0
    assert metrics["accuracy"] is None
    assert metrics["exact_match"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["abstention_accuracy"] == 1.0
    assert metrics["false_abstention_rate"] == 0.0
    assert metrics["retrieval"]["query_count"] == 3
    assert metrics["question_type_counts"] == {
        "comparison_query": 1,
        "inference_query": 1,
        "null_query": 1,
        "temporal_query": 1,
    }
    assert 0.0 <= metrics["retrieval"]["evidence_recall_at_k"] <= 1.0
    assert metrics["kv_memory_tokens"]["count"] == 4
    assert summary["setup"]["chunk_count"] == len(chunks)

    null_record = next(record for record in records if record["question_type"] == "null_query")
    assert null_record["retrieval"] is None
    assert null_record["answer_metrics"]["predicted_insufficient"] is True
    titled = records[0]["retrieved_chunks"][0]
    assert titled["title"] == docs_by_id[titled["doc_id"]].title

    rows = build_query_metric_rows(records)
    assert len(rows) == 4
    write_csv(tmp_path / "query_metrics.csv", rows, QUERY_METRICS_COLUMNS)
    header = (tmp_path / "query_metrics.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == ",".join(QUERY_METRICS_COLUMNS)
