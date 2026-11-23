from __future__ import annotations

from benchmarks.common.clients import ChatResult
from benchmarks.common.vector_types import RetrievalMetrics, SearchHit
from benchmarks.rag.config import RagBenchConfig
from benchmarks.rag.data_types import EvidenceRef, RagDocument, RagQuery
from benchmarks.rag.evaluation import build_query_record

DOC = RagDocument(
    doc_id="dA",
    title="Alpha Corp stuns the energy market",
    source="The Example Times",
    published_at="2023-10-01",
    category="technology",
    url="https://example.com/alpha",
    body="body",
)
QUERY = RagQuery(
    query_id="q0007",
    question="Who announced it?",
    gold_answers=("Alpha Corp", "Alpha"),
    question_type="inference_query",
    evidence=(EvidenceRef(doc_id="dA", title=DOC.title, url=DOC.url, fact="Alpha announced it"),),
)
HITS = [
    SearchHit(
        id="dA:2", payload={"doc_id": "dA", "chunk_index": 2}, score=0.9, distance=0.1, rank=1
    ),
    SearchHit(
        id="dX:0", payload={"doc_id": "dX", "chunk_index": 0}, score=0.5, distance=0.5, rank=2
    ),
    SearchHit(id=7, payload={}, score=0.1, distance=0.9, rank=3),
]


def record_for(answer: ChatResult, retrieval_metrics: RetrievalMetrics | None = None) -> dict:
    return build_query_record(
        RagBenchConfig(run_id="run", answer_backend="prompt-injection"),
        QUERY,
        HITS,
        answer,
        retrieval_metrics=retrieval_metrics,
        retrieval_quality={"evidence_recall_at_k": 1.0},
        judge_payload={"correct": None, "status": "skipped"},
        docs_by_id={"dA": DOC},
    )


def test_category_mirrors_the_question_type_for_the_reports_shared_with_the_memory_benchmark():
    record = record_for(ChatResult(content="Alpha Corp"))

    assert record["category"] == record["question_type"] == "inference_query"
