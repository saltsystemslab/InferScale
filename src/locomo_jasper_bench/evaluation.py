from __future__ import annotations

import time
from typing import Any

from .clients_factory import RuntimeClients
from .config import BenchmarkConfig
from .data import ConversationSample, QuestionAnswer
from .judging import judge_qa, skipped_judge_payload
from .modes import result_mode
from .vector_types import RetrievalMetrics, SearchHit, SearchMetrics


class QuestionEvaluator:
    def __init__(self, config: BenchmarkConfig, clients: RuntimeClients) -> None:
        self.config = config
        self.clients = clients

    def answer_from_hits(
        self,
        sample: ConversationSample,
        qa: QuestionAnswer,
        hits: list[SearchHit],
        *,
        retrieval_metrics: RetrievalMetrics | None = None,
        ttft_started_at: float | None = None,
        query_started_at: float | None = None,
    ) -> dict[str, Any]:
        kv_answer = getattr(self.clients.answer_client, "answer_with_retrieved_memory", None)
        if not callable(kv_answer):
            raise RuntimeError(f"{self.config.answer_backend} answer backend cannot answer with retrieved memory.")
        answer = kv_answer(
            sample=sample,
            qa=qa,
            hits=hits,
            max_tokens=self.config.max_answer_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            ttft_started_at=ttft_started_at,
            query_started_at=query_started_at,
        )
        return self.record_answer(
            sample,
            qa,
            hits,
            answer,
            retrieval_metrics=retrieval_metrics,
        )

    def record_answer(
        self,
        sample: ConversationSample,
        qa: QuestionAnswer,
        hits: list[SearchHit],
        answer: Any,
        *,
        retrieval_metrics: RetrievalMetrics | None = None,
    ) -> dict[str, Any]:
        if self.config.skip_judge:
            judge_payload = skipped_judge_payload()
        else:
            if self.clients.judge_client is None:
                raise RuntimeError("Judge client is not configured. Use --skip-judge to write unjudged predictions.")
            judge_payload = judge_qa(self.config, self.clients.judge_client, qa, answer.content)

        metrics: dict[str, Any] = {
            "time_to_first_token_ms": answer.ttft_ms,
            **getattr(answer, "metrics", {}),
        }
        if retrieval_metrics is not None:
            metrics.update(
                {
                    "query_embedding_time_ms": retrieval_metrics.embedding_time_ms,
                    "vector_db_query_time_ms": retrieval_metrics.search_time_ms,
                    "query_retrieval_time_ms": retrieval_metrics.total_time_ms,
                }
            )
            answer_ttft_ms = metrics.get("answer_time_to_first_token_ms")
            if answer_ttft_ms is None:
                answer_ttft_ms = answer.ttft_ms
            if answer_ttft_ms is not None:
                query_to_first_token_ms = retrieval_metrics.total_time_ms + float(answer_ttft_ms)
                metrics["query_to_first_token_ms"] = query_to_first_token_ms
        return {
            "run_id": self.config.run_id,
            "mode": result_mode(self.config),
            "sample_id": sample.sample_id,
            "question_id": qa.question_id,
            "category": qa.category,
            "question": qa.question,
            "gold_answer": qa.answer,
            "predicted_answer": answer.content,
            "evidence": qa.evidence,
            "retrieved_memories": [
                {
                    "id": hit.id,
                    "rank": hit.rank,
                    "score": hit.score,
                    "distance": hit.distance,
                    "memory": hit.payload.get("memory") or hit.payload.get("text") or "",
                    "metadata": hit.payload.get("metadata", {}),
                }
                for hit in hits
            ],
            "judge": judge_payload,
            "metrics": metrics,
        }

    def retrieve_mem0_memory(self, memory: Any, query: str) -> tuple[list[SearchHit], RetrievalMetrics]:
        from .retrieval.memory_builder import embed_mem0_query

        vector_store = getattr(memory, "vector_store", None)
        search = getattr(vector_store, "search", None)
        if not callable(search):
            raise RuntimeError("Mem0 memory has no searchable vector_store.")

        started = time.perf_counter()
        embedding_started = time.perf_counter()
        query_embedding = embed_mem0_query(memory, query)
        embedding_time_ms = (time.perf_counter() - embedding_started) * 1000
        hits = search(query=query, vectors=query_embedding, top_k=self.config.top_k)
        total_time_ms = (time.perf_counter() - started) * 1000
        store_metrics = self._mem0_store_search_metrics(memory)
        return hits, RetrievalMetrics(
            embedding_time_ms=embedding_time_ms,
            search_time_ms=store_metrics.search_time_ms,
            total_time_ms=total_time_ms,
        )

    def search_mem0_memory(self, memory: Any, query: str) -> list[SearchHit]:
        hits, _metrics = self.retrieve_mem0_memory(memory, query)
        return hits

    def _search_mem0_memory(self, memory: Any, query: str) -> list[SearchHit]:
        return self.search_mem0_memory(memory, query)

    def _mem0_store_search_metrics(self, memory: Any) -> SearchMetrics:
        vector_store = getattr(memory, "vector_store", None)
        metrics = getattr(vector_store, "last_search_metrics", None)
        if isinstance(metrics, SearchMetrics):
            return metrics
        return SearchMetrics(
            search_time_ms=float(getattr(metrics, "search_time_ms", 0.0) or 0.0),
        )
