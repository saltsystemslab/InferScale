from __future__ import annotations

from typing import Any

from .clients_factory import RuntimeClients
from .config import BenchmarkConfig
from .data import ConversationSample, QuestionAnswer
from .judging import judge_qa, skipped_judge_payload
from .modes import result_mode
from .vector_types import SearchHit


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
        ttft_started_at: float | None = None,
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
        )
        return self.record_answer(sample, qa, hits, answer)

    def record_answer(
        self,
        sample: ConversationSample,
        qa: QuestionAnswer,
        hits: list[SearchHit],
        answer: Any,
    ) -> dict[str, Any]:
        if self.config.skip_judge:
            judge_payload = skipped_judge_payload()
        else:
            if self.clients.judge_client is None:
                raise RuntimeError("Judge client is not configured. Use --skip-judge to write unjudged predictions.")
            judge_payload = judge_qa(self.config, self.clients.judge_client, qa, answer.content)

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
            "metrics": {
                "time_to_first_token_ms": answer.ttft_ms,
                **getattr(answer, "metrics", {}),
            },
        }

    def search_mem0_memory(self, memory: Any, query: str) -> list[SearchHit]:
        from .retrieval.memory_builder import embed_mem0_query

        vector_store = getattr(memory, "vector_store", None)
        search = getattr(vector_store, "search", None)
        if not callable(search):
            raise RuntimeError("Mem0 memory has no searchable vector_store.")

        query_embedding = embed_mem0_query(memory, query)
        return search(query=query, vectors=query_embedding, top_k=self.config.top_k)

    def _search_mem0_memory(self, memory: Any, query: str) -> list[SearchHit]:
        return self.search_mem0_memory(memory, query)
