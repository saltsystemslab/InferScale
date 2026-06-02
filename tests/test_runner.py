from __future__ import annotations

import json

from locomo_jasper_bench.clients import ChatResult
from locomo_jasper_bench.config import BenchmarkConfig
from locomo_jasper_bench.jasper_store import BuildMetrics, SearchMetrics
from locomo_jasper_bench.results import read_jsonl
from locomo_jasper_bench.runner import RuntimeClients, run_benchmark


class FakeAnswerClient:
    def __init__(self) -> None:
        self.calls = []

    def chat(self, messages, *, max_tokens, temperature, top_p):
        assert messages
        self.calls.append(messages)
        return ChatResult(content="Alice adopted Pixel.", latency_ms=12.0, prompt_tokens=10, completion_tokens=4)


class FakeJudgeClient:
    def chat(self, messages, *, max_tokens, temperature, top_p):
        assert temperature == 0.0
        return ChatResult(content='{"correct": true, "reason": "matches"}', latency_ms=5.0)


def _write_dataset(path):
    path.write_text(
        json.dumps(
            [
                {
                    "sample_id": "conv-1",
                    "conversation": {
                        "session_1": [
                            {"speaker": "Alice", "text": "I adopted a cat named Pixel."},
                            {"speaker": "Bob", "text": "Pixel likes the window."},
                        ]
                    },
                    "qa": [
                        {
                            "question_id": "q1",
                            "question": "Who adopted Pixel?",
                            "answer": "Alice",
                            "category": "single-hop",
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )


def test_runner_full_context_mode_with_mocks(tmp_path):
    dataset_path = tmp_path / "locomo.json"
    _write_dataset(dataset_path)
    answer_client = FakeAnswerClient()
    config = BenchmarkConfig(
        mode="baseline",
        dataset_path=dataset_path,
        results_dir=tmp_path / "results",
        run_id="test-run",
        context_mode="full",
    )
    clients = RuntimeClients(
        answer_client=answer_client,
        judge_client=FakeJudgeClient(),
    )

    summary = run_benchmark(config, clients)

    assert summary["accuracy"] == 1.0
    rows = read_jsonl(tmp_path / "results" / "test-run" / "predictions.jsonl")
    assert len(rows) == 1
    assert rows[0]["judge"]["correct"] is True
    assert rows[0]["retrieved_memories"] == []
    assert rows[0]["memory"]["backend"] == "none"
    assert rows[0]["index"]["backend"] == "none"
    assert rows[0]["latency_ms"]["answer_generation_ms"] == 12.0
    assert rows[0]["vllm"]["answer"]["latency_ms"] == 12.0
    prompt = answer_client.calls[0][1]["content"]
    assert "Full conversation transcript:" in prompt
    assert "Alice: I adopted a cat named Pixel." in prompt
    assert "Retrieved memory context:" not in prompt


class FakeMem0VectorStore:
    def __init__(self) -> None:
        self.finalized = False
        self.closed = False
        self.last_search_metrics = SearchMetrics(
            backend="jasper",
            search_time_ms=2.5,
            indexed_vector_count=2,
            embedding_dim=3,
        )

    def finalize(self):
        self.finalized = True
        return BuildMetrics(
            backend="jasper",
            graph_build_time_ms=7.0,
            indexed_vector_count=2,
            embedding_dim=3,
            graph_path="graph",
        )

    def close(self):
        self.closed = True


class FakeMem0Memory:
    def __init__(self) -> None:
        self.add_calls = []
        self.search_calls = []
        self.vector_store = FakeMem0VectorStore()

    def add(self, messages, *, user_id, infer, metadata):
        self.add_calls.append(
            {
                "messages": messages,
                "user_id": user_id,
                "infer": infer,
                "metadata": metadata,
            }
        )
        return {"results": [{"id": metadata["turn_id"], "memory": messages[0]["content"]}]}

    def search(self, *, query, filters, top_k):
        self.search_calls.append({"query": query, "filters": filters, "top_k": top_k})
        return {
            "results": [
                {
                    "id": "mem-1",
                    "memory": "Alice: I adopted a cat named Pixel.",
                    "score": 0.91,
                    "metadata": {"turn_id": "conv-1:session_1:0"},
                }
            ]
        }


def test_runner_mem0_context_mode_with_mocked_mem0(tmp_path, monkeypatch):
    dataset_path = tmp_path / "locomo.json"
    _write_dataset(dataset_path)
    fake_memories: list[FakeMem0Memory] = []

    def fake_create_mem0_memory(**kwargs):
        assert kwargs["store_root"] == tmp_path / "results" / "test-run" / "mem0" / "conv-1"
        memory = FakeMem0Memory()
        fake_memories.append(memory)
        return memory

    monkeypatch.setattr("locomo_jasper_bench.runner.create_mem0_memory", fake_create_mem0_memory)
    answer_client = FakeAnswerClient()
    config = BenchmarkConfig(
        mode="baseline",
        dataset_path=dataset_path,
        results_dir=tmp_path / "results",
        run_id="test-run",
        context_mode="mem0",
        top_k=5,
    )
    clients = RuntimeClients(
        answer_client=answer_client,
        judge_client=FakeJudgeClient(),
    )

    summary = run_benchmark(config, clients)

    assert summary["accuracy"] == 1.0
    assert len(fake_memories) == 1
    memory = fake_memories[0]
    assert memory.vector_store.finalized is True
    assert memory.vector_store.closed is True
    assert len(memory.add_calls) == 2
    assert memory.add_calls[0]["infer"] is False
    assert memory.add_calls[0]["user_id"] == "conv-1"
    assert memory.add_calls[0]["messages"] == [{"role": "user", "content": "Alice: I adopted a cat named Pixel."}]
    assert memory.add_calls[0]["metadata"]["sample_id"] == "conv-1"
    assert memory.add_calls[0]["metadata"]["turn_id"] == "conv-1:session_1:0"
    assert memory.add_calls[0]["metadata"]["speaker"] == "Alice"
    assert memory.search_calls == [
        {"query": "Who adopted Pixel?", "filters": {"user_id": "conv-1"}, "top_k": 5}
    ]

    rows = read_jsonl(tmp_path / "results" / "test-run" / "predictions.jsonl")
    assert len(rows) == 1
    assert rows[0]["judge"]["correct"] is True
    assert rows[0]["retrieved_memories"]
    assert rows[0]["retrieved_memories"][0]["memory"] == "Alice: I adopted a cat named Pixel."
    assert rows[0]["memory"]["backend"] == "mem0-jasper"
    assert rows[0]["memory"]["vector_search_ms"] == 2.5
    assert rows[0]["index"]["backend"] == "jasper"
    assert rows[0]["index"]["memory_add_count"] == 2
    assert rows[0]["index"]["infer"] is False
    assert rows[0]["latency_ms"]["answer_generation_ms"] == 12.0
    assert rows[0]["latency_ms"]["memory_search_ms"] >= 0.0
    prompt = answer_client.calls[0][1]["content"]
    assert "Retrieved memory context:" in prompt
    assert "Alice: I adopted a cat named Pixel." in prompt
