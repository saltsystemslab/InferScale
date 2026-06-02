from __future__ import annotations

import json

from locomo_jasper_bench.clients import ChatResult, HashEmbeddingClient
from locomo_jasper_bench.config import BenchmarkConfig
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


def test_runner_retrieval_context_mode_with_mocks(tmp_path):
    dataset_path = tmp_path / "locomo.json"
    _write_dataset(dataset_path)
    config = BenchmarkConfig(
        mode="baseline",
        dataset_path=dataset_path,
        results_dir=tmp_path / "results",
        run_id="test-run",
        context_mode="retrieval",
        embedding_provider="hash",
        vector_backend="numpy",
        hash_embedding_dim=64,
    )
    clients = RuntimeClients(
        answer_client=FakeAnswerClient(),
        judge_client=FakeJudgeClient(),
        embedding_client=HashEmbeddingClient(dim=64),
    )

    summary = run_benchmark(config, clients)

    assert summary["accuracy"] == 1.0
    rows = read_jsonl(tmp_path / "results" / "test-run" / "predictions.jsonl")
    assert len(rows) == 1
    assert rows[0]["judge"]["correct"] is True
    assert rows[0]["retrieved_memories"]
