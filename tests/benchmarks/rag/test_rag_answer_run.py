from __future__ import annotations

import json
from typing import Any

import pytest

from benchmarks.common.files import read_jsonl
from benchmarks.rag import answer_run, retrieval
from benchmarks.rag.answer_run import run_answer
from benchmarks.rag.corpus import load_rag_corpus
from benchmarks.rag.preembed import preembed_rag_embeddings
from benchmarks.rag.results import QUERY_METRICS_COLUMNS

SYSTEM_METADATA = {"platform": "test host"}


@pytest.fixture
def prompt_injection_run(
    multihop_config, embedding_api, jasper_indexes, engine, ticking_clock, monkeypatch
):
    monkeypatch.setattr(answer_run, "collect_system_metadata", lambda: SYSTEM_METADATA)
    ticking_clock(answer_run)
    ticking_clock(retrieval)
    preembed_rag_embeddings(multihop_config(run_id="preembed"))

    def configure(**settings: Any):
        return multihop_config(answer_backend="prompt-injection", **settings)

    return configure


def saved(config, filename: str) -> Any:
    return json.loads((config.run_dir / filename).read_text(encoding="utf-8"))


def test_run_saves_config_system_predictions_summary_setup_and_the_query_report(
    prompt_injection_run,
):
    config = prompt_injection_run()

    summary = run_answer(config)

    assert sorted(path.name for path in config.run_dir.iterdir()) == [
        "config.json",
        "predictions.jsonl",
        "query_metrics.csv",
        "setup.json",
        "summary.json",
        "system.json",
    ]
    assert saved(config, "config.json") == config.to_jsonable()
    assert saved(config, "system.json") == SYSTEM_METADATA
    assert saved(config, "summary.json") == summary
    assert (summary["run_id"], summary["mode"]) == ("run", "rag-prompt-injection")
    assert (summary["question_count"], summary["judged_count"]) == (4, 0)
    assert (summary["config"], summary["system"]) == (config.to_jsonable(), SYSTEM_METADATA)
    predictions = read_jsonl(config.run_dir / "predictions.jsonl")
    assert [record["query_id"] for record in predictions] == ["q0000", "q0001", "q0002", "q0003"]
    report = (config.run_dir / "query_metrics.csv").read_text(encoding="utf-8").splitlines()
    assert report[0] == ",".join(QUERY_METRICS_COLUMNS)
    assert [row.split(",")[0] for row in report[1:]] == ["q0000", "q0001", "q0002", "q0003"]


def test_each_prediction_records_the_answer_its_evidence_and_the_query_latency(
    prompt_injection_run, engine
):
    config = prompt_injection_run()
    corpus = load_rag_corpus(config)

    run_answer(config)

    first, *_, last = read_jsonl(config.run_dir / "predictions.jsonl")
    assert list(first) == [
        "run_id",
        "mode",
        "dataset",
        "query_id",
        "question_type",
        "category",
        "question",
        "gold_answer",
        "gold_answers",
        "predicted_answer",
        "evidence",
        "retrieved_chunks",
        "answer_metrics",
        "retrieval",
        "judge",
        "metrics",
    ]
    query = corpus.queries[0]
    assert (first["question_type"], first["category"]) == ("inference_query", "inference_query")
    assert (first["question"], first["gold_answers"]) == (query.question, ["Alpha Corp"])
    assert first["predicted_answer"] == f"answer over {len(engine.prompts[0])} prompt tokens"
    assert [ref["fact"] for ref in first["evidence"]] == [ref.fact for ref in query.evidence]
    assert [chunk["rank"] for chunk in first["retrieved_chunks"]] == [1, 2, 3]
    assert first["retrieved_chunks"][0]["title"] in {doc.title for doc in corpus.docs}
    assert first["retrieval"]["retrieved_chunk_count"] == 3
    assert first["judge"] == {
        "correct": None,
        "reason": "skipped",
        "raw": "",
        "status": "skipped",
        "provider": "none",
        "model": config.judge_model,
    }
    assert first["metrics"]["time_to_first_token_ms"] == 12.0
    assert first["metrics"]["resolved_vector_backend"] == "fake"
    assert first["metrics"]["query_to_answer_ms"] == (100.25 - 2.0) * 1000
    assert last["metrics"]["query_to_answer_ms"] == (100.25 - 5.0) * 1000
    assert (last["question_type"], last["retrieval"]) == ("null_query", None)


def test_answer_client_validates_its_caches_before_anything_is_written_for_the_run(
    prompt_injection_run, multihop_config, jasper_indexes
):
    config = multihop_config(answer_backend="kv-injection")

    with pytest.raises(RuntimeError, match="24 of 24 RAG KV chunk files are missing"):
        run_answer(config)

    assert not config.run_dir.exists()
    assert jasper_indexes == []
