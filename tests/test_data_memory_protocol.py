from __future__ import annotations

import json
from pathlib import Path

from locomo_jasper_bench.config import BenchmarkConfig
from locomo_jasper_bench.data import (
    ConversationSample,
    QuestionAnswer,
    Turn,
    format_turn_for_memory,
    load_locomo,
)
from locomo_jasper_bench.evaluation import _evidence_context
from locomo_jasper_bench.protocol import (
    ANSWER_PROMPT_PROTOCOL,
    JUDGE_PROMPT_PROTOCOL,
    MEM0AI_VERSION,
    MEMORY_BENCHMARKS_COMMIT,
    MEMORY_BENCHMARKS_REPOSITORY,
    MEMORY_INGESTION_PROTOCOL,
)


def test_memory_ingestion_text_matches_pinned_locomo_protocol() -> None:
    turn = Turn(
        sample_id="sample-1",
        session_id="session_1",
        session_index=1,
        turn_index=0,
        speaker="Alice",
        text="Look at this.",
        timestamp="1:56 pm on 8 May, 2023",
        image_caption="a green kayak",
        raw={"query": "What color is it?"},
    )

    assert format_turn_for_memory(turn) == (
        "Alice: Look at this. [Sharing image - query: What color is it?. "
        "The image shows: a green kayak]"
    )
    assert turn.timestamp not in format_turn_for_memory(turn)


def test_sessions_are_numbered_in_chronological_date_order(tmp_path: Path) -> None:
    dataset_path = tmp_path / "locomo.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "sample_id": "sample-1",
                    "conversation": {
                        "speaker_a": "Alice",
                        "speaker_b": "Bob",
                        "session_1": [{"speaker": "Alice", "text": "Later"}],
                        "session_1_date_time": "1:00 pm on 2 May, 2023",
                        "session_2": [{"speaker": "Bob", "text": "Earlier"}],
                        "session_2_date_time": "1:00 pm on 1 May, 2023",
                    },
                    "qa": [],
                }
            ]
        ),
        encoding="utf-8",
    )

    turns = load_locomo(dataset_path)[0].turns

    assert [(turn.session_id, turn.session_index) for turn in turns] == [
        ("session_2", 1),
        ("session_1", 2),
    ]


def test_run_config_records_fixed_benchmark_provenance() -> None:
    config = BenchmarkConfig().to_jsonable()

    assert config["memory_benchmarks_repository"] == MEMORY_BENCHMARKS_REPOSITORY
    assert config["memory_benchmarks_commit"] == MEMORY_BENCHMARKS_COMMIT
    assert config["mem0ai_version"] == MEM0AI_VERSION
    assert config["memory_ingestion_protocol"] == MEMORY_INGESTION_PROTOCOL
    assert config["answer_prompt_protocol"] == ANSWER_PROMPT_PROTOCOL
    assert config["judge_prompt_protocol"] == JUDGE_PROMPT_PROTOCOL


def test_evidence_context_matches_pinned_locomo_judge_format() -> None:
    turn = Turn(
        sample_id="sample-1",
        session_id="session_1",
        session_index=1,
        turn_index=0,
        speaker="Alice",
        text="We went to Lyon.",
        timestamp="1:56 pm on 8 May, 2023",
        raw={"dia_id": "D1:1"},
    )
    sample = ConversationSample("sample-1", [turn], [], {})
    qa = QuestionAnswer(
        sample_id="sample-1",
        question_id="q1",
        question="Where did Alice go?",
        answer="Lyon",
        category="1",
        evidence=["D1:1"],
    )

    assert _evidence_context(sample, qa) == (
        '[D1:1, said on 1:56 pm on 8 May, 2023] Alice: "We went to Lyon."'
    )
