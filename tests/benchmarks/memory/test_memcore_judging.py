from __future__ import annotations

from typing import Any

from memory_config import make_memory_config

from benchmarks.common.clients import ChatResult
from benchmarks.common.vector_types import RetrievalMetrics, SearchHit
from benchmarks.memory.data import ConversationSample, QuestionAnswer, Turn
from benchmarks.memory.evaluation import answer_question, build_prediction_record
from benchmarks.memory.facts import text_hash
from benchmarks.memory.judging import build_judge_messages, judge_record
from benchmarks.memory.runtime_clients import RuntimeClients

QUESTION = QuestionAnswer(
    "conv-1", "q3", "Where did Alice go?", "Lyon", "2", evidence=["D1:2", "D1:1"]
)
JUDGE_PROMPT = (
    "You are evaluating the correctness of an answer about a conversation. "
    "Compare the predicted answer to the reference answer. "
    "Answer with exactly one lowercase word: true or false.\n\n"
    "{evidence}"
    "Question: Where did Alice go?\n"
    "Reference answer: Lyon\n"
    "Predicted answer: She went to Lyon.\n\n"
    "Return true if the prediction conveys the same essential information as the "
    "reference answer, even if worded differently. "
    "Return false for contradictions, unsupported answers, or missing key facts. "
    "Output only true or false. Do not return JSON, punctuation, or an explanation."
)
EVIDENCE = '[D1:2] Bob: "Where to?"\n[D1:1, said on 1:56 pm on 8 May, 2023] Alice: "To Lyon."'


class Judge:
    def __init__(self, verdict: str = "true") -> None:
        self.verdict = verdict
        self.requests: list[tuple[list[dict[str, str]], dict[str, Any]]] = []

    def chat(self, messages: list[dict[str, str]], **options: Any) -> ChatResult:
        self.requests.append((messages, options))
        return ChatResult(self.verdict)


class AnswerClient:
    def __init__(self, answer: ChatResult) -> None:
        self.answer = answer
        self.requests: list[dict[str, Any]] = []

    def answer_with_retrieved_memory(self, **request: Any) -> ChatResult:
        self.requests.append(request)
        return self.answer


def conversation() -> ConversationSample:
    turns = [
        Turn(
            "conv-1",
            "session_1",
            1,
            0,
            "Alice",
            "To Lyon.",
            "1:56 pm on 8 May, 2023",
            None,
            {"dia_id": "D1:1"},
        ),
        Turn("conv-1", "session_1", 1, 1, "Bob", "Where to?", None, None, {"dia_id": "D1:2"}),
        Turn("conv-1", "session_1", 1, 2, "Bob", "No dialogue id.", "", None, {}),
        Turn("conv-1", "session_1", 1, 3, "Bob", "No record."),
    ]
    return ConversationSample("conv-1", turns, [QUESTION], {})


def hit(memory_id: str, **payload: Any) -> SearchHit:
    return SearchHit(id=memory_id, payload=payload, score=0.5, distance=0.25, rank=3)


def test_judge_prompt_is_one_user_message_with_the_pinned_wording():
    messages = build_judge_messages(QUESTION, "She went to Lyon.")

    assert messages == [{"role": "user", "content": JUDGE_PROMPT.format(evidence="")}]


def saved_record() -> dict[str, Any]:
    return {
        "sample_id": "conv-1",
        "question_id": "q3",
        "category": "2",
        "question": "Where did Alice go?",
        "gold_answer": "Lyon",
        "predicted_answer": "She went to Lyon.",
        "evidence": ["D1:2", "D1:1"],
        "evidence_context": EVIDENCE,
    }


def test_saved_record_is_judged_from_its_own_question_gold_answer_and_prediction():
    judge = Judge()

    verdict = judge_record(make_memory_config(), judge, saved_record())

    [(messages, _)] = judge.requests
    assert messages == build_judge_messages(QUESTION, "She went to Lyon.")
    assert verdict["correct"] is True


def test_prediction_record_holds_the_question_the_answer_the_memories_and_the_metrics():
    config = make_memory_config(run_id="run")
    answer = ChatResult("She went to Lyon.", ttft_ms=12.5, metrics={"kv_memory_tokens": 40})
    hits = [
        hit(
            "fact-1",
            memory="Alice went to Lyon.",
            created_at="2023-05-08",
            source_session_id="session_1",
            source_session_index=1,
            source_turn_id="conv-1:session_1:0",
            metadata={"kind": "fact"},
        ),
        hit(
            "fact-2",
            text="Older text key.",
            metadata={"session_id": "session_4", "session_index": 4, "turn_id": "t"},
        ),
        hit("fact-3"),
    ]

    record = build_prediction_record(
        config,
        conversation(),
        QUESTION,
        hits,
        answer,
        judge_payload={"correct": True},
        evidence=EVIDENCE,
        retrieval_metrics=RetrievalMetrics(
            embedding_time_ms=1.5,
            search_time_ms=2.5,
            total_time_ms=4.0,
            vector_backend="jasper",
            jasper_effective_beam_width=128,
        ),
    )

    assert record == {
        "run_id": "run",
        "mode": "mem0-kv",
        "sample_id": "conv-1",
        "question_id": "q3",
        "category": "2",
        "question": "Where did Alice go?",
        "gold_answer": "Lyon",
        "predicted_answer": "She went to Lyon.",
        "evidence": ["D1:2", "D1:1"],
        "evidence_context": EVIDENCE,
        "retrieved_memories": [
            {
                "id": "fact-1",
                "rank": 3,
                "score": 0.5,
                "distance": 0.25,
                "memory": "Alice went to Lyon.",
                "created_at": "2023-05-08",
                "source_session_id": "session_1",
                "source_session_index": 1,
                "source_turn_id": "conv-1:session_1:0",
                "metadata": {"kind": "fact"},
            },
            {
                "id": "fact-2",
                "rank": 3,
                "score": 0.5,
                "distance": 0.25,
                "memory": "Older text key.",
                "created_at": None,
                "source_session_id": "session_4",
                "source_session_index": 4,
                "source_turn_id": "t",
                "metadata": {"session_id": "session_4", "session_index": 4, "turn_id": "t"},
            },
            {
                "id": "fact-3",
                "rank": 3,
                "score": 0.5,
                "distance": 0.25,
                "memory": "",
                "created_at": None,
                "source_session_id": None,
                "source_session_index": None,
                "source_turn_id": None,
                "metadata": {},
            },
        ],
        "judge": {"correct": True},
        "metrics": {
            "kv_memory_tokens": 40,
            "memory_retrieved_fact_ids": ["fact-1", "fact-2", "fact-3"],
            "memory_retrieved_fact_text_hashes": [
                text_hash("Alice went to Lyon."),
                text_hash("Older text key."),
                text_hash(""),
            ],
            "time_to_first_token_ms": 12.5,
            "query_embedding_time_ms": 1.5,
            "vector_db_query_time_ms": 2.5,
            "query_retrieval_time_ms": 4.0,
            "resolved_vector_backend": "jasper",
            "jasper_effective_beam_width": 128,
        },
    }
    assert list(record) == [
        "run_id",
        "mode",
        "sample_id",
        "question_id",
        "category",
        "question",
        "gold_answer",
        "predicted_answer",
        "evidence",
        "evidence_context",
        "retrieved_memories",
        "judge",
        "metrics",
    ]
    assert answer.metrics == {"kv_memory_tokens": 40}


def answered(config: Any, judge: Any, answer: ChatResult | None = None, **options: Any) -> Any:
    answer_client = AnswerClient(answer or ChatResult("reasoning... ANSWER:  She went to Lyon. "))
    clients = RuntimeClients(answer_client=answer_client, judge_client=judge)
    sample = conversation()
    record = answer_question(config, clients, sample, QUESTION, [hit("fact-1")], **options)
    return record, answer_client.requests, sample


def test_question_is_answered_with_the_generation_settings_and_timing_marks():
    config = make_memory_config(
        inferscale={"generation": {"max_tokens": 64, "temperature": 0.5, "top_p": 0.9}},
        skip_judge=True,
    )

    _, requests, sample = answered(config, None, ttft_started_at=12.5, query_started_at=10.25)
    _, default_requests, _ = answered(config, None)

    [request] = requests
    assert request == {
        "sample": sample,
        "qa": QUESTION,
        "hits": [hit("fact-1")],
        "max_tokens": 64,
        "temperature": 0.5,
        "top_p": 0.9,
        "ttft_started_at": 12.5,
        "query_started_at": 10.25,
    }
    assert (default_requests[0]["ttft_started_at"], default_requests[0]["query_started_at"]) == (
        None,
        None,
    )
