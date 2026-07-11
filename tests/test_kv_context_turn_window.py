from __future__ import annotations

from typing import Any

import pytest

from locomo_jasper_bench.data import ConversationSample, QuestionAnswer, Turn
from locomo_jasper_bench.config import BenchmarkConfig, parse_args
from locomo_jasper_bench.kv.chunked_rope import ChunkedRopeSampleComposer
from locomo_jasper_bench.kv.answer_client import _require_same_memory_token_ids
from locomo_jasper_bench.kv.context import (
    build_fact_context_encoding_plan,
    build_memory_fact_plan,
    reverse_ranked_memory_facts,
    format_memory_turn,
    memory_context_metrics,
    memory_fact_from_hit,
    previous_turn_context_turns,
    unique_memory_facts,
)
from locomo_jasper_bench.kv.prompting import (
    EMPTY_MEMORY_TEXT,
    MEMORY_SYSTEM_PROMPT,
    build_kv_equivalence_prompt_from_query_tokens,
    build_kv_query_tokens_for_memory,
    build_memory_prompt_token_ids,
    extract_memory_scaffold_token_ids,
    format_memory_fact,
)
from locomo_jasper_bench.kv.tokenization import encode_text_no_special
from locomo_jasper_bench.kv.types import EncodedChunk, FactContextEncodingPlan
from locomo_jasper_bench.kv.vllm_runtime import common_vllm_kwargs
from locomo_jasper_bench.vector_types import SearchHit


def _turn(session_index: int, turn_index: int) -> Turn:
    return Turn(
        sample_id="sample-1",
        session_id=f"session_{session_index}",
        session_index=session_index,
        turn_index=turn_index,
        speaker="speaker",
        text=f"session {session_index}, turn {turn_index}",
        timestamp=f"2023-01-0{session_index}",
    )


def _sample() -> ConversationSample:
    return ConversationSample(
        sample_id="sample-1",
        turns=[_turn(1, 0), _turn(2, 0), _turn(3, 0), _turn(4, 0)],
        qa=[],
        raw={},
    )


def _hit(
    memory_id: str,
    text: str,
    session_index: int,
    *,
    turn_index: int = 0,
    created_at: str | None = None,
    rank: int = 1,
) -> SearchHit:
    return SearchHit(
        id=memory_id,
        payload={
            "memory": text,
            "created_at": created_at or f"2023-01-0{session_index}T00:00:00Z",
            "source_session_index": session_index,
            "source_session_id": f"session_{session_index}",
            "source_turn_index": turn_index,
            "source_turn_id": f"sample-1:session_{session_index}:{turn_index}",
        },
        score=float(-rank),
        distance=float(rank),
        rank=rank,
    )


def test_fact_parser_uses_promoted_metadata_with_nested_fallback() -> None:
    promoted = memory_fact_from_hit(_hit("m1", "Alice likes tea.", 2))
    fallback = memory_fact_from_hit(
        SearchHit(
            id="m2",
            payload={
                "memory": "Bob likes coffee.",
                "created_at": "2023-01-03",
                "metadata": {
                    "session_index": 3,
                    "session_id": "session_3",
                    "turn_id": "sample-1:session_3:0",
                },
            },
            score=1.0,
            distance=0.0,
            rank=1,
        )
    )

    assert promoted.memory_id == "m1"
    assert promoted.text == "Alice likes tea."
    assert len(promoted.text_hash) == 64
    assert fallback.source_session_index == 3
    assert fallback.source_session_id == "session_3"
    assert fallback.source_turn_id == "sample-1:session_3:0"


def test_fact_selection_is_deduplicated_and_reverse_ranked() -> None:
    hits = [
        _hit("m3b", "third-b", 3, turn_index=1, rank=1),
        _hit("m1", "first", 1, rank=2),
        _hit("m3a", "third-a", 3, turn_index=0, rank=3),
        _hit("m2", "second", 2, rank=4),
        _hit("m2", "second", 2, rank=5),
    ]

    facts = reverse_ranked_memory_facts(hits)

    assert [fact.memory_id for fact in facts] == ["m2", "m3a", "m1", "m3b"]
    assert len(unique_memory_facts(hits)) == 4


def test_reverse_ranked_facts_ignore_source_turn_order() -> None:
    facts = reverse_ranked_memory_facts(
        [
            _hit("turn-10", "later", 1, turn_index=10, rank=1),
            _hit("turn-2", "earlier", 1, turn_index=2, rank=2),
        ]
    )

    assert [fact.memory_id for fact in facts] == ["turn-2", "turn-10"]


def test_conflicting_duplicate_fact_ids_are_rejected() -> None:
    with pytest.raises(RuntimeError, match="Conflicting Mem0 fact payloads"):
        unique_memory_facts(
            [_hit("same", "first", 1), _hit("same", "different", 1)]
        )


def test_window_selects_immediately_preceding_turns_across_sessions() -> None:
    sample = _sample()
    target = memory_fact_from_hit(_hit("target", "target", 3))
    tokenizer = _DeterministicTokenizer()
    expected_context = [
        token_id
        for turn in (sample.turns[0], sample.turns[1])
        for token_id in encode_text_no_special(tokenizer, format_memory_turn(turn))
    ]

    plan = build_fact_context_encoding_plan(
        target,
        sample,
        tokenizer=tokenizer,
        context_window=2,
        max_input_tokens=10_000,
        fact_token_ids={"target": [30, 31]},
    )

    assert plan.context_token_ids == expected_context
    assert plan.input_token_ids == expected_context + [30, 31]
    assert (plan.slice_start, plan.slice_end) == (
        len(expected_context),
        len(expected_context) + 2,
    )
    assert plan.context_turn_ids == (sample.turns[0].id, sample.turns[1].id)


def test_window_excludes_source_turn_and_uses_available_prefix_at_start() -> None:
    sample = _sample()
    first = memory_fact_from_hit(_hit("first", "first", 1))
    tokenizer = _DeterministicTokenizer()

    assert previous_turn_context_turns(sample, first.source_turn_id, 3) == []
    plan = build_fact_context_encoding_plan(
        first,
        sample,
        tokenizer=tokenizer,
        context_window=3,
        max_input_tokens=10_000,
        fact_token_ids={"first": [10]},
    )

    assert plan.context_token_ids == []
    assert plan.context_turn_ids == ()
    assert plan.input_token_ids == [10]


def test_zero_window_returns_no_context_and_negative_raises() -> None:
    sample = _sample()

    assert previous_turn_context_turns(sample, sample.turns[2].id, 0) == []
    with pytest.raises(ValueError, match=">= 0"):
        previous_turn_context_turns(sample, sample.turns[2].id, -1)
    with pytest.raises(RuntimeError, match="not present in sample"):
        previous_turn_context_turns(sample, "sample-1:session_9:9", 1)


def test_context_overflow_truncates_oldest_tokens_and_never_target() -> None:
    sample = _sample()
    target = memory_fact_from_hit(_hit("target", "target", 3))
    tokenizer = _DeterministicTokenizer()
    full_context = [
        token_id
        for turn in (sample.turns[0], sample.turns[1])
        for token_id in encode_text_no_special(tokenizer, format_memory_turn(turn))
    ]
    max_input_tokens = len(full_context) - 3 + 2  # force a 3-token overflow

    plan = build_fact_context_encoding_plan(
        target,
        sample,
        tokenizer=tokenizer,
        context_window=2,
        max_input_tokens=max_input_tokens,
        fact_token_ids={"target": [30, 31]},
    )

    assert plan.raw_context_tokens == len(full_context)
    assert plan.context_truncated_tokens == 3
    assert plan.context_token_ids == full_context[3:]
    assert plan.target_token_ids == [30, 31]
    assert plan.input_token_ids == full_context[3:] + [30, 31]
    assert plan.slice_start == len(full_context) - 3


def test_target_fact_larger_than_encoding_limit_fails() -> None:
    target = memory_fact_from_hit(_hit("target", "target", 1))

    with pytest.raises(RuntimeError, match="without context"):
        build_fact_context_encoding_plan(
            target,
            _sample(),
            tokenizer=_DeterministicTokenizer(),
            context_window=5,
            max_input_tokens=1,
            fact_token_ids={"target": [1, 2]},
        )


class _DeterministicTokenizer:
    bos_token_id = None

    def encode(self, text: str, **_: Any) -> list[int]:
        return [ord(character) for character in text]


class _FakeChunkEncoder:
    model = "fake-model"
    device = "cpu"
    max_position = 10_000

    def __init__(self) -> None:
        self.tokenizer = _DeterministicTokenizer()
        self.fact_plans: list[FactContextEncodingPlan] = []

    def encode_token_ids_chunk(self, chunk_id: str, token_ids: list[int]) -> EncodedChunk:
        del chunk_id
        return EncodedChunk(token_ids=list(token_ids), kv_by_layer={})

    def encode_fact_chunk(self, plan: FactContextEncodingPlan) -> EncodedChunk:
        self.fact_plans.append(plan)
        return EncodedChunk(
            token_ids=list(plan.target_token_ids),
            kv_by_layer={},
            context_turn_ids=plan.context_turn_ids,
            context_prefix_tokens=len(plan.context_token_ids),
            raw_context_prefix_tokens=plan.raw_context_tokens,
            context_prefix_truncated_tokens=plan.context_truncated_tokens,
        )


def _fake_composer(
    monkeypatch: pytest.MonkeyPatch,
    *,
    context_window: int,
    catalog: list[SearchHit],
) -> tuple[ChunkedRopeSampleComposer, _FakeChunkEncoder]:
    encoder = _FakeChunkEncoder()
    composer = ChunkedRopeSampleComposer(  # type: ignore[arg-type]
        encoder=encoder,
        context_window=context_window,
    )
    composer.encode_sample(_sample(), catalog)
    monkeypatch.setattr(
        ChunkedRopeSampleComposer,
        "_compose_chunks",
        lambda self, chunks: {},
    )
    return composer, encoder


def test_w0_prefix_and_kv_use_identical_header_fact_footer_question_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = _sample()
    qa = QuestionAnswer(
        sample_id=sample.sample_id,
        question_id="q1",
        question="What happened?",
        answer="Something",
        category="1",
    )
    catalog = [
        _hit("later", "Later fact.", 3, rank=1),
        _hit("earlier", "Earlier fact.", 1, rank=2),
    ]
    composer, encoder = _fake_composer(
        monkeypatch,
        context_window=0,
        catalog=catalog,
    )

    composed = composer.compose(catalog, memory_token_budget=10_000)
    scaffold = extract_memory_scaffold_token_ids(encoder.tokenizer, sample)
    header_text = "".join(chr(token_id) for token_id in scaffold.header_token_ids)
    prompted = build_memory_prompt_token_ids(
        encoder.tokenizer,
        sample,
        catalog,
        context_window=0,
        memory_token_budget=10_000,
        memory_scaffold=scaffold,
    )
    query_tokens = build_kv_query_tokens_for_memory(
        encoder.tokenizer,
        scaffold.header_token_ids,
        sample,
        qa,
        memory_scaffold=scaffold,
    )
    kv_prompt = build_kv_equivalence_prompt_from_query_tokens(
        composed.token_ids,
        query_tokens,
    )
    prefix_prompt = build_kv_equivalence_prompt_from_query_tokens(
        prompted.token_ids,
        query_tokens,
    )

    assert composed.selected_fact_ids == ["earlier", "later"]
    assert prompted.selected_fact_ids == ["earlier", "later"]
    assert composed.fact_plan.retrieved_fact_ids == ("later", "earlier")
    assert prompted.fact_plan.retrieved_fact_ids == ("later", "earlier")
    assert composed.fact_plan.injected_fact_ids == ("earlier", "later")
    assert MEMORY_SYSTEM_PROMPT in header_text
    assert composed.token_ids == prompted.token_ids
    assert kv_prompt.prompt_token_ids == prefix_prompt.prompt_token_ids
    assert all(plan.context_token_ids == [] for plan in encoder.fact_plans)


def test_w_greater_than_zero_discards_prefix_tokens_from_visible_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = [
        _hit("context", "Context fact.", 1),
        _hit("target", "Target fact.", 2),
    ]
    composer, encoder = _fake_composer(
        monkeypatch,
        context_window=1,
        catalog=catalog,
    )
    target_plan = next(plan for plan in encoder.fact_plans if plan.memory_id == "target")
    target_chunk = composer.chunks["target"]

    assert target_plan.context_token_ids
    assert target_plan.slice_start == len(target_plan.context_token_ids)
    assert target_plan.input_token_ids == (
        target_plan.context_token_ids + target_plan.target_token_ids
    )
    assert target_chunk.token_ids == target_plan.target_token_ids
    assert len(target_chunk.token_ids) < len(target_plan.input_token_ids)


def test_empty_results_render_upstream_marker_and_compose_scaffold_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = _sample()
    composer, encoder = _fake_composer(
        monkeypatch,
        context_window=0,
        catalog=[],
    )

    composed = composer.compose([], memory_token_budget=10_000)
    scaffold = extract_memory_scaffold_token_ids(encoder.tokenizer, sample)
    prompted = build_memory_prompt_token_ids(
        encoder.tokenizer,
        sample,
        [],
        memory_token_budget=10_000,
        memory_scaffold=scaffold,
    )

    assert composed.selected_fact_ids == []
    assert prompted.selected_fact_ids == []
    assert composed.token_ids == prompted.token_ids
    assert encode_text_no_special(encoder.tokenizer, EMPTY_MEMORY_TEXT) == (
        scaffold.empty_memory_token_ids
    )
    assert memory_context_metrics(composed.fact_plan) == {
        "memory_context_window": 0,
        "memory_retrieved_fact_ids": [],
        "memory_retrieved_fact_text_hashes": [],
        "memory_injected_fact_ids": [],
        "memory_context_turn_ids": [],
        "memory_context_turn_count": 0,
        "memory_context_encoding_tokens_total": 0,
        "memory_context_encoding_tokens_max": 0,
        "memory_context_encoding_truncated_tokens": 0,
        "memory_context_text_tokens": 0,
        "memory_token_budget": 10_000,
    }


def test_fact_metrics_report_selected_context_encoding() -> None:
    selected = reverse_ranked_memory_facts([_hit("target", "Target.", 3)])
    chunk = EncodedChunk(
        token_ids=[30],
        kv_by_layer={},
        context_turn_ids=("sample-1:session_1:0", "sample-1:session_2:0"),
        context_prefix_tokens=5,
        raw_context_prefix_tokens=8,
        context_prefix_truncated_tokens=3,
    )
    plan = build_memory_fact_plan(
        selected,
        context_window=2,
        memory_token_budget=10,
        scaffold_token_count=2,
        fact_token_ids={"target": [30]},
        encoded_chunks={"target": chunk},
    )

    assert memory_context_metrics(plan) == {
        "memory_context_window": 2,
        "memory_retrieved_fact_ids": ["target"],
        "memory_retrieved_fact_text_hashes": [selected[0].text_hash],
        "memory_injected_fact_ids": ["target"],
        "memory_context_turn_ids": ["sample-1:session_1:0", "sample-1:session_2:0"],
        "memory_context_turn_count": 2,
        "memory_context_encoding_tokens_total": 5,
        "memory_context_encoding_tokens_max": 5,
        "memory_context_encoding_truncated_tokens": 3,
        "memory_context_text_tokens": 0,
        "memory_token_budget": 10,
    }


def test_visible_memory_budget_never_drops_retrieved_facts() -> None:
    selected = reverse_ranked_memory_facts([_hit("target", "Target.", 1)])

    with pytest.raises(RuntimeError, match="retrieved Mem0 facts cannot fit"):
        build_memory_fact_plan(
            selected,
            context_window=0,
            memory_token_budget=2,
            scaffold_token_count=2,
            fact_token_ids={"target": [30]},
        )


def test_kv_block_size_is_configured_for_both_vllm_backends() -> None:
    config = parse_args(["--skip-judge", "--kv-block-size", "32"])

    assert config.kv_block_size == 32
    assert common_vllm_kwargs(config)["block_size"] == 32
    assert common_vllm_kwargs(BenchmarkConfig(answer_backend="vllm-prefix"))["block_size"] == 16


def test_negative_or_zero_kv_block_size_is_rejected() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--skip-judge", "--kv-block-size", "0"])


def test_live_vllm_memory_tokens_must_match_precomputed_hf_tokens() -> None:
    _require_same_memory_token_ids([1, 2, 3], [1, 2, 3])

    with pytest.raises(RuntimeError, match="differ from the live vLLM tokenizer at index=1"):
        _require_same_memory_token_ids([1, 2, 3], [1, 9, 3])


def test_prefix_renders_deduplicated_context_turns_before_their_facts() -> None:
    sample = _sample()
    tokenizer = _DeterministicTokenizer()
    scaffold = extract_memory_scaffold_token_ids(tokenizer, sample)
    # Both facts come from session 3 turn 0 and session 4 turn 0; with a
    # 2-turn window their contexts overlap on session 2/3 turns.
    hits = [
        _hit("late", "Late fact.", 4, rank=1),
        _hit("early", "Early fact.", 3, rank=2),
    ]

    prompted = build_memory_prompt_token_ids(
        tokenizer,
        sample,
        hits,
        context_window=2,
        memory_token_budget=10_000,
        memory_scaffold=scaffold,
        render_context_turns=True,
    )

    turn_tokens = {
        turn.id: encode_text_no_special(tokenizer, format_memory_turn(turn))
        for turn in sample.turns
    }
    fact_tokens = {
        fact.memory_id: encode_text_no_special(tokenizer, format_memory_fact(fact))
        for fact in reverse_ranked_memory_facts(hits)
    }
    # Facts remain reverse-ranked. Context turns stay chronological within
    # each fact's window and are rendered only on their first occurrence.
    expected_stream = (
        turn_tokens[sample.turns[0].id]
        + turn_tokens[sample.turns[1].id]
        + fact_tokens["early"]
        + turn_tokens[sample.turns[2].id]
        + fact_tokens["late"]
    )
    expected = (
        list(scaffold.header_token_ids)
        + list(scaffold.memory_list_header_token_ids)
        + expected_stream
        + list(scaffold.footer_token_ids)
    )

    assert prompted.token_ids == expected
    assert prompted.fact_plan.context_turn_ids == (
        sample.turns[0].id,
        sample.turns[1].id,
        sample.turns[2].id,
    )
    assert prompted.fact_plan.context_text_tokens == sum(
        len(turn_tokens[sample.turns[index].id]) for index in range(3)
    )
    assert prompted.fact_plan.memory_tokens == len(expected)


def test_prefix_without_context_rendering_matches_kv_fact_only_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = [
        _hit("context", "Context fact.", 1),
        _hit("target", "Target fact.", 2),
    ]
    composer, encoder = _fake_composer(
        monkeypatch,
        context_window=1,
        catalog=catalog,
    )
    sample = _sample()
    scaffold = extract_memory_scaffold_token_ids(encoder.tokenizer, sample)

    composed = composer.compose(catalog, memory_token_budget=10_000)
    fact_only = build_memory_prompt_token_ids(
        encoder.tokenizer,
        sample,
        catalog,
        context_window=1,
        memory_token_budget=10_000,
        memory_scaffold=scaffold,
    )
    with_context = build_memory_prompt_token_ids(
        encoder.tokenizer,
        sample,
        catalog,
        context_window=1,
        memory_token_budget=10_000,
        memory_scaffold=scaffold,
        render_context_turns=True,
    )

    # The KV verification path (render_context_turns=False) stays token-identical
    # to the composed KV memory at any window; the prefix answer path renders
    # the context turns and is deliberately longer.
    assert composed.token_ids == fact_only.token_ids
    assert len(with_context.token_ids) > len(fact_only.token_ids)
    assert with_context.fact_plan.context_text_tokens > 0
    assert fact_only.fact_plan.context_text_tokens == 0
