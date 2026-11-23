from __future__ import annotations

import pytest
from library_fakes import FakeTokenizer

from benchmarks.rag import answer_prompt_injection
from benchmarks.rag.answer_prompt_injection import RagPromptInjectionAnswerClient
from benchmarks.rag.corpus import load_rag_corpus
from benchmarks.rag.prompting import build_rag_query_tokens, extract_rag_scaffold_token_ids
from inferscale.v1.types import SearchHit


@pytest.fixture
def serving(multihop_config, engine, ticking_clock, monkeypatch):
    config = multihop_config()
    corpus = load_rag_corpus(config)
    ticking_clock(answer_prompt_injection)
    released: list[dict] = []
    monkeypatch.setattr(
        answer_prompt_injection, "empty_cuda_cache", lambda **options: released.append(options)
    )
    client = RagPromptInjectionAnswerClient(
        config, chunks_by_id=corpus.chunks_by_id, prompt_profile=corpus.spec.prompt_profile
    )
    return client, corpus, engine, released


def hits_for(*chunk_ids: str) -> list[SearchHit]:
    return [
        SearchHit(id=chunk_id, payload={}, score=1.0, distance=0.0, rank=rank)
        for rank, chunk_id in enumerate(chunk_ids, start=1)
    ]


def test_prompt_is_the_scaffolded_passages_followed_by_the_question(serving):
    client, corpus, engine, _ = serving
    client.start_llm()
    best, second_best = corpus.chunks[5], corpus.chunks[2]
    query = corpus.queries[1]

    answer = client.answer(query, hits_for(best.chunk_id, second_best.chunk_id))

    scaffold = extract_rag_scaffold_token_ids(
        FakeTokenizer(), system_prompt=corpus.spec.prompt_profile.system_prompt, block_size=16
    )
    memory_token_ids = [
        *scaffold.header_token_ids,
        *second_best.token_ids,
        *best.token_ids,
        *scaffold.footer_token_ids,
    ]
    query_tokens = build_rag_query_tokens(
        FakeTokenizer(),
        memory_token_ids,
        query,
        answer_instruction=corpus.spec.prompt_profile.answer_instruction,
    )
    assert engine.started_with is None
    assert engine.prompts == [[*memory_token_ids, *query_tokens.token_ids]]
    assert answer.content == f"answer over {len(engine.prompts[0])} prompt tokens"
    assert answer.ttft_ms == 12.0
    assert answer.metrics == {
        "kv_memory_tokens": len(memory_token_ids),
        "kv_query_tokens": len(query_tokens.token_ids),
        "kv_query_bos_stripped": 0,
        "kv_selected_chunk_ids": [second_best.chunk_id, best.chunk_id],
        "kv_block_size": 16,
        "retrieved_chunk_count": 2,
        "answer_generate_time_ms": 250.0,
        "answer_total_time_ms": 99250.0,
        "prompt_injection_engine_time_to_first_token_ms": 12.0,
        "answer_time_to_first_token_ms": 99012.0,
    }
