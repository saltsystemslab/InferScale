from __future__ import annotations

import pytest

from benchmarks.common.config import ConfigError, RuntimeConfig
from benchmarks.rag.config import RagBenchConfig


@pytest.fixture
def bare_runtime(tmp_path) -> RuntimeConfig:
    return RuntimeConfig.from_dict({}, root=tmp_path)


def parse(settings: dict, runtime: RuntimeConfig) -> RagBenchConfig:
    return RagBenchConfig.from_dict({"run_id": "run", **settings}, runtime=runtime)


def test_rag_answers_are_capped_shorter_than_the_library_default_the_memory_benchmark_uses(
    bare_runtime,
):
    from inferscale.v1.config import GenerationConfig

    assert parse({}, bare_runtime).max_answer_tokens == 64 < GenerationConfig().max_tokens
    authored = parse({"inferscale": {"generation": {"max_tokens": 200}}}, bare_runtime)
    assert authored.max_answer_tokens == 200


@pytest.mark.parametrize("kv_section", [{}, {"device": "cuda:1"}])
def test_corpus_kv_always_stays_in_host_ram(bare_runtime, kv_section):
    config = parse({"inferscale": {"kv": kv_section}}, bare_runtime)

    assert config.to_jsonable()["kv_store_backend"] == "cpu"


def test_memory_budget_leaves_512_tokens_for_the_scaffold_and_the_question(bare_runtime):
    limits = {"inferscale": {"kv": {"max_position": 4096}}}
    fits = parse({"top_k": 7, "chunk_size": 512, **limits}, bare_runtime)

    with pytest.raises(ConfigError) as raised:
        parse({"top_k": 7, "chunk_size": 513, **limits}, bare_runtime)

    assert fits.top_k * fits.chunk_size + 512 == 4096
    assert str(raised.value) == (
        "top_k x chunk_size + 512 scaffold/query margin exceeds the memory budget "
        "min(inferscale.kv.max_position, inferscale.engine.max_model_len - "
        "inferscale.generation.max_tokens) = 4096."
    )
