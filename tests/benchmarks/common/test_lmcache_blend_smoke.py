"""GPU smoke test of the LMCache CacheBlend arm against the real vLLM and LMCache.

Run on the pod with ``RUN_LMCACHE_INTEGRATION=1``: it starts a small engine with
``BlendConnector``, warms two facts, and checks that a blend prompt matches
exactly the prefix the layout predicts and still answers from the facts.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("vllm")
pytest.importorskip("lmcache")

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LMCACHE_INTEGRATION") != "1",
    reason="set RUN_LMCACHE_INTEGRATION=1 on a GPU host with vllm and lmcache installed",
)

MODEL = os.environ.get("LMCACHE_SMOKE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
FACTS = (
    "Alice adopted a grey cat named Miso in 2023.\n",
    "Alice moved to Berlin in March 2021.\n",
)


def test_blend_prompt_matches_the_warmed_facts_and_answers_from_them() -> None:
    from benchmarks.common.lmcache.config import (
        LMCacheConfig,
        engine_config_for_lmcache,
        lmcache_kv_transfer_config,
    )
    from benchmarks.common.lmcache.load_stats import reset_load_stats, snapshot_load_stats
    from benchmarks.common.lmcache.runtime import (
        clear_cache,
        kv_bytes_per_token,
        register_model,
        require_separator_parity,
        shutdown,
        warm_prompts,
    )
    from benchmarks.common.lmcache.segments import blend_memory_layout, warm_up_prompts
    from benchmarks.memory.data import QuestionAnswer
    from benchmarks.memory.prompting import (
        build_query_tokens_for_memory,
        extract_memory_scaffold_token_ids,
    )
    from inferscale.v1.config import EngineConfig
    from inferscale.v1.kv.prompt import build_prompt_tokens
    from inferscale.v1.kv.tokenization import encode_text_no_special
    from inferscale.v1.serving.vllm import VLLMEngine, force_vllm_inprocess_mode

    force_vllm_inprocess_mode()
    settings = LMCacheConfig(max_local_cpu_size_gb=2)
    engine_config = engine_config_for_lmcache(
        EngineConfig(gpu_memory_utilization=0.4, max_model_len=4096)
    )
    engine = VLLMEngine(MODEL, "bfloat16", engine_config)
    try:
        engine.start(kv_transfer_config=lmcache_kv_transfer_config(settings))
        register_model(engine.llm)
        tokenizer = engine.tokenizer
        separator = require_separator_parity(tokenizer, settings.blend_special_str)
        assert kv_bytes_per_token() > 0

        scaffold = extract_memory_scaffold_token_ids(tokenizer)
        chunks = [encode_text_no_special(tokenizer, fact) for fact in FACTS]
        clear_cache()
        warm_prompts(
            engine.llm, warm_up_prompts(scaffold.header_token_ids, chunks, separator), batch_size=8
        )

        layout = blend_memory_layout(scaffold, chunks, separator)
        qa = QuestionAnswer(
            sample_id="smoke",
            question_id="q1",
            question="What is the name of Alice's cat?",
            answer="Miso",
            category="4",
        )
        query = build_query_tokens_for_memory(tokenizer, layout.token_ids, qa)
        prompt = build_prompt_tokens(layout.token_ids, query)
        reset_load_stats()
        output = engine.generate(prompt.prompt_token_ids, max_tokens=16, temperature=0.0, top_p=1.0)
        stats = snapshot_load_stats()

        assert stats["tokens_matched"] == layout.expected_matched_tokens, stats
        assert output.ttft_ms > 0
        assert "miso" in output.text.lower(), output.text
    finally:
        shutdown()
        engine.close()
