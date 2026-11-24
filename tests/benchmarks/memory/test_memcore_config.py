from __future__ import annotations

import json
from typing import Any

import pytest
from memory_config import make_memory_config, memory_config_data

from benchmarks.common.config import ConfigError
from benchmarks.memory.protocol import PROTOCOL_STAMP

LMCACHE_SWEEP = {
    "top_k": [5],
    "variants": [{"name": "blend", "answer_backend": "lmcache-blend", "vector_backend": "qdrant"}],
}
KV_SWEEP = {
    "top_k": [5],
    "variants": [{"name": "kv", "answer_backend": "kv-injection", "vector_backend": "jasper"}],
}
WITHOUT_PREFIX_CACHING = {
    **memory_config_data()["inferscale"],
    "engine": {**memory_config_data()["inferscale"]["engine"], "enable_prefix_caching": False},
}


def rejection(**settings: Any) -> str:
    with pytest.raises(ConfigError) as rejected:
        make_memory_config(**settings)
    return str(rejected.value)


def test_backend_requirements_are_checked_after_the_library_section_and_before_the_sweeps():
    prompt_injection_without_prefix_caching = {
        "answer_backend": "prompt-injection",
        "inferscale": WITHOUT_PREFIX_CACHING,
    }

    assert rejection(**prompt_injection_without_prefix_caching, sweeps=5) == (
        "memory config: answer_backend prompt-injection requires "
        "inferscale.engine.enable_prefix_caching."
    )


def test_lmcache_section_is_required_by_an_lmcache_run_or_sweep():
    required = (
        "memory config.lmcache is required when answer_backend or a sweep variant is "
        "'lmcache-blend'."
    )

    assert rejection(answer_backend="lmcache-blend", lmcache=None, sweeps={}) == required
    assert rejection(lmcache=None, sweeps={"blend": LMCACHE_SWEEP}) == required
    assert rejection(lmcache=None, sweeps={"kv": KV_SWEEP, "blend": LMCACHE_SWEEP}) == required


@pytest.mark.parametrize("settings", [{}])
def test_protocol_stamp_is_saved_verbatim_at_the_end_of_every_run_config(settings):
    saved = list(make_memory_config(**settings).to_jsonable().items())

    assert saved[-len(PROTOCOL_STAMP) :] == list(PROTOCOL_STAMP.items())


def test_saved_config_lists_the_settings_then_the_protocol_stamp():
    saved = make_memory_config(run_id="run").to_jsonable()

    assert list(saved) == [
        "benchmark",
        "model",
        "label",
        "dataset_path",
        "results_dir",
        "run_id",
        "answer_backend",
        "vector_backend",
        "top_k",
        "context_window",
        "max_samples",
        "max_questions",
        "log_every",
        "log_level",
        "preembed_workers",
        "embedding_cache_enabled",
        "embedding_cache_dir",
        "kv_chunk_cache_enabled",
        "cache_root",
        "local_store_dir",
        "skip_judge",
        "rejudge",
        "memory_llm_provider",
        "memory_llm_model",
        "memory_llm_base_url",
        "memory_llm_api_key",
        "memory_llm_cache_dir",
        "inferscale",
        "judge",
        "with_evidence",
        "jasper_effective_beam_width",
        "extraction",
        "lmcache",
        "sweeps",
        "memory_unit",
        "mem0_infer",
        "memory_benchmarks_repository",
        "memory_benchmarks_commit",
        "mem0ai_version",
        "memory_extraction_response_protocol",
        "memory_extraction_max_model_len",
        "memory_extraction_max_tokens",
        "memory_extraction_max_facts",
        "memory_extraction_max_text_chars",
        "memory_ingestion_protocol",
        "answer_prompt_protocol",
        "judge_prompt_protocol",
        "context_window_unit",
        "context_window_semantics",
    ]
    assert json.loads(json.dumps(saved)) == saved
    assert (saved["benchmark"], saved["model"], saved["label"]) == (
        "memory-accuracy-latency",
        "meta-llama/Llama-3.1-8B-Instruct",
        "llama",
    )
    assert saved["sweeps"]["lmcache"] == {
        "top_k": [50],
        "variants": [
            {
                "name": "lmcache-blend",
                "answer_backend": "lmcache-blend",
                "vector_backend": "qdrant",
                "kv_store_backend": "gpu",
                "kv_staging_slots": None,
                "context_window": [0],
            }
        ],
    }
    assert saved["extraction"]["extra_vllm_args"] == []
    assert make_memory_config(lmcache=None, sweeps={}).to_jsonable()["lmcache"] is None
