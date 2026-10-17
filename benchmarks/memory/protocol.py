"""Fixed vocabulary and protocol constants of the memory benchmark.

Everything here is recorded into every run's ``config.json`` so results
remain comparable: the upstream scorer the benchmark reproduces, the Mem0
extraction settings, and the prompt protocols.
"""

from __future__ import annotations

from typing import Any

# Benchmark axes.
ANSWER_BACKENDS = ("kv-injection", "prompt-injection")
VECTOR_BACKENDS = ("jasper", "qdrant")
KV_STORE_BACKENDS = ("gpu", "cpu")

# LoCoMo question categories as the upstream memory-benchmarks scorer names
# them. Category 5 (adversarial) is never answered; its name stays so saved
# records still summarize.
CATEGORY_NAMES = {
    "1": "multi-hop",
    "2": "temporal",
    "3": "open-domain",
    "4": "single-hop",
    "5": "adversarial",
}
SCORED_CATEGORIES = frozenset({"1", "2", "3", "4"})

MEMORY_BENCHMARKS_REPOSITORY = "mem0ai/memory-benchmarks"
MEMORY_BENCHMARKS_COMMIT = "4b61c5d31b9c668a12b4f5e78064248a02c82d2b"
MEM0AI_VERSION = "2.0.11"
MEMORY_EXTRACTION_RESPONSE_PROTOCOL = "bounded-json-schema-v1"
# Extraction sampling temperature. Part of the fact-catalog and LLM-cache
# identity, so changing it never replays catalogs extracted at another value.
MEMORY_LLM_TEMPERATURE = 0.0
MEMORY_EXTRACTION_MAX_MODEL_LEN = 16384
MEMORY_EXTRACTION_MAX_TOKENS = 4096
MEMORY_EXTRACTION_MAX_FACTS = 5
MEMORY_EXTRACTION_MAX_TEXT_CHARS = 600
# Attempt 1 always uses the baseline sampling params (temperature 0.0); each
# retry after a failed extraction validation uses the next temperature here,
# with top_p widened to MEMORY_EXTRACTION_RETRY_TOP_P: mem0's config default
# top_p=0.1 is sent on every request and would otherwise keep the nucleus at
# essentially the argmax token, reproducing the same degenerate response.
# Retries never enter the cache identity; responses are stored under the
# baseline digest.
MEMORY_EXTRACTION_RETRY_TEMPERATURES = (0.2, 0.5)
MEMORY_EXTRACTION_RETRY_TOP_P = 1.0
MEMORY_INGESTION_PROTOCOL = "locomo-mem0-v3-chunk1-session-date-bounded-json-v2"
ANSWER_PROMPT_PROTOCOL = "reverse-ranked-memory-block-safe-v1"
# The judge answers with one boolean token; the name records the upstream
# JSON-scoring protocol whose verdicts it reproduces.
JUDGE_PROMPT_PROTOCOL = "memory-benchmarks-json-jscore-v1"

MEMORY_UNIT = "mem0-fact"
CONTEXT_WINDOW_UNIT = "turns"
CONTEXT_WINDOW_SEMANTICS = "encoding-prefix-discard-v1"

# Recorded verbatim in every run's config.json.
PROTOCOL_STAMP: dict[str, Any] = {
    "memory_unit": MEMORY_UNIT,
    "mem0_infer": True,
    "memory_benchmarks_repository": MEMORY_BENCHMARKS_REPOSITORY,
    "memory_benchmarks_commit": MEMORY_BENCHMARKS_COMMIT,
    "mem0ai_version": MEM0AI_VERSION,
    "memory_extraction_response_protocol": MEMORY_EXTRACTION_RESPONSE_PROTOCOL,
    "memory_extraction_max_model_len": MEMORY_EXTRACTION_MAX_MODEL_LEN,
    "memory_extraction_max_tokens": MEMORY_EXTRACTION_MAX_TOKENS,
    "memory_extraction_max_facts": MEMORY_EXTRACTION_MAX_FACTS,
    "memory_extraction_max_text_chars": MEMORY_EXTRACTION_MAX_TEXT_CHARS,
    "memory_ingestion_protocol": MEMORY_INGESTION_PROTOCOL,
    "answer_prompt_protocol": ANSWER_PROMPT_PROTOCOL,
    "judge_prompt_protocol": JUDGE_PROMPT_PROTOCOL,
    "context_window_unit": CONTEXT_WINDOW_UNIT,
    "context_window_semantics": CONTEXT_WINDOW_SEMANTICS,
}
