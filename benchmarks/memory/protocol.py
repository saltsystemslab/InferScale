from __future__ import annotations

from typing import Any

LMCACHE_BACKEND = "lmcache-blend"
PROMPT_INJECTION_BACKEND = "prompt-injection"
ANSWER_BACKENDS = ("kv-injection", PROMPT_INJECTION_BACKEND, LMCACHE_BACKEND)
WHY_BACKEND_TAKES_NO_CONTEXT_WINDOW = {
    LMCACHE_BACKEND: "stores each fact alone",
    PROMPT_INJECTION_BACKEND: "sends only the retrieved facts",
}
VECTOR_BACKENDS = ("jasper", "qdrant")
KV_STORE_BACKENDS = ("gpu", "cpu")

SCORED_CATEGORY_NAMES = {
    "1": "multi-hop",
    "2": "temporal",
    "3": "open-domain",
    "4": "single-hop",
}
UNSCORED_CATEGORY_NAMES = {"5": "adversarial"}
CATEGORY_NAMES = {**SCORED_CATEGORY_NAMES, **UNSCORED_CATEGORY_NAMES}
SCORED_CATEGORIES = frozenset(SCORED_CATEGORY_NAMES)

MEMORY_BENCHMARKS_REPOSITORY = "mem0ai/memory-benchmarks"
MEMORY_BENCHMARKS_COMMIT = "4b61c5d31b9c668a12b4f5e78064248a02c82d2b"
MEM0AI_VERSION = "2.0.11"
MEMORY_EXTRACTION_RESPONSE_PROTOCOL = "bounded-json-schema-v1"
MEMORY_LLM_TEMPERATURE = 0.0
MEMORY_EXTRACTION_MAX_MODEL_LEN = 16384
MEMORY_EXTRACTION_MAX_TOKENS = 4096
MEMORY_EXTRACTION_MAX_FACTS = 5
MEMORY_EXTRACTION_MAX_TEXT_CHARS = 600
MEMORY_EXTRACTION_RETRY_TEMPERATURES = (0.2, 0.5)
MEMORY_EXTRACTION_RETRY_TOP_P = 1.0
MEMORY_INGESTION_PROTOCOL = "locomo-mem0-v3-chunk1-session-date-bounded-json-v2"
FACT_CATALOG_FORMAT_VERSION = 6
ANSWER_PROMPT_PROTOCOL = "reverse-ranked-memory-block-safe-v1"
JUDGE_PROMPT_PROTOCOL = "memory-benchmarks-json-jscore-v1"

MEMORY_UNIT = "mem0-fact"
CONTEXT_WINDOW_UNIT = "turns"
CONTEXT_WINDOW_SEMANTICS = "previous-turn-facts-prefix-discard-v1"

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
