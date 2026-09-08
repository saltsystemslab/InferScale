from __future__ import annotations


MEMORY_BENCHMARKS_REPOSITORY = "mem0ai/memory-benchmarks"
MEMORY_BENCHMARKS_COMMIT = "4b61c5d31b9c668a12b4f5e78064248a02c82d2b"
MEM0AI_VERSION = "2.0.11"
MEMORY_EXTRACTION_RESPONSE_PROTOCOL = "bounded-json-schema-v1"
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
JUDGE_PROMPT_PROTOCOL = "memory-benchmarks-json-jscore-v1"
