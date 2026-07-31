from __future__ import annotations

from typing import Any

from loguru import logger


def load_rag_tokenizer(model: str, *, allow_transformers_fallback: bool = False) -> Any:
    """Load the answer model's tokenizer through vLLM's resolver.

    Chunk token ids must be byte-identical to what the vLLM engine sees (the
    KV connector matches injected chunks as a prompt prefix), so every stage
    that produces token ids uses the same resolver as the engine. The
    transformers AutoTokenizer fallback exists only for --estimate-only on
    hosts without vLLM; it may tokenize differently (notably for Mistral) and
    must never feed the embedding, KV, or answer stages.
    """
    try:
        from locomo_jasper_bench.kv.chunked_rope import load_encoder_tokenizer
    except ModuleNotFoundError as exc:  # pragma: no cover - import environment specific
        raise RuntimeError(
            "locomo_jasper_bench is required to load the RAG tokenizer."
        ) from exc

    try:
        return load_encoder_tokenizer(model)
    except ModuleNotFoundError as exc:
        if not allow_transformers_fallback:
            raise RuntimeError(
                "vLLM is required to load the tokenizer for this stage so token ids "
                "match the engine. Run this command on the GPU host, or use "
                "--estimate-only which permits the transformers fallback."
            ) from exc
        logger.warning(
            "vLLM is unavailable ({}); falling back to transformers AutoTokenizer for "
            "model {}. Estimates only: token counts may differ slightly from the engine.",
            exc,
            model,
        )
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model)
