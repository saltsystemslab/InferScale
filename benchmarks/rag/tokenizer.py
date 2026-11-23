from __future__ import annotations

from typing import Any

from loguru import logger


def load_rag_tokenizer(model: str, *, allow_transformers_fallback: bool = False) -> Any:
    try:
        from inferscale.v1.kv.encoder import load_encoder_tokenizer
    except ModuleNotFoundError as exc:  # pragma: no cover - import environment specific
        raise RuntimeError("inferscale is required to load the RAG tokenizer.") from exc

    try:
        return load_encoder_tokenizer(model)
    except ModuleNotFoundError as exc:
        if not allow_transformers_fallback:
            raise RuntimeError(
                "vLLM is required to load the tokenizer for this stage so token ids "
                "match the engine. Run this stage on the GPU host, or select "
                "the estimate stage, which permits the transformers fallback."
            ) from exc
        logger.warning(
            "vLLM is unavailable ({}); falling back to transformers AutoTokenizer for "
            "model {}. Estimates only: token counts may differ slightly from the engine.",
            exc,
            model,
        )
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model)
