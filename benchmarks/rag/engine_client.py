"""What every RAG answer client shares: a vLLM engine, the prompt scaffold, and the budget check."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from loguru import logger

from benchmarks.common.clients import ChatResult
from benchmarks.rag.config import RagBenchConfig
from benchmarks.rag.data_types import RagChunk, RagPromptProfile, RagQuery
from benchmarks.rag.prompting import (
    build_rag_query_tokens,
    extract_rag_scaffold_token_ids,
    require_memory_within_budget,
)
from inferscale.v1.config import EngineConfig
from inferscale.v1.kv.prompt import PromptTokens, QueryTokens, ScaffoldTokens, memory_token_budget
from inferscale.v1.serving.vllm import GenerationOutput, VLLMEngine, force_vllm_inprocess_mode
from inferscale.v1.types import SearchHit


class RagAnswerClient(Protocol):
    """The contract the answer run drives."""

    def start_llm(self) -> None: ...

    def answer(
        self, query: RagQuery, hits: list[SearchHit], *, query_started_at: float | None = None
    ) -> ChatResult: ...

    def setup_stats(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


class RagEngineClient:
    """A vLLM engine over the corpus chunks, started with the dataset's prompt scaffold.

    Subclasses add how passages reach the model: injected KV or prompt text.
    """

    label = "answer"

    def __init__(
        self,
        config: RagBenchConfig,
        *,
        chunks_by_id: Mapping[str, RagChunk],
        prompt_profile: RagPromptProfile,
    ) -> None:
        force_vllm_inprocess_mode()
        self.config = config
        self._chunks_by_id = chunks_by_id
        self._prompt_profile = prompt_profile
        self._engine = VLLMEngine(config.model, config.kv_dtype, self._engine_config())
        self._scaffold: ScaffoldTokens | None = None

    def start_llm(self) -> None:
        if self._engine.started:
            return
        force_vllm_inprocess_mode()
        try:
            self._prepare()
            self._engine.start(kv_transfer_config=self._kv_transfer_config())
            self._scaffold = extract_rag_scaffold_token_ids(
                self._engine.tokenizer,
                system_prompt=self._prompt_profile.system_prompt,
                block_size=self.config.kv_block_size,
            )
            self._after_start()
        except Exception:
            self.close()
            raise
        logger.info("Started {} engine model={}", self.label, self.config.model)

    def setup_stats(self) -> dict[str, Any]:
        return {}

    def close(self) -> None:
        self._engine.close()
        self._scaffold = None

    def _prepare(self) -> None:
        """Load what the client needs beside the engine; runs once before the engine starts."""

    def _after_start(self) -> None:
        """Runs once the engine and the scaffold exist, before the first answer."""

    def _engine_config(self) -> EngineConfig:
        return self.config.engine_config()

    def _kv_transfer_config(self) -> Mapping[str, Any] | None:
        return None

    def _started_scaffold(self) -> ScaffoldTokens:
        if self._scaffold is None:
            raise RuntimeError(
                f"{type(self).__name__}.start_llm() must be called before answering."
            )
        return self._scaffold

    def _chunk_token_ids(self, chunk_id: str) -> list[int]:
        chunk = self._chunks_by_id.get(chunk_id)
        if chunk is None:
            raise RuntimeError(f"Retrieved chunk id {chunk_id} is not part of the corpus.")
        return chunk.token_ids

    def _query_tokens(self, memory_token_ids: list[int], query: RagQuery) -> QueryTokens:
        return build_rag_query_tokens(
            self._engine.tokenizer,
            memory_token_ids,
            query,
            answer_instruction=self._prompt_profile.answer_instruction,
        )

    def _require_budget(self, memory_token_ids: Sequence[int], query_tokens: QueryTokens) -> None:
        budget = memory_token_budget(
            query_token_count=len(query_tokens.token_ids),
            max_position=self.config.kv_max_position,
            max_model_len=self.config.kv_max_model_len,
            max_answer_tokens=self.config.max_answer_tokens,
        )
        require_memory_within_budget(
            len(memory_token_ids),
            budget,
            top_k=self.config.top_k,
            chunk_size=self.config.chunk_size,
        )

    def _generate(self, prompt: PromptTokens) -> GenerationOutput:
        return self._engine.generate(
            prompt.prompt_token_ids,
            max_tokens=self.config.max_answer_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
