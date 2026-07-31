from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from ..data_types import RagDocument, RagPromptProfile, RagQuery

ChunkingPolicy = Literal["token-window", "document"]


@dataclass(slots=True, frozen=True)
class RagDatasetSpec:
    """One evaluable RAG dataset.

    chunking selects how the corpus becomes retrieval units:
      token-window - fixed token-size chunks
      document     - one unit per document, no chunking (planned for MT-RAG)

    prompt_profile carries the dataset's scaffold system prompt and answer
    instruction (including its abstention phrase).
    """

    name: str
    corpus_filename: str
    queries_filename: str
    download_urls: dict[str, str]
    chunking: ChunkingPolicy
    prompt_profile: RagPromptProfile
    load: Callable[[Path], tuple[list[RagDocument], list[RagQuery]]]


def _build_registry() -> dict[str, RagDatasetSpec]:
    from .multihop_rag import MULTIHOP_RAG_SPEC
    from .qasper import QASPER_SPEC

    return {
        MULTIHOP_RAG_SPEC.name: MULTIHOP_RAG_SPEC,
        QASPER_SPEC.name: QASPER_SPEC,
    }


DATASETS: dict[str, RagDatasetSpec] = _build_registry()


def get_dataset(name: str) -> RagDatasetSpec:
    key = name.strip().lower()
    spec = DATASETS.get(key)
    if spec is None:
        known = ", ".join(sorted(DATASETS))
        raise ValueError(f"Unknown RAG dataset {name!r}; known datasets: {known}.")
    return spec
