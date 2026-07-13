from __future__ import annotations

from dataclasses import dataclass, field

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata

from .connector_utils import (  # noqa: F401  (re-exported for the GPU connector)
    align_to_block_size,
    build_slot_mapping,
    extra_config,
    extract_user_id,
)


@dataclass
class MemoryLoadMeta:
    user_id: str
    slot_mapping: torch.Tensor
    num_tokens: int
    # Memory tokens covered by a native prefix-cache hit; the injected KV
    # starts at this offset into the user's stored memory.
    skip_tokens: int = 0
    # Originating vLLM request, so load accounting can be per request rather
    # than per event.
    request_id: str = ""


@dataclass
class MemoryConnectorMetadata(KVConnectorMetadata):
    loads: list[MemoryLoadMeta] = field(default_factory=list)

    def add_load(
        self,
        *,
        user_id: str,
        block_ids: list[int],
        block_size: int,
        num_tokens: int,
        skip_tokens: int = 0,
        request_id: str = "",
    ) -> None:
        slot_mapping = build_slot_mapping(block_ids, block_size, num_tokens)
        self.loads.append(
            MemoryLoadMeta(
                user_id=user_id,
                slot_mapping=torch.tensor(slot_mapping, dtype=torch.long),
                num_tokens=num_tokens,
                skip_tokens=skip_tokens,
                request_id=request_id,
            )
        )
