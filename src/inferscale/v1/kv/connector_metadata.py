from __future__ import annotations

from dataclasses import dataclass, field

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata

from .connector_utils import build_slot_mapping


@dataclass
class MemoryLoadMeta:
    user_id: str
    slot_mapping: torch.Tensor
    num_tokens: int
    skip_tokens: int = 0
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
