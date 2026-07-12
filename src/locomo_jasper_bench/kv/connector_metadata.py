from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata

from .request_identity import extract_user_id


@dataclass
class MemoryLoadMeta:
    user_id: str
    slot_mapping: torch.Tensor
    num_tokens: int


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
    ) -> None:
        block_ids_tensor = torch.tensor(block_ids, dtype=torch.long)
        block_offsets = torch.arange(0, block_size, dtype=torch.long)
        slot_mapping = (
            block_offsets.reshape(1, block_size)
            + block_ids_tensor.reshape(block_ids_tensor.shape[0], 1) * block_size
        )
        self.loads.append(
            MemoryLoadMeta(
                user_id=user_id,
                slot_mapping=slot_mapping.flatten()[:num_tokens],
                num_tokens=num_tokens,
            )
        )


def align_to_block_size(num_tokens: int, block_size: int) -> int:
    return (num_tokens // block_size) * block_size


def extra_config(kv_transfer_config: Any, key: str, default: Any = None) -> Any:
    getter = getattr(kv_transfer_config, "get_from_extra_config", None)
    if callable(getter):
        return getter(key, default)
    extra = getattr(kv_transfer_config, "kv_connector_extra_config", None) or {}
    return extra.get(key, default)
