# SPDX-License-Identifier: Apache-2.0
"""
GPU-Resident Memory Store for the MemoryKVConnector.

Holds pre-encoded KV cache tensors on GPU, indexed by user_id.
Each user's memory is a list of (key, value) tensors per layer,
representing the KV cache produced by encoding their memory text
through the model.

Usage:
    store = GPUMemoryStore(num_layers=32, device="cuda:0")
    store.add_user_memory("user_123", kv_pairs, num_tokens=512)
    kv, n = store.get_user_memory("user_123")
"""

import os
import threading
from dataclasses import dataclass

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class UserMemory:
    """Pre-encoded KV cache for a single user's memories."""

    # Per-layer KV tensors.
    # Each element is a tensor of shape [2, num_tokens, ...] for standard
    # attention, or [num_tokens, ...] for MLA.
    # The "2" dimension is [key, value].
    # The exact trailing dimensions depend on the attention backend
    # (e.g., num_kv_heads * head_dim for standard, or kv_lora_rank for MLA).
    kv_by_layer: dict[str, torch.Tensor]

    # Number of valid tokens in the KV cache
    num_tokens: int

    # The token IDs that produced this KV cache.
    # These must be prepended to the prompt so the scheduler sees them
    # as part of the request, and the connector can report them as
    # externally cached (skipping prefill).
    token_ids: list[int] | None = None

    # Original memory text (for debugging / logging)
    memory_text: str = ""


class GPUMemoryStore:
    """
    Thread-safe GPU-resident store for pre-encoded user memory KV caches.

    The store holds KV tensors on GPU, organized by user_id and layer name.
    These tensors are in the same format that vLLM's attention backends
    expect, so they can be directly scatter-copied into the paged KV cache
    during start_load_kv().

    The store does NOT manage vLLM block allocation — that's handled by the
    scheduler side of the connector. This store only holds the raw KV data
    that gets injected into allocated blocks.
    """

    def __init__(self, device: str = "cuda:0"):
        self._memories: dict[str, UserMemory] = {}
        self._device = device
        self._lock = threading.Lock()

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def add_user_memory(
        self,
        user_id: str,
        kv_by_layer: dict[str, torch.Tensor],
        num_tokens: int,
        token_ids: list[int] | None = None,
        memory_text: str = "",
    ) -> None:
        """
        Store pre-encoded KV cache for a user.

        Args:
            user_id: Unique user identifier.
            kv_by_layer: Dict mapping layer_name -> KV tensor on GPU.
            num_tokens: Number of tokens in the encoded memory.
            token_ids: The token IDs that produced this KV cache.
                Must be prepended to prompts for this user.
            memory_text: Original memory text (for debugging).
        """
        with self._lock:
            device_kv = {}
            for layer_name, tensor in kv_by_layer.items():
                if tensor.device != torch.device(self._device):
                    device_kv[layer_name] = tensor.to(self._device).contiguous()
                else:
                    device_kv[layer_name] = tensor.contiguous()

            self._memories[user_id] = UserMemory(
                kv_by_layer=device_kv,
                num_tokens=num_tokens,
                token_ids=token_ids,
                memory_text=memory_text,
            )
            logger.info(
                "Stored memory for user %s: %d tokens, %d layers",
                user_id,
                num_tokens,
                len(device_kv),
            )

    def get_user_memory(self, user_id: str) -> UserMemory | None:
        """
        Retrieve pre-encoded KV cache for a user.

        Returns:
            UserMemory if found, None otherwise.
        """
        with self._lock:
            return self._memories.get(user_id)

    def has_user_memory(self, user_id: str) -> bool:
        """Check if a user has stored memory."""
        with self._lock:
            return user_id in self._memories

    def remove_user_memory(self, user_id: str) -> bool:
        """
        Remove a user's memory and free GPU tensors.

        Returns:
            True if memory was found and removed.
        """
        with self._lock:
            if user_id in self._memories:
                mem = self._memories.pop(user_id)
                # Explicitly delete tensors to free GPU memory
                del mem.kv_by_layer
                logger.info("Removed memory for user %s", user_id)
                return True
            return False

    def update_user_memory(
        self,
        user_id: str,
        kv_by_layer: dict[str, torch.Tensor],
        num_tokens: int,
        memory_text: str = "",
    ) -> None:
        """
        Update a user's memory (remove old, add new).
        """
        self.remove_user_memory(user_id)
        self.add_user_memory(user_id, kv_by_layer, num_tokens, memory_text)

    def get_all_user_ids(self) -> list[str]:
        """Get list of all user IDs with stored memories."""
        with self._lock:
            return list(self._memories.keys())

    def get_stats(self) -> dict:
        """Get memory store statistics."""
        with self._lock:
            total_tokens = sum(m.num_tokens for m in self._memories.values())
            total_bytes = 0
            for mem in self._memories.values():
                for tensor in mem.kv_by_layer.values():
                    total_bytes += tensor.nbytes
            return {
                "num_users": len(self._memories),
                "total_tokens": total_tokens,
                "total_gpu_mb": total_bytes / (1024 * 1024),
            }

    # ──────────────────────────────────────────────
    # Disk persistence (safetensors)
    # ──────────────────────────────────────────────

    def save_user_to_disk(self, user_id: str, directory: str) -> str:
        """
        Save a single user's memory KV to disk as a safetensors file.

        File layout:
            {directory}/{user_id}/kv_cache.safetensors
            {directory}/{user_id}/metadata.json

        The safetensors file contains all layers keyed by layer name.
        The metadata file stores num_tokens and memory_text.

        Returns:
            Path to the user's directory.
        """
        import json
        import safetensors.torch

        mem = self.get_user_memory(user_id)
        if mem is None:
            raise ValueError(f"No memory found for user {user_id}")

        user_dir = os.path.join(directory, user_id)
        os.makedirs(user_dir, exist_ok=True)

        # Save KV tensors — safetensors requires CPU tensors
        cpu_tensors = {
            layer_name: tensor.cpu().contiguous()
            for layer_name, tensor in mem.kv_by_layer.items()
        }
        safetensors_path = os.path.join(user_dir, "kv_cache.safetensors")
        safetensors.torch.save_file(cpu_tensors, safetensors_path)

        # Save metadata
        metadata = {
            "user_id": user_id,
            "num_tokens": mem.num_tokens,
            "memory_text": mem.memory_text,
            "num_layers": len(mem.kv_by_layer),
            "layer_names": list(mem.kv_by_layer.keys()),
            "token_ids": mem.token_ids,
        }
        # Add shape info from first layer for quick inspection
        first_tensor = next(iter(mem.kv_by_layer.values()))
        metadata["kv_shape"] = list(first_tensor.shape)
        metadata["kv_dtype"] = str(first_tensor.dtype)

        metadata_path = os.path.join(user_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            "Saved memory for user %s to %s (%d tokens, %d layers)",
            user_id, user_dir, mem.num_tokens, len(mem.kv_by_layer),
        )
        return user_dir

    def load_user_from_disk(
        self, user_id: str, directory: str, device: str | None = None,
    ) -> None:
        """
        Load a single user's memory KV from disk into GPU.

        Args:
            user_id: User ID (must match a subdirectory name in directory).
            directory: Root directory containing user subdirectories.
            device: Target device (defaults to self._device).
        """
        import json
        import safetensors.torch

        device = device or self._device
        user_dir = os.path.join(directory, user_id)

        # Load metadata
        metadata_path = os.path.join(user_dir, "metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        num_tokens = metadata["num_tokens"]
        memory_text = metadata.get("memory_text", "")
        token_ids = metadata.get("token_ids", None)

        # Load KV tensors and move to GPU
        safetensors_path = os.path.join(user_dir, "kv_cache.safetensors")
        cpu_tensors = safetensors.torch.load_file(safetensors_path)
        gpu_tensors = {
            name: tensor.to(device).contiguous()
            for name, tensor in cpu_tensors.items()
        }

        self.add_user_memory(
            user_id=user_id,
            kv_by_layer=gpu_tensors,
            num_tokens=num_tokens,
            token_ids=token_ids,
            memory_text=memory_text,
        )

    def save_all_to_disk(self, directory: str) -> int:
        """
        Save all user memories to disk.

        Returns:
            Number of users saved.
        """
        os.makedirs(directory, exist_ok=True)
        user_ids = self.get_all_user_ids()
        for user_id in user_ids:
            self.save_user_to_disk(user_id, directory)
        logger.info("Saved %d users to %s", len(user_ids), directory)
        return len(user_ids)

    def load_all_from_disk(
        self, directory: str, device: str | None = None, max_users: int | None = None,
    ) -> int:
        """
        Load user memories from disk.

        Scans directory for subdirectories containing metadata.json
        and kv_cache.safetensors files.

        Args:
            directory: Root directory containing user subdirectories.
            device: Target device (defaults to self._device).
            max_users: Maximum number of users to load. None = load all.

        Returns:
            Number of users loaded.
        """
        if not os.path.isdir(directory):
            logger.warning("Memory directory %s does not exist", directory)
            return 0

        loaded = 0
        for entry in sorted(os.listdir(directory)):
            if max_users is not None and loaded >= max_users:
                break

            user_dir = os.path.join(directory, entry)
            metadata_path = os.path.join(user_dir, "metadata.json")
            safetensors_path = os.path.join(user_dir, "kv_cache.safetensors")

            if os.path.isfile(metadata_path) and os.path.isfile(safetensors_path):
                try:
                    self.load_user_from_disk(entry, directory, device)
                    loaded += 1
                except Exception as e:
                    logger.error("Failed to load memory for user %s: %s", entry, e)

        logger.info(
            "Loaded %d users from %s (total GPU: %.1f MB)",
            loaded, directory, self.get_stats()["total_gpu_mb"],
        )
        return loaded
