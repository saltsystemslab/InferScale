"""Sweep grids declared in a memory benchmark JSON file and the run cells they expand to."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from benchmarks.common.json_fields import (
    ConfigError,
    int_list,
    optional,
    reject_unknown_keys,
    require,
)

from .protocol import ANSWER_BACKENDS, KV_STORE_BACKENDS, LMCACHE_BACKEND, VECTOR_BACKENDS

SWEEP_KEYS = ("top_k", "variants")
VARIANT_KEYS = (
    "name",
    "answer_backend",
    "vector_backend",
    "kv_store_backend",
    "kv_staging_slots",
    "context_window",
)


@dataclass(slots=True, frozen=True)
class SweepVariant:
    name: str
    answer_backend: str
    vector_backend: str
    kv_store_backend: str = "gpu"
    kv_staging_slots: int | None = None
    context_windows: tuple[int, ...] = (0,)


@dataclass(slots=True, frozen=True)
class MemorySweep:
    name: str
    top_k: tuple[int, ...]
    variants: tuple[SweepVariant, ...]


def sweeps_use_backend(sweeps: Iterable[MemorySweep], answer_backend: str) -> bool:
    return any(
        variant.answer_backend == answer_backend for sweep in sweeps for variant in sweep.variants
    )


@dataclass(slots=True, frozen=True)
class MemoryCell:
    """One (variant, top_k, context_window) point of a sweep."""

    variant: str
    answer_backend: str
    vector_backend: str
    top_k: int
    context_window: int
    kv_store_backend: str = "gpu"
    kv_staging_slots: int | None = None

    def run_id(self, label: str, max_samples: int | None, stamp: str) -> str:
        samples = str(max_samples) if max_samples is not None else "all"
        return (
            f"{label}-{self.variant}-mem0-{self.vector_backend}{samples}"
            f"-k{self.top_k}-s{self.context_window}-{stamp}"
        )

    def materialize(self, data: Mapping[str, Any]) -> dict[str, Any]:
        """One run's JSON: the grid's file with this cell's axes written in."""
        result = deepcopy(dict(data))
        result.update(
            answer_backend=self.answer_backend,
            vector_backend=self.vector_backend,
            top_k=self.top_k,
            context_window=self.context_window,
        )
        kv = result.setdefault("inferscale", {}).setdefault("kv", {})
        kv["store_backend"] = self.kv_store_backend
        if self.kv_staging_slots is not None:
            kv["staging_slots"] = self.kv_staging_slots
        return result


def parse_sweeps(section: Mapping[str, Any], where: str) -> dict[str, MemorySweep]:
    return {
        str(name): _parse_sweep(str(name), raw, f"{where}.{name}") for name, raw in section.items()
    }


def sweep_dict(sweep: MemorySweep) -> dict[str, Any]:
    return {
        "top_k": list(sweep.top_k),
        "variants": [
            {
                "name": variant.name,
                "answer_backend": variant.answer_backend,
                "vector_backend": variant.vector_backend,
                "kv_store_backend": variant.kv_store_backend,
                "kv_staging_slots": variant.kv_staging_slots,
                "context_window": list(variant.context_windows),
            }
            for variant in sweep.variants
        ],
    }


def sweep_cells(sweep: MemorySweep) -> list[MemoryCell]:
    """Expand a sweep in the order top_k, variant, context_window."""
    return [
        MemoryCell(
            variant=variant.name,
            answer_backend=variant.answer_backend,
            vector_backend=variant.vector_backend,
            top_k=top_k,
            context_window=window,
            kv_store_backend=variant.kv_store_backend,
            kv_staging_slots=variant.kv_staging_slots,
        )
        for top_k in sweep.top_k
        for variant in sweep.variants
        for window in variant.context_windows
    ]


def kv_context_windows(sweeps: Iterable[MemorySweep], base_window: int) -> tuple[int, ...]:
    """Every context window a KV injection variant encodes, for the precompute-kv stage."""
    windows = {base_window}
    for sweep in sweeps:
        for variant in sweep.variants:
            if variant.answer_backend == "kv-injection":
                windows.update(variant.context_windows)
    return tuple(sorted(windows))


def _parse_sweep(name: str, raw: Any, where: str) -> MemorySweep:
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} must be an object.")
    reject_unknown_keys(raw, SWEEP_KEYS, where)
    variants_raw = raw.get("variants")
    if not isinstance(variants_raw, list) or not variants_raw:
        raise ConfigError(f"{where}.variants must be a non-empty list.")
    variants = [
        _parse_variant(item, f"{where}.variants[{index}]")
        for index, item in enumerate(variants_raw)
    ]
    names = [variant.name for variant in variants]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ConfigError(f"{where} has a duplicate variant name {duplicates[0]!r}.")
    top_ks = int_list(raw, "top_k", where)
    if any(top_k < 1 for top_k in top_ks):
        raise ConfigError(f"{where}.top_k entries must be >= 1.")
    return MemorySweep(name=name, top_k=top_ks, variants=tuple(variants))


def _parse_variant(item: Any, where: str) -> SweepVariant:
    if not isinstance(item, dict):
        raise ConfigError(f"{where} must be an object.")
    reject_unknown_keys(item, VARIANT_KEYS, where)
    answer_backend = require(item, "answer_backend", str, where)
    vector_backend = require(item, "vector_backend", str, where)
    if answer_backend not in ANSWER_BACKENDS or vector_backend not in VECTOR_BACKENDS:
        raise ConfigError(f"{where}: unknown answer or vector backend.")
    kv_store_backend = optional(item, "kv_store_backend", str, "gpu", where)
    if kv_store_backend not in KV_STORE_BACKENDS:
        raise ConfigError(f"{where}.kv_store_backend must be gpu or cpu.")
    staging = optional(item, "kv_staging_slots", int, None, where)
    if staging is not None and staging < 1:
        raise ConfigError(f"{where}.kv_staging_slots must be >= 1.")
    windows = int_list(item, "context_window", where, default=(0,))
    if any(window < 0 for window in windows):
        raise ConfigError(f"{where}.context_window entries must be >= 0.")
    if answer_backend == LMCACHE_BACKEND and windows != (0,):
        raise ConfigError(
            f"{where}: {LMCACHE_BACKEND} stores each fact alone, so context_window must be [0]."
        )
    return SweepVariant(
        name=require(item, "name", str, where),
        answer_backend=answer_backend,
        vector_backend=vector_backend,
        kv_store_backend=kv_store_backend,
        kv_staging_slots=staging,
        context_windows=windows,
    )
