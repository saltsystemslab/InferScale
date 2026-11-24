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

from .protocol import (
    ANSWER_BACKENDS,
    KV_STORE_BACKENDS,
    LMCACHE_BACKEND,
    PROMPT_INJECTION_BACKEND,
    VECTOR_BACKENDS,
)

SWEEP_KEYS = ("top_k", "variants")
VARIANT_KEYS = (
    "name",
    "answer_backend",
    "vector_backend",
    "kv_store_backend",
    "kv_staging_slots",
    "context_window",
)
ALL_SAMPLES = "all"


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
    variant: str
    answer_backend: str
    vector_backend: str
    top_k: int
    context_window: int
    kv_store_backend: str = "gpu"
    kv_staging_slots: int | None = None

    def run_id(self, label: str, max_samples: int | None, stamp: str) -> str:
        sample_count = str(max_samples) if max_samples is not None else ALL_SAMPLES
        return (
            f"{label}-{self.variant}-mem0-{self.vector_backend}{sample_count}"
            f"-k{self.top_k}-s{self.context_window}-{stamp}"
        )

    def materialize(self, data: Mapping[str, Any]) -> dict[str, Any]:
        run = deepcopy(dict(data))
        run.update(
            answer_backend=self.answer_backend,
            vector_backend=self.vector_backend,
            top_k=self.top_k,
            context_window=self.context_window,
        )
        kv_settings = run.setdefault("inferscale", {}).setdefault("kv", {})
        kv_settings["store_backend"] = self.kv_store_backend
        if self.kv_staging_slots is not None:
            kv_settings["staging_slots"] = self.kv_staging_slots
        return run


def parse_sweeps(section: Mapping[str, Any], where: str) -> dict[str, MemorySweep]:
    return {
        str(name): _parse_sweep(str(name), authored_sweep, f"{where}.{name}")
        for name, authored_sweep in section.items()
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
    return [
        MemoryCell(
            variant=variant.name,
            answer_backend=variant.answer_backend,
            vector_backend=variant.vector_backend,
            top_k=top_k,
            context_window=context_window,
            kv_store_backend=variant.kv_store_backend,
            kv_staging_slots=variant.kv_staging_slots,
        )
        for top_k in sweep.top_k
        for variant in sweep.variants
        for context_window in variant.context_windows
    ]


def kv_context_windows(sweeps: Iterable[MemorySweep], base_window: int) -> tuple[int, ...]:
    windows = {base_window}
    for sweep in sweeps:
        for variant in sweep.variants:
            if variant.answer_backend == "kv-injection":
                windows.update(variant.context_windows)
    return tuple(sorted(windows))


def _parse_sweep(name: str, authored_sweep: Any, where: str) -> MemorySweep:
    if not isinstance(authored_sweep, dict):
        raise ConfigError(f"{where} must be an object.")
    reject_unknown_keys(authored_sweep, SWEEP_KEYS, where)
    authored_variants = authored_sweep.get("variants")
    if not isinstance(authored_variants, list) or not authored_variants:
        raise ConfigError(f"{where}.variants must be a non-empty list.")
    variants = [
        _parse_variant(authored_variant, f"{where}.variants[{position}]")
        for position, authored_variant in enumerate(authored_variants)
    ]
    _reject_duplicate_variant_names(variants, where)
    top_ks = int_list(authored_sweep, "top_k", where)
    if any(top_k < 1 for top_k in top_ks):
        raise ConfigError(f"{where}.top_k entries must be >= 1.")
    return MemorySweep(name=name, top_k=top_ks, variants=tuple(variants))


def _reject_duplicate_variant_names(variants: list[SweepVariant], where: str) -> None:
    variant_names = [variant.name for variant in variants]
    duplicate_names = sorted({name for name in variant_names if variant_names.count(name) > 1})
    if duplicate_names:
        raise ConfigError(f"{where} has a duplicate variant name {duplicate_names[0]!r}.")


def _parse_variant(authored_variant: Any, where: str) -> SweepVariant:
    if not isinstance(authored_variant, dict):
        raise ConfigError(f"{where} must be an object.")
    reject_unknown_keys(authored_variant, VARIANT_KEYS, where)
    answer_backend = require(authored_variant, "answer_backend", str, where)
    vector_backend = require(authored_variant, "vector_backend", str, where)
    if answer_backend not in ANSWER_BACKENDS or vector_backend not in VECTOR_BACKENDS:
        raise ConfigError(f"{where}: unknown answer or vector backend.")
    kv_store_backend = optional(authored_variant, "kv_store_backend", str, "gpu", where)
    if kv_store_backend not in KV_STORE_BACKENDS:
        raise ConfigError(f"{where}.kv_store_backend must be gpu or cpu.")
    kv_staging_slots = optional(authored_variant, "kv_staging_slots", int, None, where)
    if kv_staging_slots is not None and kv_staging_slots < 1:
        raise ConfigError(f"{where}.kv_staging_slots must be >= 1.")
    context_windows = int_list(authored_variant, "context_window", where, default=(0,))
    if any(context_window < 0 for context_window in context_windows):
        raise ConfigError(f"{where}.context_window entries must be >= 0.")
    if answer_backend == LMCACHE_BACKEND and context_windows != (0,):
        raise ConfigError(
            f"{where}: {LMCACHE_BACKEND} stores each fact alone, so context_window must be [0]."
        )
    if answer_backend == PROMPT_INJECTION_BACKEND and context_windows != (0,):
        raise ConfigError(
            f"{where}: {PROMPT_INJECTION_BACKEND} sends only the retrieved facts, so "
            "context_window must be [0]."
        )
    return SweepVariant(
        name=require(authored_variant, "name", str, where),
        answer_backend=answer_backend,
        vector_backend=vector_backend,
        kv_store_backend=kv_store_backend,
        kv_staging_slots=kv_staging_slots,
        context_windows=context_windows,
    )
