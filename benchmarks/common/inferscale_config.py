"""Build the library's ``InferScaleConfig`` from a benchmark JSON ``inferscale`` section."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from benchmarks.common.json_fields import ConfigError, section_of, validate_dataclass_types

TOP_LEVEL_KEYS = ("model", "top_k", "context_window")


def inferscale_section(
    data: Mapping[str, Any],
    *,
    model: str,
    top_k: int,
    context_window: int,
    where: str,
) -> Any:
    """Validate the section and fill in the model, retrieval depth, and window from the run."""
    from inferscale.v1 import InferScaleConfig

    section = dict(section_of(data, "inferscale", where))
    if "vector_backend" in section:
        raise ConfigError(
            f"{where}.inferscale.vector_backend is supported only by the standalone InferScale "
            "API; benchmarks select their retrieval backends separately. Remove it from the "
            "section."
        )
    for key in TOP_LEVEL_KEYS:
        if key in section:
            raise ConfigError(
                f"{where}.inferscale.{key} is set from the top level; remove it from the section."
            )
    section["model"] = model
    section["top_k"] = top_k
    section["context_window"] = context_window
    # Benchmarks pick their retrieval backend elsewhere; the library config still
    # validates the Jasper tuning fields.
    section["vector_backend"] = "jasper"
    try:
        validate_dataclass_types(section, InferScaleConfig, f"{where}.inferscale")
        return InferScaleConfig.from_dict(section)
    except ValueError as exc:
        raise ConfigError(f"{where}.inferscale: {exc}") from exc
