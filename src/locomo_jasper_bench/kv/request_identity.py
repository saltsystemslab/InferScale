from __future__ import annotations

from typing import Any

MEMORY_USER_ID_EXTRA_ARG = "locomo_memory_user_id"


def extract_user_id(request: Any, default_user_id: str | None = None) -> str | None:
    user_id = getattr(request, "user", None)
    if user_id:
        return str(user_id)

    sampling_params = getattr(request, "sampling_params", None)
    if sampling_params is not None:
        user_id = getattr(sampling_params, "user", None)
        if user_id:
            return str(user_id)
        extra_args = getattr(sampling_params, "extra_args", None)
        if isinstance(extra_args, dict):
            user_id = extra_args.get(MEMORY_USER_ID_EXTRA_ARG)
            if user_id:
                return str(user_id)

    metadata = getattr(request, "metadata", None)
    if metadata:
        user_id = metadata.get("user_id")
        if user_id:
            return str(user_id)

    if default_user_id:
        return default_user_id
    return None
