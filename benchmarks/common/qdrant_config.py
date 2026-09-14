"""Connection and search settings for the benchmark's Qdrant server."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any
from urllib.parse import urlsplit


@dataclass(slots=True, frozen=True)
class QdrantConfig:
    url: str = "http://127.0.0.1:6333"
    grpc_port: int = 6334
    prefer_grpc: bool = True
    timeout: float = 60.0
    exact: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.url, str):
            raise ValueError("qdrant.url must be an HTTP or HTTPS URL.")
        parsed = urlsplit(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("qdrant.url must be an HTTP or HTTPS URL.")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("qdrant.url must not contain credentials, a query, or a fragment.")
        # Accessing port also rejects malformed and out-of-range URL ports.
        _ = parsed.port
        if type(self.grpc_port) is not int or not 1 <= self.grpc_port <= 65535:
            raise ValueError("qdrant.grpc_port must be an integer between 1 and 65535.")
        if type(self.prefer_grpc) is not bool or type(self.exact) is not bool:
            raise ValueError("qdrant.prefer_grpc and qdrant.exact must be booleans.")
        if (
            isinstance(self.timeout, bool)
            or not isinstance(self.timeout, (int, float))
            or not math.isfinite(self.timeout)
            or self.timeout <= 0
        ):
            raise ValueError("qdrant.timeout must be a positive finite number of seconds.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QdrantConfig":
        if not isinstance(data, Mapping):
            raise ValueError("qdrant must be an object.")
        unknown = sorted(set(data) - {item.name for item in fields(cls)})
        if unknown:
            raise ValueError(f"qdrant has unknown keys: {', '.join(unknown)}.")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
