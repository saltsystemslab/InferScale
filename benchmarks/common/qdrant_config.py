"""Connection and search settings for the benchmark's Qdrant server."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from typing import Any
from urllib.parse import urlsplit


@dataclass(slots=True, frozen=True)
class QdrantConfig:
    url: str = field(default_factory=lambda: os.environ.get("QDRANT_URL", ""))
    grpc_port: int = 6334
    prefer_grpc: bool = False
    timeout: float = 60.0
    exact: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.url, str):
            raise ValueError("qdrant.url must be an HTTP or HTTPS URL.")
        if self.url:
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

    def client_kwargs(self) -> dict[str, Any]:
        """Resolve connection credentials without retaining them in run snapshots."""
        if not self.url:
            endpoint_guidance = (
                "direct host URL (https://YOUR_QDRANT_HOST for TLS) and qdrant.grpc_port "
                "to the public TCP port mapped to container port 6334"
                if self.prefer_grpc else
                "HTTPS REST endpoint (https://POD_ID-6333.proxy.runpod.net)"
            )
            raise ValueError(
                "Qdrant URL is not configured. Set QDRANT_URL to your Runpod CPU Pod's "
                f"{endpoint_guidance}, or set qdrant.url in the runtime configuration."
            )
        hostname = (urlsplit(self.url).hostname or "").rstrip(".").lower()
        if self.prefer_grpc and (
            hostname == "proxy.runpod.net" or hostname.endswith(".proxy.runpod.net")
        ):
            raise ValueError(
                "Runpod's HTTPS proxy does not support Qdrant gRPC. Set QDRANT_URL "
                "to the CPU Pod's direct host URL and qdrant.grpc_port to the public TCP "
                "port mapped to container port 6334. Use qdrant.prefer_grpc=false "
                "only when connecting through the HTTPS REST proxy."
            )
        connection = {
            "url": self.url,
            # Preserve an explicit REST port or the scheme's default port.
            "port": None,
            "grpc_port": self.grpc_port,
            "prefer_grpc": self.prefer_grpc,
            "timeout": self.timeout,
            "api_key": os.environ.get("QDRANT_API_KEY") or None,
        }
        if self.prefer_grpc:
            # The client's compatibility probe uses HTTP even for gRPC-only hosts.
            connection["check_compatibility"] = False
        return connection

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
