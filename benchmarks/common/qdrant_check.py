"""Check access to the configured Qdrant server without changing its data."""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from .config import load_runtime_config
from .qdrant_config import QdrantConfig


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward the API key to a redirect destination.
        return None


def check_qdrant(config: QdrantConfig) -> None:
    """Wait up to the configured timeout for an authenticated collections read."""
    connection = config.client_kwargs()
    if config.prefer_grpc:
        _check_grpc(config, connection)
        return
    headers = {"Accept": "application/json"}
    if connection["api_key"]:
        headers["api-key"] = connection["api_key"]
    request = urllib.request.Request(
        f"{config.url.rstrip('/')}/collections", headers=headers,
    )
    opener = urllib.request.build_opener(_NoRedirect())
    deadline = time.monotonic() + config.timeout
    last_error = "no response"
    while (remaining := deadline - time.monotonic()) > 0:
        try:
            with opener.open(request, timeout=min(10.0, remaining)) as response:
                body = json.load(response)
                if (
                    response.status == 200
                    and isinstance(body, dict)
                    and body.get("status") == "ok"
                    and isinstance(body.get("result"), dict)
                    and isinstance(body["result"].get("collections"), list)
                ):
                    return
                last_error = "response was not a Qdrant collections result"
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403}:
                raise RuntimeError(
                    f"Qdrant rejected authentication (HTTP {exc.code}). "
                    "Set QDRANT_API_KEY to the CPU Pod's QDRANT__SERVICE__API_KEY."
                ) from None
            if 300 <= exc.code < 400:
                raise RuntimeError(
                    "Qdrant redirected the request. Set QDRANT_URL to the direct HTTPS "
                    "API endpoint, not the Runpod console or dashboard URL."
                ) from None
            last_error = f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError, OSError):
            last_error = "network connection failed"
        except (ValueError, UnicodeError):
            last_error = "response was not valid Qdrant JSON"
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    raise RuntimeError(
        f"Cannot reach Qdrant at {config.url} within {config.timeout:g} seconds: "
        f"{last_error}. Check the Runpod CPU Pod logs and HTTP port 6333."
    )


def _check_grpc(config: QdrantConfig, connection: dict[str, Any]) -> None:
    try:
        import grpc
        from qdrant_client import QdrantClient
        from qdrant_client.grpc import ListCollectionsRequest
    except ImportError:
        raise RuntimeError(
            "Install the project's Python dependencies before checking gRPC "
            "(qdrant-client is required)."
        ) from None

    deadline = time.monotonic() + config.timeout
    last_error = "no response"
    client = QdrantClient(**connection)
    try:
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                # Use the same authenticated channel as the benchmark, with a
                # per-attempt deadline. No REST endpoint is needed for this RPC.
                client.grpc_collections.List(
                    ListCollectionsRequest(), timeout=min(10.0, remaining),
                )
                return
            except grpc.RpcError as exc:
                code = exc.code()
                if code in {grpc.StatusCode.UNAUTHENTICATED, grpc.StatusCode.PERMISSION_DENIED}:
                    raise RuntimeError(
                        f"Qdrant rejected authentication (gRPC {code.name}). "
                        "Set QDRANT_API_KEY to the CPU Pod's QDRANT__SERVICE__API_KEY."
                    ) from None
                last_error = f"gRPC {code.name}"
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    finally:
        client.close()
    raise RuntimeError(
        f"Cannot reach Qdrant gRPC at {urlsplit(config.url).hostname}:{config.grpc_port} "
        f"within {config.timeout:g} seconds: {last_error}. "
        "Check the Runpod CPU Pod, the external TCP port mapped to 6334, "
        "and the server's TLS certificate and URL scheme."
    )


def main() -> int:
    try:
        runtime = load_runtime_config(os.environ.get("INFERSCALE_RUNTIME_CONFIG"))
        check_qdrant(runtime.qdrant)
    except (ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    config = runtime.qdrant
    if config.prefer_grpc:
        print(f"Qdrant is ready at {urlsplit(config.url).hostname}:{config.grpc_port} (gRPC).")
    else:
        print(f"Qdrant is ready at {config.url} (REST API).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
