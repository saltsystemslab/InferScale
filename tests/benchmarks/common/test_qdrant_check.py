from __future__ import annotations

import io
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest
import grpc

from benchmarks.common import qdrant_check
from benchmarks.common.qdrant_config import QdrantConfig


class Response(io.BytesIO):
    status = 200


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    now = [0.0]
    monkeypatch.setattr(qdrant_check.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(qdrant_check.time, "sleep", lambda delay: now.__setitem__(0, now[0] + delay))
    return now


def test_check_authenticates_against_rest_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_API_KEY", "test-secret")
    config = QdrantConfig(url="https://cpu-pod-6333.proxy.runpod.net/", timeout=2)
    requests = []

    def open_request(request, timeout):
        requests.append(request)
        assert timeout <= 2
        return Response(b'{"result":{"collections":[]},"status":"ok"}')

    monkeypatch.setattr(qdrant_check.urllib.request, "build_opener", lambda *args: SimpleNamespace(open=open_request))
    qdrant_check.check_qdrant(config)

    assert len(requests) == 1
    assert requests[0].full_url == "https://cpu-pod-6333.proxy.runpod.net/collections"
    assert requests[0].get_header("Api-key") == "test-secret"
    assert requests[0].get_method() == "GET"


@pytest.mark.parametrize("code", [401, 403, 302])
def test_auth_errors_and_redirects_fail_without_retry_or_secret_output(
    monkeypatch: pytest.MonkeyPatch, clock: list[float], code: int,
) -> None:
    monkeypatch.setenv("QDRANT_API_KEY", "test-secret")

    def fail(request, timeout):
        raise HTTPError(request.full_url, code, "test-secret", {}, None)

    monkeypatch.setattr(qdrant_check.urllib.request, "build_opener", lambda *args: SimpleNamespace(open=fail))
    with pytest.raises(RuntimeError, match="QDRANT_") as error:
        qdrant_check.check_qdrant(QdrantConfig(url="https://example.invalid", prefer_grpc=False))
    assert "test-secret" not in str(error.value)
    assert clock[0] == 0


@pytest.mark.parametrize("body", [b"<html>Not ready</html>", b"null", b'{"status":"ok"}'])
def test_check_rejects_proxy_pages_and_invalid_api_results(
    monkeypatch: pytest.MonkeyPatch, clock: list[float], body: bytes,
) -> None:
    monkeypatch.setattr(
        qdrant_check.urllib.request, "build_opener",
        lambda *args: SimpleNamespace(open=lambda *args, **kwargs: Response(body)),
    )
    with pytest.raises(RuntimeError, match="Cannot reach Qdrant"):
        qdrant_check.check_qdrant(QdrantConfig(url="https://example.invalid", prefer_grpc=False, timeout=2))
    assert clock[0] == 2


def test_check_retries_while_pod_starts(monkeypatch: pytest.MonkeyPatch, clock: list[float]) -> None:
    attempts = []

    def open_request(request, timeout):
        attempts.append(request)
        if len(attempts) == 1:
            raise URLError("not started")
        return Response(b'{"result":{"collections":[]},"status":"ok"}')

    monkeypatch.setattr(qdrant_check.urllib.request, "build_opener", lambda *args: SimpleNamespace(open=open_request))
    qdrant_check.check_qdrant(QdrantConfig(url="https://example.invalid", prefer_grpc=False, timeout=2))
    assert len(attempts) == 2
    assert clock[0] == 1


def test_check_requires_endpoint_before_network_access(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.setattr(
        qdrant_check.urllib.request, "build_opener",
        lambda *args: pytest.fail("Missing endpoint must fail before network access"),
    )
    with pytest.raises(ValueError, match="QDRANT_URL"):
        qdrant_check.check_qdrant(QdrantConfig())


class RpcFailure(grpc.RpcError):
    def __init__(self, code: grpc.StatusCode) -> None:
        super().__init__("test-secret must not appear in error output")
        self._code = code

    def code(self) -> grpc.StatusCode:
        return self._code


@pytest.fixture
def grpc_client(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    client = SimpleNamespace(options=None, calls=[], closed=False, failures=[])

    def list_collections(request, timeout):
        from qdrant_client.grpc import ListCollectionsRequest, ListCollectionsResponse

        assert isinstance(request, ListCollectionsRequest)
        client.calls.append(timeout)
        if client.failures:
            raise RpcFailure(client.failures.pop(0))
        return ListCollectionsResponse()

    def close():
        client.closed = True

    def create(**kwargs):
        client.options = kwargs
        return client

    client.grpc_collections = SimpleNamespace(List=list_collections)
    client.close = close
    monkeypatch.setattr("qdrant_client.QdrantClient", create)
    monkeypatch.setattr(
        qdrant_check.urllib.request, "build_opener",
        lambda *args: pytest.fail("gRPC readiness must not contact REST"),
    )
    return client


def test_check_grpc_uses_mapped_port_credentials_and_closes_client(
    monkeypatch: pytest.MonkeyPatch, clock: list[float], grpc_client: SimpleNamespace,
) -> None:
    monkeypatch.setenv("QDRANT_API_KEY", "test-secret")
    config = QdrantConfig(url="https://qdrant.example", grpc_port=16334, prefer_grpc=True, timeout=2)
    qdrant_check.check_qdrant(config)

    assert grpc_client.options == config.client_kwargs()
    assert grpc_client.options["prefer_grpc"] is True
    assert grpc_client.options["grpc_port"] == 16334
    assert grpc_client.options["api_key"] == "test-secret"
    assert grpc_client.options["check_compatibility"] is False
    assert grpc_client.calls == [2.0]
    assert grpc_client.closed


@pytest.mark.parametrize("code", [grpc.StatusCode.UNAUTHENTICATED, grpc.StatusCode.PERMISSION_DENIED])
def test_check_grpc_auth_failure_is_immediate_and_secret_safe(
    clock: list[float], grpc_client: SimpleNamespace, code: grpc.StatusCode,
) -> None:
    grpc_client.failures = [code]
    with pytest.raises(RuntimeError, match="QDRANT_API_KEY") as error:
        qdrant_check.check_qdrant(QdrantConfig(url="https://qdrant.example", prefer_grpc=True))
    assert "test-secret" not in str(error.value)
    assert len(grpc_client.calls) == 1
    assert clock[0] == 0
    assert grpc_client.closed


def test_check_grpc_retries_startup_and_respects_deadline(
    clock: list[float], grpc_client: SimpleNamespace,
) -> None:
    grpc_client.failures = [grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED]
    with pytest.raises(RuntimeError, match="external TCP port mapped to 6334"):
        qdrant_check.check_qdrant(QdrantConfig(url="https://qdrant.example", prefer_grpc=True, timeout=2))
    assert grpc_client.calls == [2.0, 1.0]
    assert clock[0] == 2
    assert grpc_client.closed


def test_check_grpc_succeeds_when_pod_becomes_ready(
    clock: list[float], grpc_client: SimpleNamespace,
) -> None:
    grpc_client.failures = [grpc.StatusCode.UNAVAILABLE]
    qdrant_check.check_qdrant(QdrantConfig(url="https://qdrant.example", prefer_grpc=True, timeout=2))
    assert grpc_client.calls == [2.0, 1.0]
    assert grpc_client.closed
