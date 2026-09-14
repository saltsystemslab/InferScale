from __future__ import annotations

import os

import pytest

from benchmarks.common.qdrant_config import QdrantConfig
from benchmarks.common.vector_types import VectorStoreConfig


@pytest.fixture(params=[False, True], ids=["http", "grpc"])
def qdrant_server_config(request: pytest.FixtureRequest) -> VectorStoreConfig:
    url_variable = "QDRANT_TEST_GRPC_URL" if request.param else "QDRANT_TEST_URL"
    url = os.environ.get(url_variable)
    if not url:
        pytest.skip(f"Set {url_variable} to run Qdrant server integration tests.")
    return VectorStoreConfig(
        backend="qdrant",
        qdrant=QdrantConfig(
            url=url,
            grpc_port=int(os.environ.get("QDRANT_TEST_GRPC_PORT", "6334")) if request.param else 6334,
            prefer_grpc=request.param,
        ),
    )
