from __future__ import annotations

import numpy as np


def _cuda_description(torch) -> str:
    device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)
    name = torch.cuda.get_device_name(device)
    return f"{name} (sm_{major}{minor}, torch CUDA {torch.version.cuda})"


def main() -> None:
    import torch
    import jasper

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(128, 64)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    queries = vectors[:2].copy()

    try:
        graph = jasper.Graph.build(
            torch.from_numpy(vectors).to(device="cuda", dtype=torch.float32),
            n_neighbors=32,
            distance="ip",
            workspace_budget="1GB",
        )
    except RuntimeError as exc:
        if "cudaErrorUnsupportedPtxVersion" not in str(exc):
            raise
        raise SystemExit(
            "Jasper failed with cudaErrorUnsupportedPtxVersion on "
            f"{_cuda_description(torch)}. Rebuild Jasper with "
            "JASPER_CUDA_ARCHITECTURES=native, or set it to the exact target "
            "SM value when building away from the GPU. If the rebuilt native "
            "binary still fails, use a CUDA toolkit supported by the installed "
            "NVIDIA driver."
        ) from exc
    indices, distances = graph.search(
        torch.from_numpy(queries).to(device="cuda", dtype=torch.float32),
        k=5,
        beam_width=32,
    )
    print("indices:", indices.detach().cpu().tolist())
    print("distances:", distances.detach().cpu().tolist())
    graph.free()


if __name__ == "__main__":
    main()
