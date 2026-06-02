from __future__ import annotations

import numpy as np


def main() -> None:
    import torch
    import jasper

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(128, 64)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    queries = vectors[:2].copy()

    graph = jasper.Graph.build(
        torch.from_numpy(vectors).to(device="cuda", dtype=torch.float32),
        n_neighbors=32,
        distance="ip",
        workspace_budget="1GB",
    )
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
