from __future__ import annotations

import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_ai_memory_paths() -> None:
    """Make the ai-memory-code submodule importable without packaging it."""
    root = repo_root()
    paths = [
        root / "ai-memory-code",
        root / "ai-memory-code" / "chunked-rope",
    ]
    for path in paths:
        text = str(path)
        if path.exists() and text not in sys.path:
            sys.path.insert(0, text)


def require_ai_memory_submodule() -> None:
    if not (repo_root() / "ai-memory-code").exists():
        raise RuntimeError(
            "The ai-memory-code submodule is missing. Run `git submodule update --init --recursive`."
        )
    ensure_ai_memory_paths()

