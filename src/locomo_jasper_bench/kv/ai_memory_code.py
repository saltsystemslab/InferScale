from __future__ import annotations

import sys
from pathlib import Path


REQUIRED_VENDOR_FILES = (
    Path("chunked-rope") / "encode_memories_pre_rope.py",
    Path("chunked-rope") / "rope_inject.py",
    Path("memory_connector") / "gpu_memory_store.py",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def vendor_root() -> Path:
    return repo_root() / "src" / "ai-memory-code"


def ensure_ai_memory_paths() -> None:
    """Make the vendored ai-memory-code subset importable."""
    root = vendor_root()
    _require_vendor_files(root)
    paths = [
        root,
        root / "chunked-rope",
    ]
    for path in paths:
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def require_ai_memory_code() -> None:
    ensure_ai_memory_paths()


def _require_vendor_files(root: Path) -> None:
    missing = [path.as_posix() for path in REQUIRED_VENDOR_FILES if not (root / path).exists()]
    if missing:
        raise RuntimeError(
            "The vendored ai-memory-code subset is incomplete under "
            f"{root}. Missing: {', '.join(missing)}."
        )
