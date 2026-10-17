"""Mem0 glue for the memory benchmark: fact catalogs, caches, and the vector store adapter.

Mem0 reads ``MEM0_DIR`` and ``MEM0_TELEMETRY`` when it is first imported, so
the process-wide defaults are set here, before any submodule imports the SDK.
"""

import os

from benchmarks.common.paths import mem0_dir_from_environment

os.environ.setdefault("MEM0_TELEMETRY", "false")
os.environ.setdefault("MEM0_DIR", str(mem0_dir_from_environment()))
