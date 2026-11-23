import os

from benchmarks.common.paths import mem0_dir_from_environment

os.environ.setdefault("MEM0_TELEMETRY", "false")
os.environ.setdefault("MEM0_DIR", str(mem0_dir_from_environment()))
