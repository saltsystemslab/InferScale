"""Expand JSON launch plans into runs, save them, and execute each run as a job.

scripts/launch.sh imports ``launch`` from this package.
"""

from benchmarks.common.launcher.execute import launch

__all__ = ["launch"]
