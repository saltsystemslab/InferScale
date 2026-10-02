"""InferScale: GPU-native KV injection for personalized LLM serving."""

from . import v1
from .v1 import *  # noqa: F401,F403
from .v1 import __all__ as _v1_all

__version__ = "0.1.0"
__all__ = ["__version__", "v1", *_v1_all]
