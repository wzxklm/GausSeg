"""GPU memory cleanup utility shared across pipeline stages."""

from __future__ import annotations

import gc
from contextlib import contextmanager


@contextmanager
def gpu_cleanup():
    """Release Python refs + CUDA cache on exit (also runs on exception)."""
    try:
        yield
    finally:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        gc.collect()
