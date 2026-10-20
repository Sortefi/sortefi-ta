"""Shared utilities for all indicator modules."""
import numpy as np

from sortefi_ta import _ta as ta  # noqa: F401  (the library's own building blocks)


def _nan_array(n: int) -> np.ndarray:
    return np.full(n, np.nan)
