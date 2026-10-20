"""Matching Low — two bearish candles with the same (or nearly same) close price."""
import numpy as np
from . import _helpers as h


def detect(o, hi, lo, c):
    n = len(o)
    if n < 2:
        return h.zeros(n)
    bear = h.is_bearish(o, c)
    rng = h.candle_range(hi, lo)
    tol = rng * 0.03
    out = h.zeros(n)
    out[1:] = np.where(
        bear[:-1] & bear[1:] &
        (np.abs(c[1:] - c[:-1]) <= np.maximum(tol[:-1], tol[1:])),
        1.0, 0.0
    )
    return out
