"""Identical Three Crows — three bearish candles where each opens at or near prior close."""
import numpy as np
from . import _helpers as h


def detect(o, hi, lo, c):
    n = len(o)
    if n < 3:
        return h.zeros(n)
    bear = h.is_bearish(o, c)
    br = h.body_ratio(o, c, hi, lo)
    rng = h.candle_range(hi, lo)
    tol = rng * 0.05
    out = h.zeros(n)
    out[2:] = np.where(
        bear[:-2] & bear[1:-1] & bear[2:] &
        (c[1:-1] < c[:-2]) & (c[2:] < c[1:-1]) &
        (np.abs(o[1:-1] - c[:-2]) <= tol[:-2]) &
        (np.abs(o[2:] - c[1:-1]) <= tol[1:-1]) &
        (br[:-2] > 0.50) & (br[1:-1] > 0.50) & (br[2:] > 0.50),
        1.0, 0.0
    )
    return out
