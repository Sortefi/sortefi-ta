"""Tweezer Top — two candles at top with matching highs; first bullish, second bearish."""
import numpy as np
from . import _helpers as h


def detect(o, hi, lo, c):
    n = len(o)
    if n < 2:
        return h.zeros(n)
    bull = h.is_bullish(o, c)
    bear = h.is_bearish(o, c)
    rng = h.candle_range(hi, lo)
    tol = np.maximum(rng[:-1], rng[1:]) * 0.05
    out = h.zeros(n)
    out[1:] = np.where(
        bull[:-1] & bear[1:] & (np.abs(hi[1:] - hi[:-1]) <= tol) & (tol > 0),
        1.0, 0.0
    )
    return out
