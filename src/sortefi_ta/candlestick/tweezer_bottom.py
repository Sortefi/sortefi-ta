"""Tweezer Bottom — two candles at bottom with matching lows; first bearish, second bullish."""
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
        bear[:-1] & bull[1:] & (np.abs(lo[1:] - lo[:-1]) <= tol) & (tol > 0),
        1.0, 0.0
    )
    return out
