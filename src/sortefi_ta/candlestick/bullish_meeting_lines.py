"""Bullish Meeting Lines — bearish candle followed by bullish candle that opens lower
but closes at the same level as the prior close."""
import numpy as np
from . import _helpers as h


def detect(o, hi, lo, c):
    n = len(o)
    if n < 2:
        return h.zeros(n)
    bear = h.is_bearish(o, c)
    bull = h.is_bullish(o, c)
    rng = h.candle_range(hi, lo)
    tol = rng * 0.05
    out = h.zeros(n)
    out[1:] = np.where(
        bear[:-1] & bull[1:] & (o[1:] < c[:-1]) &
        (np.abs(c[1:] - c[:-1]) <= np.maximum(tol[:-1], tol[1:])),
        1.0, 0.0
    )
    return out
