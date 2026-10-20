"""Stick Sandwich — bearish, bullish, bearish where first and third have same close."""
import numpy as np
from . import _helpers as h


def detect(o, hi, lo, c):
    n = len(o)
    if n < 3:
        return h.zeros(n)
    bear = h.is_bearish(o, c)
    bull = h.is_bullish(o, c)
    rng = h.candle_range(hi, lo)
    tol = rng * 0.03
    out = h.zeros(n)
    out[2:] = np.where(
        bear[:-2] & bull[1:-1] & bear[2:] &
        (np.abs(c[2:] - c[:-2]) <= np.maximum(tol[:-2], tol[2:])),
        1.0, 0.0
    )
    return out
