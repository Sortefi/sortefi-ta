"""Bullish Tri-Star — three consecutive dojis, middle one gaps down from first."""
import numpy as np
from . import _helpers as h


def detect(o, hi, lo, c):
    n = len(o)
    if n < 3:
        return h.zeros(n)
    br = h.body_ratio(o, c, hi, lo)
    rng = h.candle_range(hi, lo)
    is_doji = (rng > 0) & (br < 0.10)
    bc = h.body_center(o, c)
    out = h.zeros(n)
    out[2:] = np.where(
        is_doji[:-2] & is_doji[1:-1] & is_doji[2:] &
        (bc[1:-1] < bc[:-2]) & (bc[2:] > bc[1:-1]),
        1.0, 0.0
    )
    return out
