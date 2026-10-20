"""Bullish Abandoned Baby — bearish candle, doji that gaps down, bullish candle that gaps up."""
import numpy as np
from . import _helpers as h


def detect(o, hi, lo, c):
    n = len(o)
    if n < 3:
        return h.zeros(n)
    bear = h.is_bearish(o, c)
    bull = h.is_bullish(o, c)
    br = h.body_ratio(o, c, hi, lo)
    rng = h.candle_range(hi, lo)
    is_doji = (rng > 0) & (br < 0.10)
    bc = h.body_center(o, c)
    out = h.zeros(n)
    out[2:] = np.where(
        bear[:-2] & (br[:-2] > 0.50) &
        is_doji[1:-1] & (hi[1:-1] < lo[:-2]) &
        bull[2:] & (br[2:] > 0.50) & (lo[2:] > hi[1:-1]),
        1.0, 0.0
    )
    return out
