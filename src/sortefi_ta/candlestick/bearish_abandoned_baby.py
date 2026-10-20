"""Bearish Abandoned Baby — bullish candle, doji that gaps up, bearish candle that gaps down."""
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
    out = h.zeros(n)
    out[2:] = np.where(
        bull[:-2] & (br[:-2] > 0.50) &
        is_doji[1:-1] & (lo[1:-1] > hi[:-2]) &
        bear[2:] & (br[2:] > 0.50) & (hi[2:] < lo[1:-1]),
        1.0, 0.0
    )
    return out
