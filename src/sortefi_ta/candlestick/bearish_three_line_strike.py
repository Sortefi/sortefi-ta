"""Bearish Three Line Strike — three bearish candles (each closing lower) followed by
a long bullish candle that opens below the third close and closes above the first open."""
import numpy as np
from . import _helpers as h


def detect(o, hi, lo, c):
    n = len(o)
    if n < 4:
        return h.zeros(n)
    bull = h.is_bullish(o, c)
    bear = h.is_bearish(o, c)
    br = h.body_ratio(o, c, hi, lo)
    out = h.zeros(n)
    out[3:] = np.where(
        bear[:-3] & bear[1:-2] & bear[2:-1] &
        (c[1:-2] < c[:-3]) & (c[2:-1] < c[1:-2]) &
        (br[:-3] > 0.40) & (br[1:-2] > 0.40) & (br[2:-1] > 0.40) &
        bull[3:] & (o[3:] <= c[2:-1]) & (c[3:] >= o[:-3]),
        1.0, 0.0
    )
    return out
