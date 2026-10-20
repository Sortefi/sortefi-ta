"""Ladder Bottom — three consecutive bearish candles (each closing lower), then a bearish/small candle
with a long upper wick, then a bullish candle closing above the prior open."""
import numpy as np
from . import _helpers as h


def detect(o, hi, lo, c):
    n = len(o)
    if n < 5:
        return h.zeros(n)
    bear = h.is_bearish(o, c)
    bull = h.is_bullish(o, c)
    br = h.body_ratio(o, c, hi, lo)
    uwr = h.upper_wick_ratio(o, c, hi, lo)
    out = h.zeros(n)
    out[4:] = np.where(
        bear[:-4] & bear[1:-3] & bear[2:-2] &
        (c[1:-3] < c[:-4]) & (c[2:-2] < c[1:-3]) &
        (br[:-4] > 0.40) & (br[1:-3] > 0.40) & (br[2:-2] > 0.40) &
        (uwr[3:-1] > 0.40) &
        bull[4:] & (c[4:] > o[3:-1]),
        1.0, 0.0
    )
    return out
