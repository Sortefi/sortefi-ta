"""Upside Gap Two Crows — bullish candle, then two bearish candles that gap up.
Second bearish engulfs first bearish but still above the first bullish close."""
import numpy as np
from . import _helpers as h


def detect(o, hi, lo, c):
    n = len(o)
    if n < 3:
        return h.zeros(n)
    bull = h.is_bullish(o, c)
    bear = h.is_bearish(o, c)
    bt = h.body_top(o, c)
    bb = h.body_bottom(o, c)
    out = h.zeros(n)
    out[2:] = np.where(
        bull[:-2] & (h.body_ratio(o, c, hi, lo)[:-2] > 0.50) &
        bear[1:-1] & (bb[1:-1] > c[:-2]) &
        bear[2:] & (o[2:] > o[1:-1]) & (c[2:] < c[1:-1]) & (c[2:] > c[:-2]),
        1.0, 0.0
    )
    return out
