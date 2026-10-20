"""Bearish Engulfing — bullish candle followed by larger bearish candle that engulfs it."""
import numpy as np
from . import _helpers as h


def detect(o, hi, lo, c):
    n = len(o)
    if n < 2:
        return h.zeros(n)
    bull = h.is_bullish(o, c)
    bear = h.is_bearish(o, c)
    bt = h.body_top(o, c)
    bb = h.body_bottom(o, c)
    out = h.zeros(n)
    out[1:] = np.where(
        bull[:-1] & bear[1:] & (bt[1:] >= bt[:-1]) & (bb[1:] <= bb[:-1]),
        1.0, 0.0
    )
    return out
