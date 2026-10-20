"""Dark Cloud Cover — bullish candle followed by bearish that opens above prior high
and closes below the midpoint of the prior body."""
import numpy as np
from . import _helpers as h


def detect(o, hi, lo, c):
    n = len(o)
    if n < 2:
        return h.zeros(n)
    bull = h.is_bullish(o, c)
    bear = h.is_bearish(o, c)
    mid = h.body_center(o, c)
    out = h.zeros(n)
    out[1:] = np.where(
        bull[:-1] & bear[1:] & (o[1:] > hi[:-1]) & (c[1:] < mid[:-1]),
        1.0, 0.0
    )
    return out
