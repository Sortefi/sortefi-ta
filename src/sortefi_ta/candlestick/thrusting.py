"""Thrusting — bearish candle followed by bullish candle that opens below prior low
and closes into the prior body but below its midpoint."""
import numpy as np
from . import _helpers as h


def detect(o, hi, lo, c):
    n = len(o)
    if n < 2:
        return h.zeros(n)
    bear = h.is_bearish(o, c)
    bull = h.is_bullish(o, c)
    mid = h.body_center(o, c)
    bb = h.body_bottom(o, c)
    out = h.zeros(n)
    out[1:] = np.where(
        bear[:-1] & bull[1:] & (o[1:] < lo[:-1]) &
        (c[1:] > bb[:-1]) & (c[1:] < mid[:-1]),
        1.0, 0.0
    )
    return out
