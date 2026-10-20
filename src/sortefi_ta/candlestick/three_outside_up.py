"""Three Outside Up — small bearish, bullish engulfing it, then bullish closing higher."""
import numpy as np
from . import _helpers as h


def detect(o, hi, lo, c):
    n = len(o)
    if n < 3:
        return h.zeros(n)
    bear = h.is_bearish(o, c)
    bull = h.is_bullish(o, c)
    bt = h.body_top(o, c)
    bb = h.body_bottom(o, c)
    out = h.zeros(n)
    out[2:] = np.where(
        bear[:-2] &
        bull[1:-1] & (bt[1:-1] >= bt[:-2]) & (bb[1:-1] <= bb[:-2]) &
        bull[2:] & (c[2:] > c[1:-1]),
        1.0, 0.0
    )
    return out
