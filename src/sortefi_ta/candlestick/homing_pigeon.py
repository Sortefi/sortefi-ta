"""Homing Pigeon — two bearish candles where the second has a smaller body inside the first."""
import numpy as np
from . import _helpers as h


def detect(o, hi, lo, c):
    n = len(o)
    if n < 2:
        return h.zeros(n)
    bear = h.is_bearish(o, c)
    bt = h.body_top(o, c)
    bb = h.body_bottom(o, c)
    ab = h.abs_body(o, c)
    out = h.zeros(n)
    out[1:] = np.where(
        bear[:-1] & bear[1:] &
        (bt[1:] <= bt[:-1]) & (bb[1:] >= bb[:-1]) &
        (ab[1:] < ab[:-1]),
        1.0, 0.0
    )
    return out
