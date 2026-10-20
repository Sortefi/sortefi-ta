"""Bearish Hikkake — inside bar (bar 1 range inside bar 0), followed by a higher high (bar 2),
then price reverses and closes below bar 0's low (bar 3+).

Simplified to 4-bar detection."""
import numpy as np
from . import _helpers as h


def detect(o, hi, lo, c):
    n = len(o)
    if n < 4:
        return h.zeros(n)
    out = h.zeros(n)
    inside = (hi[1:-2] <= hi[:-3]) & (lo[1:-2] >= lo[:-3])
    breaks_high = hi[2:-1] > hi[1:-2]
    reversal = c[3:] < lo[:-3]
    out[3:] = np.where(inside & breaks_high & reversal, 1.0, 0.0)
    return out
