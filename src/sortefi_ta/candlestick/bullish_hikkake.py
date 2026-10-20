"""Bullish Hikkake — inside bar (bar 1 range inside bar 0), followed by a lower low (bar 2),
then price reverses and closes above bar 0's high (bar 3+).

Simplified to 4-bar detection."""
import numpy as np
from . import _helpers as h


def detect(o, hi, lo, c):
    n = len(o)
    if n < 4:
        return h.zeros(n)
    out = h.zeros(n)
    # Bar 0: mother bar, Bar 1: inside bar, Bar 2: breaks below, Bar 3: closes above bar 0 high
    inside = (hi[1:-2] <= hi[:-3]) & (lo[1:-2] >= lo[:-3])
    breaks_low = lo[2:-1] < lo[1:-2]
    reversal = c[3:] > hi[:-3]
    out[3:] = np.where(inside & breaks_low & reversal, 1.0, 0.0)
    return out
