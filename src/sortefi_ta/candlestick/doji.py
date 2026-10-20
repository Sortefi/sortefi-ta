"""Doji — open and close are nearly equal (body < 10% of range)."""
import numpy as np
from . import _helpers as h


def detect(o, hi, lo, c):
    rng = h.candle_range(hi, lo)
    br = h.body_ratio(o, c, hi, lo)
    return np.where((rng > 0) & (br < 0.10), 1.0, 0.0)
