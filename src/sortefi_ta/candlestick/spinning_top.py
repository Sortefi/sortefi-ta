"""Spinning Top — small body (<30% range), roughly equal wicks, not a doji."""
import numpy as np
from . import _helpers as h


def detect(o, hi, lo, c):
    rng = h.candle_range(hi, lo)
    safe = rng > 0
    br = h.body_ratio(o, c, hi, lo)
    is_doji = br < 0.10
    uw = h.upper_wick(o, c, hi)
    lw = h.lower_wick(o, c, lo)
    wick_min = np.minimum(uw, lw)
    wick_max = np.maximum(uw, lw)
    wicks_balanced = (wick_max > 0) & (wick_min / np.where(wick_max > 0, wick_max, 1.0) >= 0.4)
    return np.where(safe & ~is_doji & (br < 0.30) & wicks_balanced, 1.0, 0.0)
