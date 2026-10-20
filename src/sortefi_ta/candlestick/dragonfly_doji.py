"""Dragonfly Doji — doji with long lower wick (>70% range) and tiny upper wick (<15%)."""
import numpy as np
from . import _helpers as h


def detect(o, hi, lo, c):
    rng = h.candle_range(hi, lo)
    br = h.body_ratio(o, c, hi, lo)
    uwr = h.upper_wick_ratio(o, c, hi, lo)
    lwr = h.lower_wick_ratio(o, c, hi, lo)
    return np.where((rng > 0) & (br < 0.10) & (lwr > 0.70) & (uwr < 0.15), 1.0, 0.0)
