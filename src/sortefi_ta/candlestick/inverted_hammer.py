"""Inverted Hammer / Shooting Star — small body in lower third, long upper wick, little lower wick."""
import numpy as np
from . import _helpers as h


def detect(o, hi, lo, c):
    rng = h.candle_range(hi, lo)
    safe = rng > 0
    br = h.body_ratio(o, c, hi, lo)
    uwr = h.upper_wick_ratio(o, c, hi, lo)
    lwr = h.lower_wick_ratio(o, c, hi, lo)
    body_in_bottom = (hi - h.body_top(o, c)) > (rng * 0.60)
    return np.where(safe & (br < 0.33) & (uwr >= 0.60) & (lwr < 0.10) & body_in_bottom, 1.0, 0.0)
