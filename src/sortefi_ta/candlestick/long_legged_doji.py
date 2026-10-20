"""Long Legged Doji — doji with both upper and lower wicks > 40% of range."""
import numpy as np
from . import _helpers as h


def detect(o, hi, lo, c):
    rng = h.candle_range(hi, lo)
    br = h.body_ratio(o, c, hi, lo)
    uwr = h.upper_wick_ratio(o, c, hi, lo)
    lwr = h.lower_wick_ratio(o, c, hi, lo)
    return np.where((rng > 0) & (br < 0.10) & (uwr > 0.40) & (lwr > 0.40), 1.0, 0.0)
