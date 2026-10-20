"""Marubozu — body covers >90% of the candle range (almost no wicks)."""
import numpy as np
from . import _helpers as h


def detect(o, hi, lo, c):
    rng = h.candle_range(hi, lo)
    br = h.body_ratio(o, c, hi, lo)
    return np.where((rng > 0) & (br > 0.90), 1.0, 0.0)
