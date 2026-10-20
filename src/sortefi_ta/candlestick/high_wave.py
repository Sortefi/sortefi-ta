"""High Wave — very long upper and lower wicks with a small body (not a doji).

Similar to long legged doji but body is present (10-30% of range)."""
import numpy as np
from . import _helpers as h


def detect(o, hi, lo, c):
    rng = h.candle_range(hi, lo)
    safe = rng > 0
    br = h.body_ratio(o, c, hi, lo)
    uwr = h.upper_wick_ratio(o, c, hi, lo)
    lwr = h.lower_wick_ratio(o, c, hi, lo)
    has_body = br >= 0.10
    small_body = br < 0.30
    long_wicks = (uwr > 0.30) & (lwr > 0.30)
    return np.where(safe & has_body & small_body & long_wicks, 1.0, 0.0)
