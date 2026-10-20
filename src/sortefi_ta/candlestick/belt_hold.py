"""Belt Hold — opens at extreme (high or low), large body, no wick on open side.

Bullish: opens at low, closes near high (no lower wick).
Bearish: opens at high, closes near low (no upper wick).
Detected as either variant (1.0 for any belt hold)."""
import numpy as np
from . import _helpers as h


def detect(o, hi, lo, c):
    rng = h.candle_range(hi, lo)
    safe = rng > 0
    br = h.body_ratio(o, c, hi, lo)
    lwr = h.lower_wick_ratio(o, c, hi, lo)
    uwr = h.upper_wick_ratio(o, c, hi, lo)
    bullish = safe & (br > 0.60) & h.is_bullish(o, c) & (lwr < 0.05)
    bearish = safe & (br > 0.60) & h.is_bearish(o, c) & (uwr < 0.05)
    return np.where(bullish | bearish, 1.0, 0.0)
