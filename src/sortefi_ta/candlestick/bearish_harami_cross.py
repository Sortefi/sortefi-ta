"""Bearish Harami Cross — large bullish candle followed by a doji inside its body."""
import numpy as np
from . import _helpers as h


def detect(o, hi, lo, c):
    n = len(o)
    if n < 2:
        return h.zeros(n)
    bull = h.is_bullish(o, c)
    bt = h.body_top(o, c)
    bb = h.body_bottom(o, c)
    br = h.body_ratio(o, c, hi, lo)
    rng = h.candle_range(hi, lo)
    is_doji = (rng > 0) & (br < 0.10)
    out = h.zeros(n)
    out[1:] = np.where(
        bull[:-1] & is_doji[1:] & (bt[1:] <= bt[:-1]) & (bb[1:] >= bb[:-1]),
        1.0, 0.0
    )
    return out
