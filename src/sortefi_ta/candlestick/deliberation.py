"""Deliberation — two strong bullish candles followed by a small bullish candle (spinning top / small body)
that opens near or at the prior close. Signals indecision after uptrend."""
import numpy as np
from . import _helpers as h


def detect(o, hi, lo, c):
    n = len(o)
    if n < 3:
        return h.zeros(n)
    bull = h.is_bullish(o, c)
    br = h.body_ratio(o, c, hi, lo)
    rng = h.candle_range(hi, lo)
    tol = rng * 0.05
    out = h.zeros(n)
    out[2:] = np.where(
        bull[:-2] & (br[:-2] > 0.50) &
        bull[1:-1] & (br[1:-1] > 0.50) &
        (c[1:-1] > c[:-2]) &
        bull[2:] & (br[2:] < 0.30) &
        (np.abs(o[2:] - c[1:-1]) <= tol[1:-1]),
        1.0, 0.0
    )
    return out
