"""Evening Star — big bullish, small-bodied candle (center above prior close), big bearish
closing below first candle's midpoint."""
import numpy as np
from . import _helpers as h


def detect(o, hi, lo, c):
    n = len(o)
    if n < 3:
        return h.zeros(n)
    bear = h.is_bearish(o, c)
    bull = h.is_bullish(o, c)
    br = h.body_ratio(o, c, hi, lo)
    bc = h.body_center(o, c)
    mid = h.body_center(o, c)
    out = h.zeros(n)
    out[2:] = np.where(
        bull[:-2] & (br[:-2] > 0.50) &
        (br[1:-1] < 0.30) & (bc[1:-1] > c[:-2]) &
        bear[2:] & (br[2:] > 0.50) & (c[2:] < mid[:-2]),
        1.0, 0.0
    )
    return out
