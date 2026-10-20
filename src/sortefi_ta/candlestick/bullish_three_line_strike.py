"""Bullish Three Line Strike — three bullish candles (each closing higher) followed by
a long bearish candle that opens above the third close and closes below the first open."""
import numpy as np
from . import _helpers as h


def detect(o, hi, lo, c):
    n = len(o)
    if n < 4:
        return h.zeros(n)
    bull = h.is_bullish(o, c)
    bear = h.is_bearish(o, c)
    br = h.body_ratio(o, c, hi, lo)
    out = h.zeros(n)
    out[3:] = np.where(
        bull[:-3] & bull[1:-2] & bull[2:-1] &
        (c[1:-2] > c[:-3]) & (c[2:-1] > c[1:-2]) &
        (br[:-3] > 0.40) & (br[1:-2] > 0.40) & (br[2:-1] > 0.40) &
        bear[3:] & (o[3:] >= c[2:-1]) & (c[3:] <= o[:-3]),
        1.0, 0.0
    )
    return out
