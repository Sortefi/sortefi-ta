"""Falling Three Methods — long bearish, 2-3 small bullish candles within its range,
then bearish closing below the first candle's close.

Uses 5-bar window: [big bear, small, small, small, big bear]."""
import numpy as np
from . import _helpers as h


def detect(o, hi, lo, c):
    n = len(o)
    if n < 5:
        return h.zeros(n)
    bull = h.is_bullish(o, c)
    bear = h.is_bearish(o, c)
    br = h.body_ratio(o, c, hi, lo)
    out = h.zeros(n)
    out[4:] = np.where(
        bear[:-4] & (br[:-4] > 0.50) &
        (br[1:-3] < 0.50) & (hi[1:-3] <= hi[:-4]) & (lo[1:-3] >= lo[:-4]) &
        (br[2:-2] < 0.50) & (hi[2:-2] <= hi[:-4]) & (lo[2:-2] >= lo[:-4]) &
        (br[3:-1] < 0.50) & (hi[3:-1] <= hi[:-4]) & (lo[3:-1] >= lo[:-4]) &
        bear[4:] & (br[4:] > 0.50) & (c[4:] < c[:-4]),
        1.0, 0.0
    )
    return out
