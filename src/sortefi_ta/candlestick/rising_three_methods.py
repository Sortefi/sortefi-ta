"""Rising Three Methods — long bullish, 2-3 small bearish candles within its range,
then bullish closing above the first candle's close.

Uses 5-bar window: [big bull, small, small, small, big bull]."""
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
    # Bars: 0=first bull, 1-3=small candles within range, 4=final bull
    out[4:] = np.where(
        bull[:-4] & (br[:-4] > 0.50) &
        (br[1:-3] < 0.50) & (hi[1:-3] <= hi[:-4]) & (lo[1:-3] >= lo[:-4]) &
        (br[2:-2] < 0.50) & (hi[2:-2] <= hi[:-4]) & (lo[2:-2] >= lo[:-4]) &
        (br[3:-1] < 0.50) & (hi[3:-1] <= hi[:-4]) & (lo[3:-1] >= lo[:-4]) &
        bull[4:] & (br[4:] > 0.50) & (c[4:] > c[:-4]),
        1.0, 0.0
    )
    return out
