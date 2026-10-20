"""Three White Soldiers — three consecutive bullish candles, each closing higher,
each opening within prior body, significant body size."""
import numpy as np
from . import _helpers as h


def detect(o, hi, lo, c):
    n = len(o)
    if n < 3:
        return h.zeros(n)
    bull = h.is_bullish(o, c)
    br = h.body_ratio(o, c, hi, lo)
    bt = h.body_top(o, c)
    bb = h.body_bottom(o, c)
    out = h.zeros(n)
    out[2:] = np.where(
        bull[:-2] & bull[1:-1] & bull[2:] &
        (c[1:-1] > c[:-2]) & (c[2:] > c[1:-1]) &
        (o[1:-1] >= bb[:-2]) & (o[1:-1] <= bt[:-2]) &
        (o[2:] >= bb[1:-1]) & (o[2:] <= bt[1:-1]) &
        (br[:-2] > 0.50) & (br[1:-1] > 0.50) & (br[2:] > 0.50),
        1.0, 0.0
    )
    return out
