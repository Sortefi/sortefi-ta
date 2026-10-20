"""Advance Block — three bullish candles with progressively smaller bodies and longer upper wicks.
Signals weakening buying pressure."""
import numpy as np
from . import _helpers as h


def detect(o, hi, lo, c):
    n = len(o)
    if n < 3:
        return h.zeros(n)
    bull = h.is_bullish(o, c)
    ab = h.abs_body(o, c)
    uwr = h.upper_wick_ratio(o, c, hi, lo)
    out = h.zeros(n)
    out[2:] = np.where(
        bull[:-2] & bull[1:-1] & bull[2:] &
        (c[1:-1] > c[:-2]) & (c[2:] > c[1:-1]) &
        (ab[1:-1] < ab[:-2]) & (ab[2:] < ab[1:-1]) &
        (uwr[1:-1] > uwr[:-2]) & (uwr[2:] > uwr[1:-1]),
        1.0, 0.0
    )
    return out
