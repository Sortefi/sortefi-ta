"""Bullish Kicking — bearish marubozu followed by bullish marubozu that gaps up."""
import numpy as np
from . import _helpers as h


def detect(o, hi, lo, c):
    n = len(o)
    if n < 2:
        return h.zeros(n)
    br = h.body_ratio(o, c, hi, lo)
    bull = h.is_bullish(o, c)
    bear = h.is_bearish(o, c)
    maru = br > 0.80
    out = h.zeros(n)
    out[1:] = np.where(
        bear[:-1] & maru[:-1] & bull[1:] & maru[1:] & (o[1:] > o[:-1]),
        1.0, 0.0
    )
    return out
