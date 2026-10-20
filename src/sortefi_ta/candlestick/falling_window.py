"""Falling Window — gap down: current high is below previous low."""
import numpy as np
from . import _helpers as h


def detect(o, hi, lo, c):
    n = len(o)
    if n < 2:
        return h.zeros(n)
    out = h.zeros(n)
    out[1:] = np.where(hi[1:] < lo[:-1], 1.0, 0.0)
    return out
