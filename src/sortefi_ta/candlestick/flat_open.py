"""Flat Open — the candle opens at its extreme: no wick on the open side.

Bearish: a red candle (close < open) whose high is within FLAT_TOL of the open — it opened at the high.
Bullish: a green candle (close > open) whose low is within FLAT_TOL of the open — it opened at the low.
A candle with close == open is neither. FLAT_TOL is a share of PRICE (0.003 = 0.3 %), not of the
candle's range. Unlike belt hold, the body may be any size. Detected as either variant (1.0 for any
flat open); split by colour with the candle's direction (close vs open).

Change the tolerance by setting `sortefi_ta.candlestick.flat_open.FLAT_TOL` (default 0.003)."""
import numpy as np

FLAT_TOL = 0.003          # share of price


def detect(o, hi, lo, c):
    o, hi, lo, c = (np.asarray(x, dtype=float) for x in (o, hi, lo, c))
    with np.errstate(divide="ignore", invalid="ignore"):
        up_gap = (hi - o) / o
        dn_gap = (o - lo) / o
    ok = np.isfinite(o) & (o > 0)
    bearish = ok & (c < o) & (up_gap <= FLAT_TOL + 1e-12)
    bullish = ok & (c > o) & (dn_gap <= FLAT_TOL + 1e-12)
    return np.where(bullish | bearish, 1.0, 0.0)
