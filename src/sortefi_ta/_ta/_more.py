"""Klinger volume oscillator, SuperTrend and Tillson T3."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sortefi_ta._ta._base import _long_enough, _named, ema
from sortefi_ta._ta._volvol import atr

__all__ = ["kvo", "supertrend", "t3"]


def kvo(high, low, close, volume, fast: int = 34, slow: int = 55, signal: int = 13, **_):
    """Klinger volume oscillator: volume signed by the day-on-day direction of the typical price
    (high + low + close) ÷ 3, then EMA(fast) − EMA(slow); signal line = EMA(signal) of that."""
    need = max(fast, slow) + signal
    if not all(_long_enough(s, need) for s in (high, low, close, volume)):
        return None
    typical = (high + low + close) / 3.0
    direction = typical.diff(1)
    direction[direction > 0] = 1
    direction[direction < 0] = -1
    direction.iloc[0] = np.nan
    signed = volume * direction
    signed = signed.loc[signed.first_valid_index():]
    osc = ema(signed, fast) - ema(signed, slow)
    if osc is None or osc.isna().all():
        return None
    sig = ema(osc.loc[osc.first_valid_index():], signal)
    if sig is None or sig.isna().all():
        return None
    tag = f"_{fast}_{slow}_{signal}"
    return pd.DataFrame({f"KVO{tag}": osc, f"KVOs{tag}": sig}, index=close.index)


def supertrend(high, low, close, length: int = 7, multiplier: float = 3.0, **_):
    """SuperTrend: bands at the candle midpoint ± multiplier × ATR(length). The trend flips up when
    the close breaks above the previous upper band and down when it breaks below the previous lower
    band; while a trend holds, its band only ever tightens. Direction 1 = up, -1 = down."""
    if not all(_long_enough(s, length + 1) for s in (high, low, close)):
        return None
    m = close.size
    mid = ((high + low) / 2).to_numpy(dtype=float)
    width = multiplier * atr(high, low, close, length).to_numpy(dtype=float)
    lower, upper = mid - width, mid + width
    c = close.to_numpy(dtype=float)
    direction = np.ones(m)
    line = np.zeros(m)
    long_band = np.full(m, np.nan)
    short_band = np.full(m, np.nan)
    for i in range(1, m):
        if c[i] > upper[i - 1]:
            direction[i] = 1
        elif c[i] < lower[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
            if direction[i] > 0 and lower[i] < lower[i - 1]:
                lower[i] = lower[i - 1]
            if direction[i] < 0 and upper[i] > upper[i - 1]:
                upper[i] = upper[i - 1]
        if direction[i] > 0:
            line[i] = long_band[i] = lower[i]
        else:
            line[i] = short_band[i] = upper[i]
    line[0] = np.nan
    direction[:length] = np.nan
    tag = f"_{length}_{multiplier}"
    return pd.DataFrame({f"SUPERT{tag}": line, f"SUPERTd{tag}": direction,
                         f"SUPERTl{tag}": long_band, f"SUPERTs{tag}": short_band}, index=close.index)


def t3(close, length: int = 10, a: float = 0.7, **_):
    """Tillson T3: six chained EMAs combined with weights set by the volume factor `a`
    (a float strictly between 0 and 1; anything else uses 0.7)."""
    if not _long_enough(close, 5 * (length + 1)):
        return None
    a = float(a) if isinstance(a, float) and 0 < a < 1 else 0.7
    e1 = ema(close, length)
    e2 = ema(e1, length)
    e3 = ema(e2, length)
    e4 = ema(e3, length)
    e5 = ema(e4, length)
    e6 = ema(e5, length)
    c1 = -a * a ** 2
    c2 = 3 * a ** 2 + 3 * a ** 3
    c3 = -6 * a ** 2 - 3 * a - 3 * a ** 3
    c4 = a ** 3 + 3 * a ** 2 + 3 * a + 1
    return _named(c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3, f"T3_{length}_{a}")
