"""Volatility and volume building blocks: ATR, Donchian, acceleration bands, mass index,
ulcer index, accumulation/distribution, its oscillator, Chaikin money flow, force index, OBV.

Plain numpy / pandas. Each returns None when the input is shorter than its window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sortefi_ta._ta._base import (
    _long_enough, _named, _nonzero_range, ema, rma, sma, true_range,
)

__all__ = ["atr", "donchian", "accbands", "massi", "ui", "ad", "adosc", "cmf", "efi", "obv"]


def _all_long(n, *series) -> bool:
    return all(_long_enough(s, n) for s in series)


def _all_nan(s) -> bool:
    return s is None or bool(np.isnan(np.asarray(s, dtype=float)).all())


# ── volatility ───────────────────────────────────────────────────────────────
def atr(high, low, close, length: int = 14, **_):
    """Average true range: Wilder's average of the true range, seeded with the simple average
    of the first `length` true ranges."""
    if not _all_long(length + 1, high, low, close):
        return None
    tr = true_range(high, low, close)
    if _all_nan(tr):
        return None
    seeded = tr.copy()
    first = seeded.iloc[:length].mean()
    seeded.iloc[:length - 1] = np.nan
    seeded.iloc[length - 1] = first
    out = rma(seeded, length=length)
    if _all_nan(out):
        return None
    return _named(out, f"ATRr_{length}")


def donchian(high, low, lower_length: int = 20, upper_length: int = 20, **_):
    """Donchian channel: lowest low, highest high, and their midpoint over the window."""
    if not _all_long(max(lower_length, upper_length), high, low):
        return None
    lower = low.rolling(lower_length, min_periods=lower_length).min()
    upper = high.rolling(upper_length, min_periods=upper_length).max()
    mid = 0.5 * (lower + upper)
    tag = f"{lower_length}_{upper_length}"
    return pd.DataFrame({f"DCL_{tag}": lower, f"DCM_{tag}": mid, f"DCU_{tag}": upper},
                        index=high.index)


def accbands(high, low, close, length: int = 20, c: float = 4, **_):
    """Acceleration bands: simple averages of low and high widened by c × (high − low) ÷
    (high + low), plus the simple average of close as the middle line."""
    if not _all_long(length, high, low, close):
        return None
    width = c * (_nonzero_range(high, low) / (high + low))
    lower = sma(low * (1 - width), length=length)
    mid = sma(close, length=length)
    upper = sma(high * (1 + width), length=length)
    return pd.DataFrame({f"ACCBL_{length}": lower, f"ACCBM_{length}": mid,
                         f"ACCBU_{length}": upper}, index=close.index)


def massi(high, low, fast: int = 9, slow: int = 25, **_):
    """Mass index: rolling sum over `slow` candles of EMA(range) ÷ EMA(EMA(range))."""
    fast, slow = min(fast, slow), max(fast, slow)
    if not _all_long(2 * slow - fast, high, low):
        return None
    rng = _nonzero_range(high, low)
    e1 = ema(rng, length=fast)
    if _all_nan(e1):
        return None
    e2 = ema(e1, length=fast)
    if _all_nan(e2):
        return None
    out = (e1 / e2).rolling(slow, min_periods=slow).sum()
    if _all_nan(out):
        return None
    return _named(out, f"MASSI_{fast}_{slow}")


def ui(close, length: int = 14, scalar: float = 100, **_):
    """Ulcer index: root-mean-square of the percent drawdown from the rolling highest close."""
    if not _long_enough(close, 2 * length - 1):
        return None
    peak = close.rolling(length).max()
    drawdown = scalar * (close - peak) / peak
    out = np.sqrt((drawdown * drawdown).rolling(length).sum() / length)
    return _named(out, f"UI_{length}")


# ── volume ───────────────────────────────────────────────────────────────────
def _money_flow_volume(high, low, close, volume):
    """Close location in the candle (−1…+1) × volume."""
    return (2 * close - (high + low)) * (volume / _nonzero_range(high, low))


def ad(high, low, close, volume, **_):
    """Accumulation / distribution line: running total of money-flow volume."""
    if not all(isinstance(s, pd.Series) for s in (high, low, close, volume)):
        return None
    return _named(_money_flow_volume(high, low, close, volume).cumsum(), "AD")


def adosc(high, low, close, volume, fast: int = 3, slow: int = 10, **_):
    """Chaikin A/D oscillator: EMA(fast) − EMA(slow) of the accumulation / distribution line."""
    if not _all_long(max(fast, slow), high, low, close, volume):
        return None
    line = ad(high, low, close, volume)
    out = ema(line, length=fast) - ema(line, length=slow)
    return _named(out, f"ADOSC_{fast}_{slow}")


def cmf(high, low, close, volume, length: int = 20, **_):
    """Chaikin money flow: money-flow volume summed over the window ÷ volume summed."""
    if not _all_long(length, high, low, close, volume):
        return None
    mfv = _money_flow_volume(high, low, close, volume)
    out = (mfv.rolling(length, min_periods=length).sum()
           / volume.rolling(length, min_periods=length).sum())
    return _named(out, f"CMF_{length}")


def efi(close, volume, length: int = 13, drift: int = 1, **_):
    """Elder force index: EMA of (close change × volume)."""
    if not _all_long(length, close, volume):
        return None
    return _named(ema(close.diff(drift) * volume, length=length), f"EFI_{length}")


def obv(close, volume, **_):
    """On-balance volume: running total of volume signed by the close's direction (the first
    candle has no direction, so it starts the total as missing)."""
    if not _all_long(1, close, volume):
        return None
    direction = close.diff(1)
    direction[direction > 0] = 1
    direction[direction < 0] = -1
    direction.iloc[0] = np.nan
    return _named((direction * volume).cumsum(), "OBV")
