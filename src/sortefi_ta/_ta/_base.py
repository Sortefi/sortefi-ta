"""Building blocks: moving averages, true range, deviation, regression line, momentum.

Plain numpy / pandas. Every function returns None when the input is shorter than its window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = np.finfo(float).eps


# ── small helpers shared by the other _ta modules ────────────────────────────
def _long_enough(s, n) -> bool:
    return isinstance(s, pd.Series) and s.size >= n


def _named(s: pd.Series, name: str) -> pd.Series:
    s.name = name
    return s


def _nonzero_range(x: pd.Series, y: pd.Series) -> pd.Series:
    """x − y, nudged by machine epsilon when any difference is exactly zero (avoids 0 ÷ 0)."""
    d = x - y
    if d.eq(0).any():
        d = d + EPS
    return d


def _lag_diff(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full(a.size, np.nan)
    if k < a.size:
        out[k:] = a[k:] - a[:-k] if k > 0 else 0.0
    return out


# ── moving averages ──────────────────────────────────────────────────────────
def sma(close: pd.Series, length: int = 10, **_) -> pd.Series | None:
    """Simple moving average: mean of the last `length` values."""
    if not _long_enough(close, length):
        return None
    x = close.to_numpy(dtype=float)
    m = x.size - length + 1
    acc = np.zeros(m)
    w = 1.0 / length
    for j in range(length):          # add each window left to right (bit-stable on flat data)
        acc += w * x[j:j + m]
    out = np.concatenate([np.full(length - 1, np.nan), acc])
    return _named(pd.Series(out, index=close.index), f"SMA_{length}")


def ema(close: pd.Series, length: int = 10, **_) -> pd.Series | None:
    """Exponential moving average, smoothing 2 ÷ (length + 1), seeded with the simple average of
    the first `length` values (the usual charting-platform convention)."""
    if not _long_enough(close, length):
        return None
    seeded = close.astype(float).copy()
    first = seeded.iloc[:length].mean()
    seeded.iloc[:length - 1] = np.nan
    seeded.iloc[length - 1] = first
    out = seeded.ewm(span=length, adjust=False).mean()
    return _named(out, f"EMA_{length}")


def rma(close: pd.Series, length: int = 10, **_) -> pd.Series | None:
    """Wilder's moving average: an EMA with smoothing 1 ÷ length (no seed)."""
    if not _long_enough(close, length):
        return None
    alpha = 1.0 / length if length > 0 else 0.5
    return _named(close.ewm(alpha=alpha, adjust=False).mean(), f"RMA_{length}")


def wma(close: pd.Series, length: int = 10, **_) -> pd.Series | None:
    """Linearly weighted moving average: newest value weighs `length`, oldest weighs 1."""
    if not _long_enough(close, length):
        return None
    x = close.to_numpy(dtype=float)
    w = np.arange(1, length + 1, dtype=float)
    out = np.zeros_like(x)
    for i in range(length - 1, x.size):
        out[i] = (w * x[i - length + 1:i + 1]).sum()
    out *= 2 / (length * length + length)
    out[:length - 1] = np.nan
    return _named(pd.Series(out, index=close.index), f"WMA_{length}")


# ── ranges and deviation ─────────────────────────────────────────────────────
def true_range(high: pd.Series, low: pd.Series, close: pd.Series, drift: int = 1, **_):
    """Largest of: high − low, |high − previous close|, |previous close − low|."""
    if not (_long_enough(high, 1) and _long_enough(low, 1) and _long_enough(close, 1)):
        return None
    prev = close.shift(drift)
    tr = pd.concat([_nonzero_range(high, low), high - prev, prev - low], axis=1).abs().max(axis=1)
    if tr.isna().all():
        return None
    return _named(tr, f"TRUERANGE_{drift}")


def mad(close: pd.Series, length: int = 30, **_) -> pd.Series | None:
    """Rolling mean absolute deviation from the window's mean."""
    if not _long_enough(close, length):
        return None
    out = close.rolling(length, min_periods=length).apply(
        lambda w: np.abs(w - w.mean()).mean(), raw=True)
    return _named(out, f"MAD_{length}")


dev = mad  # same measure under its charting-script name


# ── regression line ──────────────────────────────────────────────────────────
def linreg(close: pd.Series, length: int = 14, slope: bool = False, intercept: bool = False,
           angle: bool = False, degrees: bool = False, **_) -> pd.Series | None:
    """Least-squares line through the last `length` values (x = 1…length). Returns its end value,
    or its slope / intercept / angle (radians unless degrees=True)."""
    if not _long_enough(close, length):
        return None
    x = np.arange(1, length + 1)
    sx = 0.5 * length * (length + 1)
    sxx = sx * (2 * length + 1) / 3
    div = length * sxx - sx * sx
    vals = []
    for y in np.lib.stride_tricks.sliding_window_view(close.to_numpy(), length):
        sy, sxy = y.sum(), (x * y).sum()
        m = (length * sxy - sx * sy) / div
        if slope:
            vals.append(m)
            continue
        b = (sy * sxx - sx * sxy) / div
        if intercept:
            vals.append(b)
        elif angle:
            vals.append(np.arctan(m) * (180 / np.pi if degrees else 1))
        else:
            vals.append(m * length + b)
    out = pd.Series([np.nan] * (length - 1) + vals, index=close.index)
    suffix = "m" if slope else "b" if intercept else "a" if angle else ""
    return _named(out, f"LINREG{suffix}_{length}")


# ── momentum ─────────────────────────────────────────────────────────────────
def mom(close: pd.Series, length: int = 10, **_) -> pd.Series | None:
    """Close minus the close `length` candles ago."""
    if not _long_enough(close, length + 1):
        return None
    out = _lag_diff(close.to_numpy(dtype=float), length)
    return _named(pd.Series(out, index=close.index), f"MOM_{length}")


def roc(close: pd.Series, length: int = 10, scalar: float = 100, **_) -> pd.Series | None:
    """Percent change versus `length` candles ago."""
    if not _long_enough(close, length + 1):
        return None
    x = close.to_numpy(dtype=float)
    prev = np.full(x.size, np.nan)
    prev[length:] = x[:-length]
    out = float(scalar) * _lag_diff(x, length) / prev
    return _named(pd.Series(out, index=close.index), f"ROC_{length}")


def change(close: pd.Series, length: int = 1, **_) -> pd.Series:
    """Difference from `length` candles ago (charting-script `change`)."""
    return close.diff(length)
