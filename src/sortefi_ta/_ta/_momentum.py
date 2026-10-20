"""Momentum building blocks: RSI, MACD, Stochastic RSI, TSI, Ultimate Oscillator, PPO, KDJ,
Fisher Transform, Awesome Oscillator, Balance of Power, Williams %R, Money Flow Index.

Plain numpy / pandas. Each returns None when the input is shorter than the window it needs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sortefi_ta._ta._base import EPS, _long_enough, _named, _nonzero_range, ema, rma, sma

__all__ = ["rsi", "macd", "stochrsi", "tsi", "uo", "ppo", "kdj", "fisher", "ao", "bop", "willr", "mfi"]


def _all_nan(s) -> bool:
    return bool(np.isnan(np.asarray(s, dtype=float)).all())


def _wilder_warm(x: pd.Series, n: int) -> pd.Series:
    """Wilder smoothing (1 ÷ n) in pandas' adjusted form, blank until n values are seen."""
    a = 1.0 / n if n > 0 else 0.5
    return x.ewm(alpha=a, min_periods=n).mean()


def rsi(close: pd.Series, length: int = 14, scalar: float = 100, drift: int = 1, **_):
    """Relative Strength Index: Wilder-smoothed average gain ÷ (average gain + average loss)."""
    if not _long_enough(close, length + 1):
        return None
    move = close.diff(drift)
    gains, losses = move.copy(), move.copy()
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    up, down = rma(gains, length=length), rma(losses, length=length)
    out = float(scalar) * up / (up + down.abs())
    return _named(out, f"RSI_{length}")


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9, **_):
    """MACD line (fast EMA − slow EMA), its signal EMA, and the histogram (line − signal).
    Columns: MACD, MACDh (histogram), MACDs (signal)."""
    if slow < fast:
        fast, slow = slow, fast
    if not _long_enough(close, slow + signal - 1):
        return None
    line = ema(close, length=fast) - ema(close, length=slow)
    sig = ema(line.loc[line.first_valid_index():], length=signal)
    hist = line - sig
    tag = f"_{fast}_{slow}_{signal}"
    return pd.DataFrame({f"MACD{tag}": line, f"MACDh{tag}": hist, f"MACDs{tag}": sig},
                        index=close.index)


def stochrsi(close: pd.Series, length: int = 14, rsi_length: int = 14, k: int = 3, d: int = 3, **_):
    """Stochastic of RSI: where RSI sits in its `length`-candle range (0–100), then %K = SMA(k)
    of that and %D = SMA(d) of %K."""
    if not _long_enough(close, length + rsi_length + 2):
        return None
    r = rsi(close, length=rsi_length)
    lo, hi = r.rolling(length).min(), r.rolling(length).max()
    stoch = 100 * (r - lo) / _nonzero_range(hi, lo)
    k_line = sma(stoch, length=k)
    d_line = sma(k_line, length=d)
    tag = f"_{length}_{rsi_length}_{k}_{d}"
    return pd.DataFrame({f"STOCHRSIk{tag}": k_line, f"STOCHRSId{tag}": d_line}, index=close.index)


def tsi(close: pd.Series, fast: int = 13, slow: int = 25, signal: int = 13, scalar: float = 100,
        drift: int = 1, **_):
    """True Strength Index: double-smoothed price change ÷ double-smoothed absolute change,
    plus an EMA signal line."""
    if slow < fast:
        fast, slow = slow, fast
    if not _long_enough(close, slow + signal + 1):
        return None
    move = close.diff(drift)
    smooth = ema(move, length=slow)
    if _all_nan(smooth):
        return None
    smooth2 = ema(smooth, length=fast)
    abs_smooth = ema(move.abs(), length=slow)
    if _all_nan(abs_smooth):
        return None
    line = float(scalar) * smooth2 / ema(abs_smooth, length=fast)
    if _all_nan(line):
        return None
    sig = ema(line, length=signal)
    tag = f"_{fast}_{slow}_{signal}"
    return pd.DataFrame({f"TSI{tag}": line, f"TSIs{tag}": sig}, index=close.index)


def uo(high: pd.Series, low: pd.Series, close: pd.Series, fast: int = 7, medium: int = 14,
       slow: int = 28, fast_w: float = 4.0, medium_w: float = 2.0, slow_w: float = 1.0,
       drift: int = 1, **_):
    """Ultimate Oscillator: weighted mix of buying pressure ÷ true range over three windows."""
    need = max(fast, medium, slow) + 1
    if not all(_long_enough(s, need) for s in (high, low, close)):
        return None
    prev = close.shift(drift)
    top = pd.concat([high, prev], axis=1).max(axis=1)
    bottom = pd.concat([low, prev], axis=1).min(axis=1)
    pressure, rng = close - bottom, top - bottom

    def avg(n):
        return pressure.rolling(n).sum() / rng.rolling(n).sum()

    mix = fast_w * avg(fast) + medium_w * avg(medium) + slow_w * avg(slow)
    out = 100 * mix / (fast_w + medium_w + slow_w)
    return _named(out, f"UO_{fast}_{medium}_{slow}")


def ppo(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9, scalar: float = 100, **_):
    """Percentage Price Oscillator: (fast SMA − slow SMA) ÷ slow SMA × 100, an EMA signal and
    the histogram. Columns: PPO, PPOh, PPOs."""
    if slow < fast:
        fast, slow = slow, fast
    if not _long_enough(close, max(fast, slow, signal)):
        return None
    slow_ma = sma(close, length=slow)
    line = float(scalar) * (sma(close, length=fast) - slow_ma) / slow_ma
    if _all_nan(line):
        return None
    sig = ema(line, length=signal)
    tag = f"_{fast}_{slow}_{signal}"
    return pd.DataFrame({f"PPO{tag}": line, f"PPOh{tag}": line - sig, f"PPOs{tag}": sig},
                        index=close.index)


def kdj(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 9, signal: int = 3, **_):
    """KDJ: raw stochastic smoothed into K, K smoothed into D, and J = 3K − 2D."""
    if not all(_long_enough(s, length + signal + 1) for s in (high, low, close)):
        return None
    hh, ll = high.rolling(length).max(), low.rolling(length).min()
    raw = 100 * (close - ll) / _nonzero_range(hh, ll)
    k_line = _wilder_warm(raw, signal)
    d_line = _wilder_warm(k_line, signal)
    tag = f"_{length}_{signal}"
    return pd.DataFrame({f"K{tag}": k_line, f"D{tag}": d_line, f"J{tag}": 3 * k_line - 2 * d_line},
                        index=close.index)


def fisher(high: pd.Series, low: pd.Series, length: int = 9, signal: int = 1, **_):
    """Fisher Transform of the median price's position in its `length`-candle range, with the
    previous value(s) as the signal line."""
    if not all(_long_enough(s, max(length, signal)) for s in (high, low)):
        return None
    mid = pd.Series(0.5 * (high.to_numpy() + low.to_numpy()), index=high.index)
    hi, lo = mid.rolling(length).max(), mid.rolling(length).min()
    span = _nonzero_range(hi, lo)
    span[span < 0.001] = 0.001
    pos = ((mid - lo) / span - 0.5).to_numpy()
    vals = [np.nan] * (length - 1) + [0]
    v = 0
    for i in range(length, high.size):
        v = 0.66 * pos[i] + 0.67 * v
        if v < -0.99:
            v = -0.999
        if v > 0.99:
            v = 0.999
        vals.append(0.5 * (np.log((1 + v) / (1 - v)) + vals[i - 1]))
    line = pd.Series(vals, index=high.index)
    if _all_nan(line):
        return None
    tag = f"_{length}_{signal}"
    return pd.DataFrame({f"FISHERT{tag}": line, f"FISHERTs{tag}": line.shift(signal)},
                        index=high.index)


def ao(high: pd.Series, low: pd.Series, fast: int = 5, slow: int = 34, **_):
    """Awesome Oscillator: fast SMA − slow SMA of the median price."""
    if slow < fast:
        fast, slow = slow, fast
    if not all(_long_enough(s, max(fast, slow)) for s in (high, low)):
        return None
    mid = 0.5 * (high + low)
    return _named(sma(mid, length=fast) - sma(mid, length=slow), f"AO_{fast}_{slow}")


def bop(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, scalar: float = 1, **_):
    """Balance of Power: (close − open) ÷ (high − low)."""
    out = float(scalar) * _nonzero_range(close, open_) / _nonzero_range(high, low)
    return _named(out, "BOP")


def willr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14, **_):
    """Williams %R: where the close sits in the `length`-candle range, from −100 to 0."""
    if not all(_long_enough(s, length) for s in (high, low, close)):
        return None
    ll = low.rolling(length, min_periods=length).min()
    hh = high.rolling(length, min_periods=length).max()
    return _named(100 * ((close - ll) / (hh - ll) - 1), f"WILLR_{length}")


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int = 14,
        drift: int = 1, **_):
    """Money Flow Index: volume-weighted RSI of the typical price ((high + low + close) ÷ 3)."""
    if not all(_long_enough(s, length + 1) for s in (high, low, close, volume)):
        return None
    m = close.size
    tp = (high.to_numpy() + low.to_numpy() + close.to_numpy()) / 3.0
    flow = tp * volume.to_numpy() * np.where(tp > np.roll(tp, shift=drift), 1, -1)
    win = np.ones(length)
    gain = np.convolve(np.maximum(flow, 0), win)[:m]
    loss = np.convolve(np.maximum(-flow, 0), win)[:m]
    out = 100.0 * gain / (gain + loss + EPS)
    out[:length] = np.nan
    return _named(pd.Series(out, index=close.index), f"MFI_{length}")
