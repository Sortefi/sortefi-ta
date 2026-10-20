"""Trend building blocks: layered / adaptive moving averages, ADX, Aroon, Parabolic SAR,
Vortex, Choppiness and volume-weighted average.

Plain numpy / pandas. Every function returns None when the input is shorter than it needs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sortefi_ta._ta._base import (
    EPS, _long_enough, _named, _nonzero_range, ema, rma, sma, true_range, wma,
)

__all__ = ["dema", "tema", "hma", "kama", "alma", "adx", "aroon", "psar", "vortex", "chop", "vwma"]


def _pos(v, default):
    """A positive number as given, else the default."""
    return v if isinstance(v, (int, float, np.integer, np.floating)) and v > 0 else default


def _snap(s: pd.Series) -> pd.Series:
    """Values within machine epsilon of zero become exactly zero."""
    return s.apply(lambda x: 0 if abs(x) < EPS else x)


def _atr(high, low, close, length=14, prenan=False):
    """Average true range: Wilder average of true range, seeded with the simple mean of the
    first `length` true ranges. `prenan` blanks the first true range (no previous close)."""
    if not (_long_enough(high, length + 1) and _long_enough(low, length + 1)
            and _long_enough(close, length + 1)):
        return None
    tr = true_range(high, low, close)
    if tr is None:
        return None
    if prenan:
        tr.iloc[:1] = np.nan
    if tr.isna().all():
        return None
    first = tr.iloc[:length].mean()
    tr.iloc[:length - 1] = np.nan
    tr.iloc[length - 1] = first
    out = rma(tr, length=length)
    if out is None or out.isna().all():
        return None
    return out


# ── layered / adaptive moving averages ───────────────────────────────────────
def dema(close: pd.Series, length: int = 10, **_):
    """Double EMA: 2 × EMA − EMA of the EMA (less lag than a plain EMA)."""
    length = _pos(length, 10)
    if not _long_enough(close, length):
        return None
    e1 = ema(close, length=length)
    e2 = ema(e1, length=length)
    out = 2 * e1 - e2
    if out.isna().all():
        return None
    return _named(out, f"DEMA_{length}")


def tema(close: pd.Series, length: int = 10, **_):
    """Triple EMA: 3 × (EMA − EMA²) + EMA³."""
    length = _pos(length, 10)
    if not _long_enough(close, 3 * length):
        return None
    e1 = ema(close, length=length)
    e2 = ema(e1, length=length)
    e3 = ema(e2, length=length)
    return _named(3 * (e1 - e2) + e3, f"TEMA_{length}")


def hma(close: pd.Series, length: int = 10, **_):
    """Hull MA: WMA over √length of (2 × WMA(length/2) − WMA(length))."""
    length = _pos(length, 10)
    if not _long_enough(close, length + 2):
        return None
    fast = wma(close, length=int(length / 2))
    slow = wma(close, length=length)
    out = wma(2 * fast - slow, length=int(np.sqrt(length)))
    return _named(out, f"HMA_{length}")


def kama(close: pd.Series, length: int = 10, fast: int = 2, slow: int = 30, drift: int = 1, **_):
    """Kaufman adaptive MA: smoothing slides between the fast and slow EMA rates according to
    the efficiency ratio (net move ÷ path walked over `length` candles)."""
    length, fast, slow = _pos(length, 10), _pos(fast, 2), _pos(slow, 30)
    if not _long_enough(close, max(fast, slow, length)):
        return None
    fr, sr = 2 / (fast + 1), 2 / (slow + 1)
    net = _nonzero_range(close, close.shift(length)).abs()
    path = _nonzero_range(close, close.shift(drift)).abs().rolling(length).sum()
    sc = ((net / path) * (fr - sr) + sr) ** 2
    seed = sma(close.iloc[:length], length=length).iloc[-1]
    vals = [np.nan] * (length - 1) + [seed]
    for i in range(length, close.size):
        vals.append(sc.iat[i] * close.iat[i] + (1 - sc.iat[i]) * vals[i - 1])
    return _named(pd.Series(vals, index=close.index), f"KAMA_{length}_{fast}_{slow}")


def alma(close: pd.Series, length: int = 9, sigma: float = 6.0, dist_offset: float = None, **_):
    """Arnaud Legoux MA: Gaussian-weighted window centred at `dist_offset` (default 0.85) of
    the window. Only the keyword `dist_offset` moves the centre."""
    length = _pos(length, 9)
    if not _long_enough(close, length):
        return None
    sigma = _pos(sigma, 6.0)
    centre = float(dist_offset) if isinstance(dist_offset, float) and 0 <= dist_offset <= 1 else 0.85
    x = np.arange(length)
    k = np.floor(centre * (length - 1))
    w = np.exp(-0.5 * ((sigma / length) * (x - k)) ** 2)
    w /= w.sum()
    win = np.lib.stride_tricks.sliding_window_view(close.to_numpy(), length)
    out = np.append(np.array([np.nan] * (length - 1)), np.tensordot(win, w, axes=1))
    return _named(pd.Series(out, index=close.index), f"ALMA_{length}_{sigma}_{centre}")


def vwma(close: pd.Series, volume: pd.Series, length: int = 10, **_):
    """Volume-weighted MA: Σ(close × volume) ÷ Σ volume over `length` candles."""
    length = _pos(length, 10)
    if not (_long_enough(close, length) and _long_enough(volume, length)):
        return None
    out = sma(close * volume, length=length) / sma(volume, length=length)
    return _named(out, f"VWMA_{length}")


# ── direction / trend strength ───────────────────────────────────────────────
def adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14,
        signal_length: int = None, adxr_length: int = 2, scalar: float = 100,
        tvmode: bool = False, drift: int = 1, **_):
    """Average Directional Index with +DI / −DI. `tvmode=True` uses the charting-platform
    seeding (first value = sum of the first `length` moves, then Wilder smoothing)."""
    length = _pos(length, 14)
    signal_length = _pos(signal_length, length)
    adxr_length = _pos(adxr_length, 2)
    need = max(length, signal_length, adxr_length)
    if not (_long_enough(high, need) and _long_enough(low, need) and _long_enough(close, need)):
        return None
    scalar = float(scalar) if isinstance(scalar, (int, float)) else 100.0
    atr_ = _atr(high, low, close, length=length, prenan=True)
    if atr_ is None:
        return None
    k = scalar / atr_
    up = high - high.shift(drift)
    dn = low.shift(drift) - low
    pos = _snap(((up > dn) & (up > 0)) * up)
    neg = _snap(((dn > up) & (dn > 0)) * dn)
    if tvmode:
        for s in (pos, neg):
            s.iloc[length - 1] = s.iloc[:length].sum()
            s.iloc[:length - 1] = 0
        a = 1 / length
        dmp = k * pos.ewm(alpha=a, adjust=False, min_periods=length).mean()
        dmn = k * neg.ewm(alpha=a, adjust=False, min_periods=length).mean()
        dx = (scalar * (dmp - dmn).abs() / (dmp + dmn)).shift(-length)
        dx.iloc[length - 1] = dx.iloc[:length].sum()
        dx.iloc[:length - 1] = 0
        adx_ = rma(dx, length=signal_length)
        adx_.iloc[:length - 1] = np.nan
        adx_ = adx_.shift(length)
    else:
        dmp = k * rma(pos, length=length)
        dmn = k * rma(neg, length=length)
        dx = scalar * (dmp - dmn).abs() / (dmp + dmn)
        adx_ = rma(dx, length=signal_length)
    adxr = 0.5 * (adx_ + adx_.shift(adxr_length))
    return pd.DataFrame({
        f"ADX_{signal_length}": adx_,
        f"ADXR_{signal_length}_{adxr_length}": adxr,
        f"DMP_{length}": dmp,
        f"DMN_{length}": dmn,
    }, index=close.index)


def aroon(high: pd.Series, low: pd.Series, length: int = 14, scalar: float = 100, **_):
    """Aroon up / down: how recently (as a % of `length`) the window's high / low was set;
    oscillator = up − down."""
    length = _pos(length, 14)
    if not (_long_enough(high, length + 1) and _long_enough(low, length + 1)):
        return None
    scalar = float(scalar) if isinstance(scalar, (int, float)) else 100.0
    since_hi = high.rolling(length + 1).apply(lambda w: int(np.argmax(w[::-1])), raw=True)
    since_lo = low.rolling(length + 1).apply(lambda w: int(np.argmin(w[::-1])), raw=True)
    up = scalar * (1 - since_hi / length)
    down = scalar * (1 - since_lo / length)
    return pd.DataFrame({
        f"AROOND_{length}": down, f"AROONU_{length}": up, f"AROONOSC_{length}": up - down,
    }, index=high.index)


def psar(high: pd.Series, low: pd.Series, close: pd.Series = None, af0: float = None,
         af: float = None, max_af: float = None, **_):
    """Parabolic SAR. Columns: long stop, short stop, acceleration factor, reversal flag."""
    if not (_long_enough(high, 1) and _long_enough(low, 1)):
        return None
    step = _pos(af, 0.02)
    af0 = _pos(af0, step)
    max_af = _pos(max_af, 0.2)
    idx = high.index
    up2 = high.iloc[:2] - high.iloc[:2].shift(1)
    dn2 = low.iloc[:2].shift(1) - low.iloc[:2]
    falling = _snap(((dn2 > up2) & (dn2 > 0)) * dn2).iloc[-1] > 0
    h, lo = high.to_numpy(), low.to_numpy()
    m = h.size
    sar, long_, short_ = np.zeros(m), np.full(m, np.nan), np.full(m, np.nan)
    rev, afs = np.zeros(m, dtype=int), np.zeros(m)
    afs[:2] = af0
    a = af0
    ep = lo[0] if falling else h[0]
    sar[0] = close.iloc[0] if isinstance(close, pd.Series) else (h[0] if falling else lo[0])
    for i in range(1, m):
        sar[i] = sar[i - 1] + a * (ep - sar[i - 1])
        if falling:
            flip = h[i] > sar[i]
            if lo[i] < ep:
                ep, a = lo[i], min(a + af0, max_af)
            sar[i] = max(h[i - 1], sar[i])
        else:
            flip = lo[i] < sar[i]
            if h[i] > ep:
                ep, a = h[i], min(a + af0, max_af)
            sar[i] = min(lo[i - 1], sar[i])
        if flip:
            sar[i], a, falling = ep, af0, not falling
            ep = lo[i] if falling else h[i]
        (short_ if falling else long_)[i] = sar[i]
        afs[i], rev[i] = a, int(flip)
    tag = f"_{af0}_{max_af}"
    return pd.DataFrame({
        f"PSARl{tag}": pd.Series(long_, index=idx), f"PSARs{tag}": pd.Series(short_, index=idx),
        f"PSARaf{tag}": pd.Series(afs, index=idx), f"PSARr{tag}": pd.Series(rev, index=idx),
    }, index=idx)


def vortex(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14, drift: int = 1, **_):
    """Vortex +VI / −VI: summed |high − previous low| and |low − previous high| ÷ summed true range."""
    length = _pos(length, 14)
    if not (_long_enough(high, length) and _long_enough(low, length) and _long_enough(close, length)):
        return None
    trs = true_range(high, low, close).rolling(length, min_periods=length).sum()
    vip = (high - low.shift(drift)).abs().rolling(length, min_periods=length).sum() / trs
    vim = (low - high.shift(drift)).abs().rolling(length, min_periods=length).sum() / trs
    return pd.DataFrame({f"VTXP_{length}": vip, f"VTXM_{length}": vim}, index=close.index)


def chop(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14,
         atr_length: int = 1, ln: bool = False, scalar: float = 100, **_):
    """Choppiness Index: 100 × log10(Σ ATR(1) ÷ (highest high − lowest low)) ÷ log10(length)."""
    length = _pos(length, 14)
    if not (_long_enough(high, length + 1) and _long_enough(low, length + 1)
            and _long_enough(close, length + 1)):
        return None
    atr_length = _pos(atr_length, 1)
    scalar = float(scalar) if isinstance(scalar, (int, float)) else 100.0
    span = high.rolling(length).max() - low.rolling(length).min()
    atr_sum = _atr(high, low, close, length=atr_length).rolling(length).sum()
    log = np.log if ln else np.log10
    out = scalar * (log(atr_sum) - log(span)) / log(length)
    return _named(out, f"CHOP{'ln' if ln else ''}_{length}_{atr_length}_{scalar}")
