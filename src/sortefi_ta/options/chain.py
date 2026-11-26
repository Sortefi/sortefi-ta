"""Option-chain analytics for one underlying.

Chain DataFrame columns (only the ones a function needs must be present):

    expiry      expiry date (any sortable value); optional — without it the frame is one expiry
    strike      strike price
    opt_type    'C' / 'P' (also 'CE' / 'PE', 'call' / 'put')
    price       option premium (settle, close or mark)
    oi          open interest
    volume      traded volume
    iv          implied vol as a decimal (optional; computed from price when missing)
    forward     forward / futures price for that expiry (or 'underlying' as a fallback)
    t           years to expiry (optional if passed as an argument)

Per-expiry functions return a pandas Series indexed by expiry when the frame has an
`expiry` column, and a single float when it does not. Missing or bad data gives NaN.
"""
import math

import numpy as np
import pandas as pd

from ._util import option_sign
from .iv import implied_vol_black76
from .pricing import black76_greeks


# -- helpers ---------------------------------------------------------------

def _signs(df: pd.DataFrame) -> np.ndarray:
    if "opt_type" not in df:
        return np.full(len(df), np.nan)
    return np.atleast_1d(option_sign(df["opt_type"].to_numpy()))


def _per_expiry(df: pd.DataFrame, fn):
    """Apply fn(sub_frame, expiry_key) per expiry; Series if `expiry` exists, else float."""
    if "expiry" not in df:
        return fn(df, None)
    keys = sorted(df["expiry"].dropna().unique())
    return pd.Series({k: fn(df[df["expiry"] == k], k) for k in keys}, dtype=float,
                     name=getattr(fn, "__name__", None))


def _lookup(value, key, sub: pd.DataFrame, cols):
    """Resolve forward / t: scalar, mapping by expiry, or a column of the frame."""
    if value is None:
        for c in cols:
            if c in sub:
                v = pd.to_numeric(sub[c], errors="coerce").dropna()
                return float(v.iloc[0]) if len(v) else np.nan
        return np.nan
    if isinstance(value, (dict, pd.Series)):
        try:
            return float(value[key])
        except (KeyError, TypeError, ValueError):
            return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _ivs(sub: pd.DataFrame, fwd: float, t: float, r: float) -> np.ndarray:
    if "iv" in sub:
        iv = pd.to_numeric(sub["iv"], errors="coerce").to_numpy(dtype=float)
    else:
        iv = np.full(len(sub), np.nan)
    need = ~np.isfinite(iv)
    if need.any() and "price" in sub and np.isfinite(fwd) and np.isfinite(t):
        price = pd.to_numeric(sub["price"], errors="coerce").to_numpy(dtype=float)
        strike = pd.to_numeric(sub["strike"], errors="coerce").to_numpy(dtype=float)
        calc = np.atleast_1d(implied_vol_black76(price, fwd, strike, t, r, _signs(sub)))
        iv = np.where(need, calc, iv)
    return iv


# -- put/call ratio, max pain ---------------------------------------------

def pcr(df: pd.DataFrame, by: str = "oi") -> float:
    """Put/call ratio = sum of puts' `by` column / sum of calls' (by = 'oi' or 'volume').

    Covers every row passed (filter the frame first for one expiry). NaN if no call total.
    """
    if by not in df:
        return np.nan
    sign = _signs(df)
    val = pd.to_numeric(df[by], errors="coerce").to_numpy(dtype=float)
    calls = np.nansum(val[sign > 0])
    puts = np.nansum(val[sign < 0])
    return float(puts / calls) if calls > 0 else np.nan


def max_pain(df: pd.DataFrame):
    """Max-pain strike: the listed strike at which option holders' total intrinsic payout,
    weighted by open interest, is smallest. Ties go to the lowest strike."""
    def one(sub, _key):
        sign = _signs(sub)
        k = pd.to_numeric(sub["strike"], errors="coerce").to_numpy(dtype=float)
        oi = pd.to_numeric(sub["oi"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(k) & np.isfinite(oi) & np.isfinite(sign)
        k, oi, sign = k[ok], oi[ok], sign[ok]
        if k.size == 0 or np.sum(oi) <= 0:
            return np.nan
        cands = np.unique(k)
        diff = cands[:, None] - k[None, :]                 # settle - strike
        payout = np.where(sign[None, :] > 0, np.maximum(diff, 0.0), np.maximum(-diff, 0.0))
        total = payout @ oi
        return float(cands[int(np.argmin(total))])
    return _per_expiry(df, one)


# -- ATM IV, term structure, constant maturity -----------------------------

def _strike_iv_curve(sub, fwd, t, r):
    """Per-strike IV: average of call and put where both are finite."""
    iv = _ivs(sub, fwd, t, r)
    k = pd.to_numeric(sub["strike"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(iv) & np.isfinite(k) & (iv > 0)
    if not ok.any():
        return np.array([]), np.array([])
    s = pd.Series(iv[ok]).groupby(k[ok]).mean()
    return s.index.to_numpy(dtype=float), s.to_numpy(dtype=float)


def _atm_one(sub, fwd, t, r):
    if not (np.isfinite(fwd) and fwd > 0):
        return np.nan
    ks, ivs = _strike_iv_curve(sub, fwd, t, r)
    if ks.size == 0 or fwd < ks[0] or fwd > ks[-1]:
        return np.nan
    return float(np.interp(fwd, ks, ivs))


def atm_iv(df: pd.DataFrame, forward=None, t=None, r: float = 0.0):
    """At-the-money IV per expiry: IV at the forward, linearly interpolated (in strike)
    between the two nearest strikes, using the average of call and put IV where both exist.

    forward / t: scalar, a mapping {expiry: value}, or None to read the frame's `forward`
    (else `underlying`) and `t` columns. IVs come from the `iv` column, or are implied from
    `price` with Black-76 at that forward when missing. NaN if the forward is outside
    the listed strikes.
    """
    def one(sub, key):
        fwd = _lookup(forward, key, sub, ("forward", "underlying"))
        tt = _lookup(t, key, sub, ("t",))
        return _atm_one(sub, fwd, tt, r)
    return _per_expiry(df, one)


def term_structure(df: pd.DataFrame, forward=None, t=None, r: float = 0.0) -> pd.DataFrame:
    """ATM IV per expiry as a DataFrame [expiry, t, atm_iv], sorted by time to expiry."""
    groups = [(k, df[df["expiry"] == k]) for k in sorted(df["expiry"].dropna().unique())] \
        if "expiry" in df else [(None, df)]
    rows = []
    for key, sub in groups:
        fwd = _lookup(forward, key, sub, ("forward", "underlying"))
        tt = _lookup(t, key, sub, ("t",))
        rows.append({"expiry": key, "t": tt, "atm_iv": _atm_one(sub, fwd, tt, r)})
    out = pd.DataFrame(rows, columns=["expiry", "t", "atm_iv"])
    return out.sort_values("t", kind="stable").reset_index(drop=True)


def constant_maturity_iv(t, ivs, days: float = 30, extrapolate: bool = False) -> float:
    """IV at a constant maturity (default 30 calendar days) by linear interpolation in
    total variance (vol^2 * t) between the two expiries around it.

    t: years to each expiry; ivs: their ATM IVs (decimals). Outside the listed range
    returns NaN, or the nearest expiry's vol when extrapolate=True.
    """
    t = np.asarray(t, dtype=float).ravel()
    v = np.asarray(ivs, dtype=float).ravel()
    ok = np.isfinite(t) & np.isfinite(v) & (t > 0) & (v > 0)
    t, v = t[ok], v[ok]
    if t.size == 0 or not days > 0:
        return np.nan
    order = np.argsort(t)
    t, v = t[order], v[order]
    target = days / 365.0
    if target < t[0] or target > t[-1]:
        if not extrapolate:
            return np.nan
        return float(v[0] if target < t[0] else v[-1])
    w = np.interp(target, t, v * v * t)
    return float(np.sqrt(w / target)) if w > 0 else np.nan


# -- IV history ------------------------------------------------------------

def iv_rank(series, lookback: int = 252, min_periods: int = 2) -> np.ndarray:
    """IV rank = (current - min) / (max - min) * 100 over the last `lookback` values
    (current included).

    Missing values (NaN) inside the window are skipped, so one day without an IV does not
    blank the year. NaN until `lookback` values have elapsed, when the current value is
    missing, when the window holds fewer than `min_periods` finite values, or when
    max == min.
    """
    x = pd.Series(np.asarray(series, dtype=float))
    n = len(x)
    out = np.full(n, np.nan)
    if lookback < 1 or n < lookback:
        return out
    roll = x.rolling(lookback, min_periods=1)
    lo, hi, cnt = roll.min().to_numpy(), roll.max().to_numpy(), roll.count().to_numpy()
    cur = x.to_numpy()
    rng = hi - lo
    with np.errstate(divide="ignore", invalid="ignore"):
        rank = (cur - lo) / rng * 100.0
    ok = (np.arange(n) >= lookback - 1) & np.isfinite(cur) & (cnt >= max(min_periods, 1)) & (rng > 0)
    out[ok] = rank[ok]
    return out


def iv_percentile(series, lookback: int = 252, min_periods: int = 1) -> np.ndarray:
    """IV percentile = % of the previous `lookback` values (current excluded) that are
    below the current value.

    Missing values (NaN) among the previous values are skipped: the percentage is of the
    finite ones. NaN until `lookback` prior values have elapsed, when the current value is
    missing, or when fewer than `min_periods` prior values are finite.
    """
    x = np.asarray(series, dtype=float)
    n = x.size
    out = np.full(n, np.nan)
    if lookback < 1 or n <= lookback:
        return out
    win = np.lib.stride_tricks.sliding_window_view(x, lookback)[: n - lookback]
    cur = x[lookback:]
    finite = np.isfinite(win)
    count = finite.sum(axis=1)
    below = (np.where(finite, win, np.inf) < cur[:, None]).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = below / count * 100.0
    pct[(count < max(min_periods, 1)) | ~np.isfinite(cur)] = np.nan
    out[lookback:] = pct
    return out


# -- skew ------------------------------------------------------------------

def _skew_one(sub, fwd, t, r, delta):
    if not (np.isfinite(fwd) and fwd > 0 and np.isfinite(t) and t > 0):
        return np.nan
    sign = _signs(sub)
    k = pd.to_numeric(sub["strike"], errors="coerce").to_numpy(dtype=float)
    iv = _ivs(sub, fwd, t, r)
    d = np.atleast_1d(black76_greeks(fwd, k, t, iv, 0.0, sign)["delta"])  # forward delta
    ok = np.isfinite(d) & np.isfinite(iv)
    call = ok & (sign > 0) & (k >= fwd)
    put = ok & (sign < 0) & (k <= fwd)

    def at(mask, target):
        if mask.sum() < 2:
            return np.nan
        dd, vv = d[mask], iv[mask]
        o = np.argsort(dd)
        dd, vv = dd[o], vv[o]
        if target < dd[0] or target > dd[-1]:
            return np.nan
        return float(np.interp(target, dd, vv))

    return at(put, -delta) - at(call, delta)


def skew_25d(df: pd.DataFrame, forward=None, t=None, r: float = 0.0, delta: float = 0.25):
    """25-delta risk-reversal skew = IV(25-delta put) - IV(25-delta call), per expiry.

    Positive means downside puts are priced at higher vol than upside calls (the usual
    equity-index shape). Deltas are undiscounted Black-76 forward deltas computed with each
    option's own IV; IVs are interpolated linearly in delta among out-of-the-money options.
    NaN when 25 delta is not bracketed by listed strikes.
    """
    def one(sub, key):
        fwd = _lookup(forward, key, sub, ("forward", "underlying"))
        tt = _lookup(t, key, sub, ("t",))
        return _skew_one(sub, fwd, tt, r, delta)
    return _per_expiry(df, one)


# -- moves and OI build-up -------------------------------------------------

def expected_move(price, iv, t):
    """One-standard-deviation move = price * iv * sqrt(t) (iv decimal, t in years)."""
    price, iv, t = np.broadcast_arrays(np.asarray(price, float), np.asarray(iv, float),
                                       np.asarray(t, float))
    with np.errstate(invalid="ignore"):
        out = price * iv * np.sqrt(np.where(t >= 0, t, np.nan))
    out = np.where((price > 0) & (iv >= 0), out, np.nan)
    return float(out) if out.ndim == 0 else out


def straddle_move(call, put):
    """At-the-money straddle price (call + put) as the market's implied move.

    Under a lognormal model the ATM straddle is about 0.8 * S * vol * sqrt(t), i.e. the
    mean absolute move to expiry. Traders often quote ~0.85 x straddle as a rough
    one-standard-deviation range; that is a rule of thumb, not something this returns.
    """
    out = np.asarray(call, float) + np.asarray(put, float)
    return float(out) if out.ndim == 0 else out


def buildup(price_change, oi_change) -> np.ndarray:
    """Classify price/OI moves: 'long_buildup' (price up, OI up), 'short_buildup'
    (price down, OI up), 'short_covering' (price up, OI down), 'long_unwinding'
    (price down, OI down), else 'neutral' (either change zero or missing)."""
    p, o = np.broadcast_arrays(np.asarray(price_change, float), np.asarray(oi_change, float))
    out = np.full(p.shape, "neutral", dtype=object)
    out[(p > 0) & (o > 0)] = "long_buildup"
    out[(p < 0) & (o > 0)] = "short_buildup"
    out[(p > 0) & (o < 0)] = "short_covering"
    out[(p < 0) & (o < 0)] = "long_unwinding"
    return out


def gamma_exposure(forward, strike, t, vol, oi, r: float = 0.0):
    """Gamma exposure of one strike: Black-76 gamma x open interest x forward^2 x 0.01 — the change in
    the option holders' delta (in money) for a 1 % move in the forward. Gamma is the same for a call
    and a put of one strike, so no option type is needed. `oi` in units of the underlying
    (contracts x lot size). Array-friendly; bad inputs give NaN."""
    g = np.asarray(black76_greeks(forward, strike, t, vol, r, "C")["gamma"], dtype=float)
    f = np.asarray(forward, dtype=float)
    out = g * np.asarray(oi, dtype=float) * f * f * 0.01
    out = np.where(np.isfinite(out), out, np.nan)
    return float(out) if out.ndim == 0 else out


def parity_forward(strikes, call, put, t: float, r: float = 0.0) -> float:
    """Forward price implied by put-call parity, F = K + e^(rt) (C - P), read at the strike where
    |C - P| is smallest among strikes with both prices (the lower strike on a tie).
    NaN when no strike has both a call and a put price."""
    k = np.asarray(strikes, dtype=float)
    c = np.asarray(call, dtype=float)
    p = np.asarray(put, dtype=float)
    ok = np.isfinite(k) & np.isfinite(c) & np.isfinite(p)
    if not ok.any() or not np.isfinite(t) or not np.isfinite(r):
        return float("nan")
    k, c, p = k[ok], c[ok], p[ok]
    order = np.argsort(k, kind="stable")
    k, c, p = k[order], c[order], p[order]
    i = int(np.argmin(np.abs(c - p)))
    return float(k[i] + math.exp(r * t) * (c[i] - p[i]))
