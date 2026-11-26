"""Multi-leg option strategies: payoff, breakevens, max profit/loss, probability of profit.

A leg is a `Leg` or a dict with the same keys:

    kind        'C' call, 'P' put, 'F' future, 'S' stock (also 'CE'/'PE'/'call'/'put')
    strike      option strike; for 'F'/'S' the entry price
    premium     option premium paid per unit (ignored for 'F'/'S')
    qty         signed quantity: + long, - short
    multiplier  contract size (lot size), default 1

All P&L figures are per the whole position: qty * multiplier * (value - cost).
"""
from dataclasses import dataclass

import numpy as np

from ._normal import norm_cdf
from ._util import option_sign
from .pricing import black76_price


@dataclass
class Leg:
    kind: str
    strike: float
    premium: float = 0.0
    qty: float = 1.0
    multiplier: float = 1.0


def _legs(legs):
    """Normalise to arrays: kind code (+1 call, -1 put, 0 linear), strike, premium, size."""
    code, k, prem, size = [], [], [], []
    for leg in legs:
        d = leg if isinstance(leg, dict) else leg.__dict__
        kind = str(d.get("kind", "")).strip().upper()
        if kind in ("F", "S", "FUT", "FUTURE", "STOCK", "SPOT"):
            c = 0.0
        else:
            c = float(option_sign(kind))
        code.append(c)
        k.append(float(d.get("strike", np.nan)))
        prem.append(0.0 if c == 0.0 else float(d.get("premium", 0.0) or 0.0))
        size.append(float(d.get("qty", 1.0)) * float(d.get("multiplier", 1.0) or 1.0))
    return (np.array(code), np.array(k), np.array(prem), np.array(size))


def payoff_at_expiry(legs, prices):
    """Total P&L at expiry for each underlying price in `prices`."""
    code, k, prem, size = _legs(legs)
    p = np.asarray(prices, dtype=float)
    flat = np.atleast_1d(p)[:, None]
    val = np.where(code > 0, np.maximum(flat - k, 0.0) - prem,
                   np.where(code < 0, np.maximum(k - flat, 0.0) - prem, flat - k))
    val = np.where(np.isnan(code), np.nan, val)
    out = (val * size).sum(axis=1)
    return float(out[0]) if p.ndim == 0 else out.reshape(p.shape)


def _kinks(legs):
    code, k, _, size = _legs(legs)
    if np.isnan(code).any() or not np.isfinite(k).all() or not np.isfinite(size).all():
        return None
    pts = np.unique(np.concatenate([[0.0], k[(code != 0) & (k > 0)]]))
    slope_inf = float(np.sum(size[code >= 0]))   # calls, futures and stock keep rising
    return pts, slope_inf


def breakevens(legs) -> np.ndarray:
    """Underlying prices (>= 0) where the expiry P&L is exactly zero, sorted.

    The payoff is piecewise linear with kinks at the strikes, so roots are exact. Where
    the P&L is zero over a whole interval, both ends of the interval are returned.
    """
    kk = _kinks(legs)
    if kk is None:
        return np.array([np.nan])
    pts, slope_inf = kk
    vals = payoff_at_expiry(legs, pts)
    roots = list(pts[vals == 0.0])
    for a, b, va, vb in zip(pts[:-1], pts[1:], vals[:-1], vals[1:]):
        if va * vb < 0:
            roots.append(a + (b - a) * va / (va - vb))
    last, vl = pts[-1], vals[-1]
    if slope_inf != 0 and vl != 0 and np.sign(vl) != np.sign(slope_inf):
        roots.append(last - vl / slope_inf)
    return np.unique(np.round(np.array(roots, dtype=float), 12))


def max_profit(legs) -> float:
    """Largest P&L at expiry over all prices >= 0; inf if it grows without bound."""
    kk = _kinks(legs)
    if kk is None:
        return np.nan
    pts, slope_inf = kk
    if slope_inf > 0:
        return np.inf
    return float(np.max(payoff_at_expiry(legs, pts)))


def max_loss(legs) -> float:
    """Largest loss at expiry as a positive number (= -min P&L); inf if unbounded.

    Negative only for a position that makes money at every price.
    """
    kk = _kinks(legs)
    if kk is None:
        return np.nan
    pts, slope_inf = kk
    if slope_inf < 0:
        return np.inf
    return float(-np.min(payoff_at_expiry(legs, pts)))


def probability_of_profit(legs, forward, vol, t) -> float:
    """Probability that the expiry P&L is > 0 with a lognormal underlying whose mean is
    the forward (risk-neutral, no extra drift): ln S_T ~ N(ln F - vol^2 t / 2, vol^2 t)."""
    try:
        forward, vol, t = float(forward), float(vol), float(t)
    except (TypeError, ValueError):
        return np.nan
    if not (forward > 0 and vol > 0 and t > 0):
        return np.nan
    be = breakevens(legs)
    if np.isnan(be).any():
        return np.nan
    edges = np.unique(np.concatenate([[0.0], be[be > 0], [np.inf]]))
    sd = vol * np.sqrt(t)

    def cdf(x):
        if x <= 0:
            return 0.0
        if np.isinf(x):
            return 1.0
        return float(norm_cdf((np.log(x / forward) + 0.5 * sd * sd) / sd))

    prob = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        mid = (a + b) / 2 if np.isfinite(b) else (a + 1.0) * 2.0
        if payoff_at_expiry(legs, mid) > 0:
            prob += cdf(b) - cdf(a)
    return float(min(max(prob, 0.0), 1.0))


def value_now(legs, forward, vol, t, r: float = 0.0):
    """Mark-to-model P&L before expiry for each underlying level in `forward`.

    Options are valued with Black-76 (vol scalar or one per leg, t years remaining,
    r discount rate); futures/stock legs at forward - entry. Returns qty * multiplier *
    (value - premium) summed over legs; at t = 0 this equals payoff_at_expiry.
    """
    code, k, prem, size = _legs(legs)
    f = np.asarray(forward, dtype=float)
    flat = np.atleast_1d(f)[:, None]
    vols = np.broadcast_to(np.asarray(vol, dtype=float), code.shape)
    total = np.zeros(flat.shape[0])
    for i in range(code.size):
        if np.isnan(code[i]):
            return np.full(f.shape, np.nan) if f.ndim else np.nan
        if code[i] == 0:
            val = flat[:, 0] - k[i]
        else:
            val = np.atleast_1d(black76_price(flat[:, 0], k[i], t, vols[i], r,
                                              "C" if code[i] > 0 else "P")) - prem[i]
        total = total + size[i] * val
    return float(total[0]) if f.ndim == 0 else total.reshape(f.shape)
