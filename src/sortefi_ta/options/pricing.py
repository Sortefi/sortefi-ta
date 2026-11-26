"""Black-76 and Black-Scholes-Merton option prices and greeks (vectorised).

Units (the same everywhere in this package):

    vol     decimal per year (0.20 = 20 %)
    t       years to expiry (calendar days / 365 is the usual choice)
    r, q    continuously compounded annual rates as decimals (0.05 = 5 %)
    delta   change in option price per 1 unit move of the underlying (forward for
            Black-76, spot for BSM)
    gamma   change in delta per 1 unit move of the underlying
    vega    change in price per 1 vol POINT (vol 0.20 -> 0.21), i.e. dV/dvol / 100
    theta   change in price per 1 CALENDAR DAY passing, i.e. dV/dt / 365 (negative
            for a long option, as usually quoted)
    rho     change in price per 1 % move in r, i.e. dV/dr / 100

Every argument broadcasts (scalars, numpy arrays, pandas Series). opt_type takes
'C'/'P', 'CE'/'PE', 'call'/'put', True/False or +1/-1. Bad inputs (non-positive
forward/spot/strike, negative vol or time, unknown option type, NaN) give NaN rather
than an exception. At t == 0 or vol == 0 the price is the (discounted) intrinsic value.
"""
import numpy as np

from ._normal import norm_cdf, norm_pdf
from ._util import as_float, option_sign, unwrap


def _prepare(f, k, t, vol, r, opt_type):
    f, k, t, vol, r, sign = as_float(f, k, t, vol, r, option_sign(opt_type))
    bad = ~((f > 0) & (k > 0) & (t >= 0) & (vol >= 0) & np.isfinite(r)
            & np.isfinite(f) & np.isfinite(k) & np.isfinite(t) & np.isfinite(vol)
            & np.isfinite(sign))
    sqrt_t = np.sqrt(np.where(t > 0, t, 0.0))
    s = vol * sqrt_t
    live = (s > 0) & ~bad
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = np.where(live, np.log(f / k) / np.where(live, s, 1.0) + 0.5 * s, np.nan)
    d2 = d1 - s
    return f, k, t, vol, r, sign, bad, sqrt_t, s, live, d1, d2


def _black_core(f, k, t, vol, r, opt_type):
    """Discounted Black-76 price and the pieces greeks need."""
    f, k, t, vol, r, sign, bad, sqrt_t, s, live, d1, d2 = _prepare(f, k, t, vol, r, opt_type)
    disc = np.exp(-r * t)
    nd1 = norm_cdf(sign * d1)
    nd2 = norm_cdf(sign * d2)
    with np.errstate(invalid="ignore"):
        undisc = np.where(live, sign * (f * nd1 - k * nd2),
                          np.maximum(sign * (f - k), 0.0))
    price = np.where(bad, np.nan, disc * undisc)
    return dict(f=f, k=k, t=t, vol=vol, r=r, sign=sign, bad=bad, sqrt_t=sqrt_t, s=s,
                live=live, d1=d1, d2=d2, disc=disc, nd1=nd1, nd2=nd2, price=price)


def black76_price(forward, strike, t, vol, r=0.0, opt_type="C"):
    """Black-76 price of a European option on a forward/future.

    price = e^(-r t) * [F N(d1) - K N(d2)]       (call)
            e^(-r t) * [K N(-d2) - F N(-d1)]     (put)
    d1 = (ln(F/K) + vol^2 t / 2) / (vol sqrt t),   d2 = d1 - vol sqrt t

    Use r = 0 for an undiscounted (forward / margined) premium.
    """
    return unwrap(_black_core(forward, strike, t, vol, r, opt_type)["price"])


def black76_greeks(forward, strike, t, vol, r=0.0, opt_type="C"):
    """Black-76 greeks: {"price", "delta", "gamma", "vega", "theta", "rho"}.

    delta/gamma are with respect to the forward F. rho holds F fixed (dV/dr = -t V),
    which is the right sensitivity for options on futures. Units as in the module doc.
    """
    c = _black_core(forward, strike, t, vol, r, opt_type)
    f, k, t, vol, r, sign, bad, sqrt_t, s, live, d1, disc, price = (
        c["f"], c["k"], c["t"], c["vol"], c["r"], c["sign"], c["bad"], c["sqrt_t"],
        c["s"], c["live"], c["d1"], c["disc"], c["price"])
    pdf = np.where(live, norm_pdf(np.where(live, d1, 0.0)), 0.0)
    itm = np.where(sign * (f - k) > 0, 1.0, np.where(f == k, 0.5, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(live, sign * disc * c["nd1"], sign * disc * itm)
        gamma = np.where(live, disc * pdf / (f * np.where(live, s, 1.0)), 0.0)
        vega = disc * f * pdf * sqrt_t
        decay = np.where(live, -disc * f * pdf * vol / (2.0 * np.where(live, sqrt_t, 1.0)), 0.0)
    theta = decay + r * price
    rho = -t * price
    out = {"price": price, "delta": delta, "gamma": gamma, "vega": vega / 100.0,
           "theta": theta / 365.0, "rho": rho / 100.0}
    return {key: unwrap(np.where(bad, np.nan, v)) for key, v in out.items()}


def bsm_price(spot, strike, t, vol, r=0.0, q=0.0, opt_type="C"):
    """Black-Scholes-Merton price with continuous dividend yield / carry q.

    Equivalent to Black-76 on the forward F = S e^((r - q) t) discounted at r.
    """
    spot, t, r, q = as_float(spot, t, r, q)
    with np.errstate(over="ignore", invalid="ignore"):
        fwd = spot * np.exp((r - q) * t)
    price = np.asarray(_black_core(fwd, strike, t, vol, r, opt_type)["price"])
    return unwrap(np.where(np.isfinite(q), price, np.nan))


def bsm_greeks(spot, strike, t, vol, r=0.0, q=0.0, opt_type="C"):
    """Black-Scholes-Merton greeks: {"price", "delta", "gamma", "vega", "theta", "rho"}.

    delta/gamma are with respect to spot S. rho is dV/dr with S and q fixed (per 1 %).
    Units as in the module doc.
    """
    spot, t, r, q = as_float(spot, t, r, q)
    with np.errstate(over="ignore", invalid="ignore"):
        fwd = spot * np.exp((r - q) * t)
    c = _black_core(fwd, strike, t, vol, r, opt_type)
    k, sign, bad, sqrt_t, s, live, d1, disc, price = (
        c["k"], c["sign"], c["bad"], c["sqrt_t"], c["s"], c["live"], c["d1"], c["disc"],
        c["price"])
    t, vol, r = c["t"], c["vol"], c["r"]
    spot, q = np.broadcast_arrays(spot, q, *([price] if price.ndim else []))[:2]
    bad = bad | ~np.isfinite(q)
    yld = np.exp(-q * t)
    pdf = np.where(live, norm_pdf(np.where(live, d1, 0.0)), 0.0)
    itm = np.where(sign * (c["f"] - k) > 0, 1.0, np.where(c["f"] == k, 0.5, 0.0))
    nd1 = np.where(live, c["nd1"], itm)
    nd2 = np.where(live, c["nd2"], itm)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = sign * yld * nd1
        gamma = np.where(live, yld * pdf / (spot * np.where(live, s, 1.0)), 0.0)
        vega = spot * yld * pdf * sqrt_t
        decay = np.where(live, -spot * yld * pdf * vol / (2.0 * np.where(live, sqrt_t, 1.0)), 0.0)
    theta = decay - sign * r * k * disc * nd2 + sign * q * spot * yld * nd1
    rho = sign * k * t * disc * nd2
    out = {"price": price, "delta": delta, "gamma": gamma, "vega": vega / 100.0,
           "theta": theta / 365.0, "rho": rho / 100.0}
    return {key: unwrap(np.where(bad, np.nan, v)) for key, v in out.items()}
