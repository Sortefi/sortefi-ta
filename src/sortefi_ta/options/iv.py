"""Implied volatility for Black-76 and Black-Scholes-Merton (vectorised, numpy only).

Method
------
Every option is first reduced to Jaeckel's normalised form ("Let's Be Rational",
P. Jaeckel, Wilmott 2015): x = ln(F/K), beta = price / (discount * sqrt(F K)). An
in-the-money option is swapped for its out-of-the-money twin by put-call parity, so
the solver only ever sees an out-of-the-money call with x <= 0 and

    b(x, s) = e^(x/2) N(x/s + s/2) - e^(-x/2) N(x/s - s/2),     s = vol * sqrt(t).

b is increasing in s with one inflection point at s_c = sqrt(-2x). Following the paper's
split into two branches:

* lower branch (beta <= b(x, s_c)): Newton on ln b(s) - ln beta. b is evaluated with the
  scaled complementary error function, so tiny far-out-of-the-money prices keep full
  relative accuracy instead of cancelling to zero;
* upper branch: Newton on ln(e^(x/2) - b(s)) - ln(e^(x/2) - beta), where
  e^(x/2) - b = e^(x/2) N(-d1) + e^(-x/2) N(d2) is computed without cancellation.

Each step is safeguarded by a bracket [lo, hi] that always contains the root; a step
that would leave it is replaced by bisection. Iteration stops when the relative step is
below 1e-14. This is not a port of Jaeckel's code (no rational-cubic initial guess or
Householder steps), just the same normalisation and branch split, so it needs a few more
iterations but stays short and dependency-free.

Prices at or below intrinsic value, at or above the no-arbitrage upper bound
(discounted forward for a call, discounted strike for a put), t <= 0, or any
non-positive / non-finite input return NaN.
"""
import numpy as np

from ._normal import erfcx, norm_cdf
from ._util import as_float, option_sign, unwrap

_SQRT2 = 1.4142135623730950488
_LOG_SQRT2PI = 0.91893853320467274178
_MAX_ITER = 100
_REL_TOL = 1e-14


def _ninv(p):
    """Inverse standard normal CDF (Acklam's approximation, |rel err| < 1.2e-9).

    Only used for the solver's starting point, so its accuracy is not critical.
    """
    p = np.asarray(p, dtype=float)
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    out = np.full_like(p, np.nan)
    lo = (p > 0) & (p < 0.02425)
    hi = (p < 1) & (p > 1 - 0.02425)
    mid = (p >= 0.02425) & (p <= 1 - 0.02425)
    if lo.any():
        q = np.sqrt(-2 * np.log(p[lo]))
        out[lo] = ((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                   / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1))
    if hi.any():
        q = np.sqrt(-2 * np.log(1 - p[hi]))
        out[hi] = -((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                    / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1))
    if mid.any():
        q = p[mid] - 0.5
        r = q * q
        out[mid] = ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
                    / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1))
    return out


def _log_b(x, s):
    """ln b(x, s) for x <= 0, s > 0 (normalised out-of-the-money call)."""
    d1 = x / s + 0.5 * s
    d2 = d1 - s
    out = np.empty_like(s)
    tail = d1 <= 0
    if tail.any():
        xt, d1t, d2t = x[tail], d1[tail], d2[tail]
        diff = erfcx(-d1t / _SQRT2) - erfcx(-d2t / _SQRT2)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[tail] = -0.5 * xt - 0.5 * d2t * d2t + np.log(0.5 * diff)
    body = ~tail
    if body.any():
        xb, d1b, d2b = x[body], d1[body], d2[body]
        val = np.exp(0.5 * xb) * norm_cdf(d1b) - np.exp(-0.5 * xb) * norm_cdf(d2b)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[body] = np.log(val)
    return out


def _log_u(x, s):
    """ln(e^(x/2) - b(x, s)) = ln(e^(x/2) N(-d1) + e^(-x/2) N(d2)), no cancellation."""
    d1 = x / s + 0.5 * s
    d2 = d1 - s
    val = np.exp(0.5 * x) * norm_cdf(-d1) + np.exp(-0.5 * x) * norm_cdf(d2)
    with np.errstate(divide="ignore"):
        return np.log(val)


def _log_vega(x, s):
    """ln of db/ds = e^(x/2) phi(d1) = exp(-x/2 - d2^2/2) / sqrt(2 pi)."""
    d2 = x / s - 0.5 * s
    return -0.5 * x - 0.5 * d2 * d2 - _LOG_SQRT2PI


def _solve_normalised(x, beta):
    """Solve b(x, s) = beta for s; x <= 0 and 0 < beta < e^(x/2) are guaranteed."""
    n = x.size
    s_c = np.sqrt(-2.0 * x)
    with np.errstate(divide="ignore"):
        log_beta = np.log(beta)
        b_c = np.where(s_c > 0, np.exp(_log_b(x, np.where(s_c > 0, s_c, 1.0))), 0.0)
    lower = (s_c > 0) & (beta <= b_c)
    upper_bound = np.exp(0.5 * x)
    with np.errstate(divide="ignore"):
        log_u_target = np.log(upper_bound - beta)

    # starting points
    s = np.empty(n)
    lo = np.zeros(n)
    hi = np.full(n, np.inf)
    if lower.any():
        xl = x[lower]
        guess = -xl / np.sqrt(np.maximum(-2.0 * log_beta[lower], 1e-300))
        s[lower] = np.clip(guess, 1e-3 * s_c[lower], s_c[lower])
        hi[lower] = s_c[lower]
    up = ~lower
    if up.any():
        xu = x[up]
        frac = (upper_bound[up] - beta[up]) / (np.exp(0.5 * xu) + np.exp(-0.5 * xu))
        guess = -2.0 * _ninv(np.clip(frac, 1e-300, 0.5))
        guess = np.where(np.isfinite(guess), guess, 1.0)
        s[up] = np.maximum(guess, s_c[up])
        lo[up] = s_c[up]
        s[up] = np.where(s[up] > 0, s[up], 1e-8)

    active = np.ones(n, dtype=bool)
    for _ in range(_MAX_ITER):
        idx = np.nonzero(active)[0]
        if idx.size == 0:
            break
        xs, ss, low = x[idx], s[idx], lower[idx]
        lv = _log_vega(xs, ss)
        step = np.empty(idx.size)
        too_low = np.empty(idx.size, dtype=bool)
        if low.any():
            lb = _log_b(xs[low], ss[low])
            g = lb - log_beta[idx][low]
            step[low] = -g / np.exp(lv[low] - lb)
            too_low[low] = g < 0
        if (~low).any():
            lu = _log_u(xs[~low], ss[~low])
            h = lu - log_u_target[idx][~low]
            step[~low] = h / np.exp(lv[~low] - lu)
            too_low[~low] = h > 0
        lo_i, hi_i = lo[idx], hi[idx]
        lo_i = np.where(too_low, np.maximum(lo_i, ss), lo_i)
        hi_i = np.where(too_low, hi_i, np.minimum(hi_i, ss))
        new = ss + step
        outside = ~np.isfinite(new) | (new <= lo_i) | (new >= hi_i)
        bisect = np.where(np.isfinite(hi_i), 0.5 * (lo_i + hi_i), 2.0 * np.maximum(ss, lo_i))
        new = np.where(outside, bisect, new)
        done = (np.abs(new - ss) <= _REL_TOL * np.abs(new)) | (hi_i - lo_i <= _REL_TOL * hi_i)
        s[idx], lo[idx], hi[idx] = new, lo_i, hi_i
        active[idx[done]] = False
    s[active] = np.nan  # did not converge (should not happen for valid inputs)
    return s


def implied_vol_black76(price, forward, strike, t, r=0.0, opt_type="C"):
    """Implied volatility (decimal, per year) from a Black-76 option price.

    price is the option premium in the same currency as forward/strike; r discounts it
    (r = 0 for an undiscounted / margined premium). Returns NaN where no volatility
    reproduces the price (see module doc).
    """
    price, f, k, t, r, sign = as_float(price, forward, strike, t, r, option_sign(opt_type))
    shape = price.shape
    price, f, k, t, r, sign = (a.ravel() for a in (price, f, k, t, r, sign))
    out = np.full(price.size, np.nan)
    ok = (np.isfinite(price) & np.isfinite(f) & np.isfinite(k) & np.isfinite(t)
          & np.isfinite(r) & np.isfinite(sign) & (price > 0) & (f > 0) & (k > 0) & (t > 0))
    if ok.any():
        p, fo, ko, to, ro, th = price[ok], f[ok], k[ok], t[ok], r[ok], sign[ok]
        root = np.sqrt(fo * ko)
        beta = p * np.exp(ro * to) / root
        x = np.log(fo / ko)
        intrinsic = np.maximum(th * (np.exp(0.5 * x) - np.exp(-0.5 * x)), 0.0)
        cap = np.where(th > 0, np.exp(0.5 * x), np.exp(-0.5 * x))
        valid = (beta > intrinsic) & (beta < cap)
        beta_otm = beta - intrinsic        # ITM -> OTM twin via put-call parity
        x_otm = -np.abs(x)                 # OTM call with x <= 0 (put(x) == call(-x))
        vol = np.full(p.size, np.nan)
        if valid.any():
            s = _solve_normalised(x_otm[valid], beta_otm[valid])
            vol[valid] = s / np.sqrt(to[valid])
        out[ok] = vol
    return unwrap(out.reshape(shape))


def implied_vol_bsm(price, spot, strike, t, r=0.0, q=0.0, opt_type="C"):
    """Implied volatility from a Black-Scholes-Merton price (spot S, rate r, yield q)."""
    spot, t, r, q = as_float(spot, t, r, q)
    with np.errstate(over="ignore", invalid="ignore"):
        fwd = spot * np.exp((r - q) * t)
    return implied_vol_black76(price, fwd, strike, t, r, opt_type)


def implied_vol_nse(price, spot, strike, t, opt_type="C"):
    """BSM implied vol with r = 10 %, q = 0 — the convention in NSE's option-chain footnote.

    Useful for reproducing the IV column NSE publishes; for analysis prefer Black-76 on
    the futures price (or the put-call-parity forward).
    """
    return implied_vol_bsm(price, spot, strike, t, 0.10, 0.0, opt_type)
