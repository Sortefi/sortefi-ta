"""Performance and risk statistics over a return (or price) series.

    from sortefi_ta import performance as pf
    r = pf.simple_returns(df["close"])
    pf.sharpe_ratio(r, risk_free=0.05, periods_per_year=252)

Unlike the calc_* indicators (one value per candle), these return ONE number for the whole series.

Conventions:
  - numpy / pandas only.
  - `risk_free` / `required_return` are ANNUAL rates, converted to per-period internally.
  - Standard deviations are sample (ddof=1); annualisation uses `periods_per_year`
    (252 for daily stock candles, 365 for daily crypto, 52 weekly, 12 monthly ...).
  - NaNs in the input are dropped. Too-short or zero-variance input returns None — never raises.
"""
from __future__ import annotations

import math

import numpy as np

from sortefi_ta.options._normal import norm_cdf, norm_pdf

__all__ = [
    "simple_returns", "log_returns", "cumulative_return", "total_return",
    "annualized_return", "annualized_volatility", "cagr",
    "sharpe_ratio", "sortino_ratio", "omega_ratio", "information_ratio", "tail_ratio",
    "downside_deviation", "value_at_risk", "conditional_value_at_risk",
    "win_rate", "skewness", "kurtosis",
    "beta", "alpha", "correlation",
]

VAR_METHODS = ("hist", "normal")


# ── internal helpers ────────────────────────────────────────────────────────

def _arr(x):
    """1-D float array (NaNs kept), or None when empty."""
    if x is None:
        return None
    a = np.asarray(x, dtype="float64").ravel()
    return a if a.size else None


def _clean(x):
    """1-D float array with NaN / inf dropped, or None when nothing is left."""
    a = _arr(x)
    if a is None:
        return None
    a = a[np.isfinite(a)]
    return a if a.size else None


def _align(a, b):
    """Two return series, trimmed to the same (latest) length, pairs with any NaN dropped."""
    x, y = _arr(a), _arr(b)
    if x is None or y is None:
        return None, None
    n = min(len(x), len(y))
    x, y = x[-n:], y[-n:]
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    return (x, y) if x.size else (None, None)


def _norm_ppf(p: float) -> float:
    """Inverse of the standard normal CDF (Acklam's rational approximation, then two Newton steps)."""
    if not 0.0 < p < 1.0:
        return math.nan
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)
    lo = 0.02425
    if p < lo:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    elif p <= 1 - lo:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    for _ in range(2):
        x -= (float(norm_cdf(x)) - p) / float(norm_pdf(x))
    return x


def _var_method(method) -> str:
    m = str(method or "hist").lower()
    m = {"historical": "hist", "gaussian": "normal", "parametric": "normal"}.get(m, m)
    if m not in VAR_METHODS:
        raise ValueError(f"method {method!r} — one of {VAR_METHODS}")
    return m


# ── returns ─────────────────────────────────────────────────────────────────

def simple_returns(prices):
    """Close-to-close simple returns from a price series (one shorter than the input)."""
    p = _arr(prices)
    if p is None or len(p) < 2:
        return np.array([])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.diff(p) / p[:-1]


def log_returns(prices):
    """Close-to-close log returns from a price series (a price <= 0 gives NaN)."""
    p = _arr(prices)
    if p is None or len(p) < 2:
        return np.array([])
    return np.diff(np.log(np.where(p > 0, p, np.nan)))


def cumulative_return(returns):
    """Total compounded return: prod(1 + r) - 1, as a fraction."""
    r = _clean(returns)
    if r is None:
        return None
    return float(np.prod(1.0 + r) - 1.0)


def total_return(prices):
    """last / first - 1 of a price series, as a fraction."""
    p = _clean(prices)
    if p is None or len(p) < 2 or p[0] == 0:
        return None
    return float(p[-1] / p[0] - 1.0)


def annualized_return(returns, periods_per_year=252):
    """Geometric annualised return of a return series, as a fraction."""
    r = _clean(returns)
    if r is None:
        return None
    cum = float(np.prod(1.0 + r))
    if cum <= 0:
        return -1.0
    return float(cum ** (periods_per_year / len(r)) - 1.0)


def annualized_volatility(returns, periods_per_year=252):
    """Sample standard deviation of returns x sqrt(periods_per_year)."""
    r = _clean(returns)
    if r is None or len(r) < 2:
        return None
    return float(np.std(r, ddof=1) * np.sqrt(periods_per_year))


def cagr(prices, periods_per_year=252):
    """Compound annual growth rate of a price series (one candle = 1 / periods_per_year of a year)."""
    p = _clean(prices)
    if p is None or len(p) < 2 or p[0] <= 0:
        return None
    if p[-1] <= 0:
        return -1.0
    return float((p[-1] / p[0]) ** (periods_per_year / (len(p) - 1)) - 1.0)


# ── risk-adjusted ratios ────────────────────────────────────────────────────

def sharpe_ratio(returns, risk_free=0.0, periods_per_year=252):
    """Annualised Sharpe ratio. `risk_free` is an annual rate."""
    r = _clean(returns)
    if r is None or len(r) < 2:
        return None
    excess = r - risk_free / periods_per_year
    sd = np.std(excess, ddof=1)
    if sd == 0:
        return None
    return float(np.mean(excess) / sd * np.sqrt(periods_per_year))


def sortino_ratio(returns, required_return=0.0, periods_per_year=252):
    """Annualised Sortino ratio (only returns below the required return count as risk)."""
    r = _clean(returns)
    if r is None:
        return None
    adj = r - required_return / periods_per_year
    downside = np.sqrt(np.mean(np.minimum(adj, 0.0) ** 2))
    if downside == 0:
        return None
    return float(np.mean(adj) / downside * np.sqrt(periods_per_year))


def omega_ratio(returns, required_return=0.0, periods_per_year=252):
    """Omega ratio: gains above the (annual) threshold return divided by losses below it."""
    r = _clean(returns)
    if r is None:
        return None
    if periods_per_year == 1:
        thresh = required_return
    else:
        if required_return <= -1:
            return None
        thresh = (1.0 + required_return) ** (1.0 / periods_per_year) - 1.0
    diff = r - thresh
    losses = -diff[diff < 0].sum()
    if losses == 0:
        return None
    return float(diff[diff > 0].sum() / losses)


def information_ratio(returns, benchmark_returns):
    """Mean active return / tracking error, per period (not annualised)."""
    r, b = _align(returns, benchmark_returns)
    if r is None or len(r) < 2:
        return None
    active = r - b
    te = np.std(active, ddof=1)
    if te == 0:
        return None
    return float(np.mean(active) / te)


def tail_ratio(returns):
    """|95th percentile| / |5th percentile| of returns."""
    r = _clean(returns)
    if r is None:
        return None
    p5 = np.percentile(r, 5)
    if p5 == 0:
        return None
    return float(abs(np.percentile(r, 95)) / abs(p5))


# ── downside risk ───────────────────────────────────────────────────────────

def downside_deviation(returns, required_return=0.0, periods_per_year=252):
    """Annualised downside deviation below an annual required return."""
    r = _clean(returns)
    if r is None:
        return None
    adj = r - required_return / periods_per_year
    return float(np.sqrt(np.mean(np.minimum(adj, 0.0) ** 2)) * np.sqrt(periods_per_year))


def value_at_risk(returns, level=0.05, method="hist"):
    """Value at risk: the `level` quantile of returns (a loss is negative).
    method "hist" = the returns' own quantile; "normal" = mean + sample std x the normal quantile."""
    r = _clean(returns)
    if r is None:
        return None
    if _var_method(method) == "hist":
        return float(np.percentile(r, 100 * level))
    if r.size < 2:
        return None
    return float(np.mean(r) + np.std(r, ddof=1) * _norm_ppf(level))


def conditional_value_at_risk(returns, level=0.05, method="hist"):
    """Expected shortfall: the average return in the worst `level` share of periods.
    method "hist" = mean of the worst returns; "normal" = mean - std x pdf(z) / level."""
    r = _clean(returns)
    if r is None:
        return None
    if _var_method(method) == "hist":
        idx = int((len(r) - 1) * level)
        return float(np.mean(np.partition(r, idx)[:idx + 1]))
    if r.size < 2:
        return None
    z = _norm_ppf(level)
    return float(np.mean(r) - np.std(r, ddof=1) * float(norm_pdf(z)) / level)


# ── distribution ────────────────────────────────────────────────────────────

def win_rate(returns):
    """Share of strictly positive returns, as a fraction."""
    r = _clean(returns)
    if r is None:
        return None
    return float(np.mean(r > 0))


def skewness(returns):
    """Skewness of returns (population form, as scipy.stats.skew with bias=True)."""
    r = _clean(returns)
    if r is None or len(r) < 2:
        return None
    d = r - np.mean(r)
    m2 = np.mean(d ** 2)
    if m2 == 0:
        return None
    return float(np.mean(d ** 3) / m2 ** 1.5)


def kurtosis(returns):
    """Excess kurtosis of returns (as scipy.stats.kurtosis with fisher=True, bias=True)."""
    r = _clean(returns)
    if r is None or len(r) < 2:
        return None
    d = r - np.mean(r)
    m2 = np.mean(d ** 2)
    if m2 == 0:
        return None
    return float(np.mean(d ** 4) / m2 ** 2 - 3.0)


# ── against a benchmark ─────────────────────────────────────────────────────

def beta(returns, benchmark_returns):
    """CAPM beta: cov(returns, benchmark) / var(benchmark)."""
    r, b = _align(returns, benchmark_returns)
    if r is None or len(r) < 2:
        return None
    var_b = np.var(b, ddof=1)
    if var_b == 0:
        return None
    return float(np.cov(r, b, ddof=1)[0, 1] / var_b)


def alpha(returns, benchmark_returns, risk_free=0.0, periods_per_year=252):
    """Annualised Jensen's alpha. `risk_free` is an annual rate."""
    r, b = _align(returns, benchmark_returns)
    if r is None or len(r) < 2:
        return None
    bta = beta(r, b)
    if bta is None:
        return None
    rf = risk_free / periods_per_year
    return float((np.mean((r - rf) - bta * (b - rf)) + 1.0) ** periods_per_year - 1.0)


def correlation(returns, benchmark_returns):
    """Pearson correlation of two return series."""
    r, b = _align(returns, benchmark_returns)
    if r is None or len(r) < 2 or np.std(r) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(r, b)[0, 1])
