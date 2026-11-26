"""Vectorised normal distribution helpers (no scipy).

numpy has no erf/erfc, so the complementary error function is computed with
W. J. Cody's rational Chebyshev approximations ("Rational Chebyshev approximations
for the error function", Math. Comp. 23 (1969) 631-637; coefficients as published
in the public-domain netlib SPECFUN routine CALERF). Three regions:

    |x| <= 0.46875        erf(x)  = x * P(x^2) / Q(x^2)
    0.46875 < |x| <= 4    erfc(x) = exp(-x^2) * P(x) / Q(x)
    |x| > 4               erfc(x) = exp(-x^2) / x * (1/sqrt(pi) + R(1/x^2) / x^2)

Relative accuracy is close to machine precision (measured worst case vs math.erfc
~2e-16 on |x| <= 26), and the whole thing is a handful of numpy array operations,
so 100k evaluations take about a millisecond.

The scaled form erfcx(x) = exp(x^2) erfc(x) is exposed too; the implied-volatility
solver uses it to evaluate far-out-of-the-money prices without underflow.
"""
import numpy as np

_SQRT2 = 1.4142135623730950488
_INV_SQRT2PI = 0.39894228040143267794
_INV_SQRTPI = 0.56418958354775628695

_A = (3.16112374387056560e00, 1.13864154151050156e02, 3.77485237685302021e02,
      3.20937758913846947e03, 1.85777706184603153e-1)
_B = (2.36012909523441209e01, 2.44024637934444173e02, 1.28261652607737228e03,
      2.84423683343917062e03)
_C = (5.64188496988670089e-1, 8.88314979438837594e00, 6.61191906371416295e01,
      2.98635138197400131e02, 8.81952221241769090e02, 1.71204761263407058e03,
      2.05107837782607147e03, 1.23033935479799725e03, 2.15311535474403846e-8)
_D = (1.57449261107098347e01, 1.17693950891312499e02, 5.37181101862009858e02,
      1.62138957456669019e03, 3.29079923573345963e03, 4.36261909014324716e03,
      3.43936767414372164e03, 1.23033935480374942e03)
_P = (3.05326634961232344e-1, 3.60344899949804439e-1, 1.25781726111229246e-1,
      1.60837851487422766e-2, 6.58749161529837803e-4, 1.63153871373020978e-2)
_Q = (2.56852019228982242e00, 1.87295284992346725e00, 5.27905102951428412e-1,
      6.05183413124413191e-2, 2.33520497626869185e-3)


def _erf_small(x):
    z = x * x
    num = _A[4] * z
    den = z
    for i in range(3):
        num = (num + _A[i]) * z
        den = (den + _B[i]) * z
    return x * (num + _A[3]) / (den + _B[3])


def _erfcx_mid(y):
    """exp(y^2) erfc(y) for 0.46875 < y <= 4."""
    num = _C[8] * y
    den = y
    for i in range(7):
        num = (num + _C[i]) * y
        den = (den + _D[i]) * y
    return (num + _C[7]) / (den + _D[7])


def _erfcx_large(y):
    """exp(y^2) erfc(y) for y > 4."""
    z = 1.0 / (y * y)
    num = _P[5] * z
    den = z
    for i in range(4):
        num = (num + _P[i]) * z
        den = (den + _Q[i]) * z
    r = z * (num + _P[4]) / (den + _Q[4])
    return (_INV_SQRTPI - r) / y


def _exp_neg_sq(y):
    """exp(-y^2) with Cody's split to keep full relative accuracy for large y."""
    ys = np.trunc(y * 16.0) / 16.0
    d = (y - ys) * (y + ys)
    return np.exp(-ys * ys) * np.exp(-d)


def erfcx(x):
    """Scaled complementary error function exp(x^2) * erfc(x), elementwise.

    For x < 0 this grows like 2 exp(x^2) and overflows to inf below about -26.6.
    """
    x = np.asarray(x, dtype=float)
    y = np.abs(x)
    out = np.empty_like(y)
    small = y <= 0.46875
    mid = (y > 0.46875) & (y <= 4.0)
    large = y > 4.0
    if small.any():
        ys = y[small]
        out[small] = np.exp(ys * ys) * (1.0 - _erf_small(ys))
    if mid.any():
        out[mid] = _erfcx_mid(y[mid])
    if large.any():
        out[large] = _erfcx_large(y[large])
    neg = (x < 0) & ~small
    if neg.any():
        xn = x[neg]
        with np.errstate(over="ignore"):
            out[neg] = 2.0 * np.exp(xn * xn) - out[neg]
    neg_small = (x < 0) & small
    if neg_small.any():
        xs = x[neg_small]
        out[neg_small] = np.exp(xs * xs) * (1.0 - _erf_small(xs))
    out[np.isnan(x)] = np.nan
    return out


def erfc(x):
    """Complementary error function, elementwise (relative error ~1e-16)."""
    x = np.asarray(x, dtype=float)
    y = np.abs(x)
    out = np.empty_like(y)
    small = y <= 0.46875
    mid = (y > 0.46875) & (y <= 4.0)
    large = (y > 4.0) & (y < 27.3)
    huge = y >= 27.3
    if small.any():
        out[small] = 1.0 - _erf_small(x[small])
    if mid.any():
        ym = y[mid]
        out[mid] = _exp_neg_sq(ym) * _erfcx_mid(ym)
    if large.any():
        yl = y[large]
        out[large] = _exp_neg_sq(yl) * _erfcx_large(yl)
    out[huge] = 0.0
    neg = (x < 0) & ~small
    out[neg] = 2.0 - out[neg]
    out[np.isnan(x)] = np.nan
    return out


def norm_cdf(x):
    """Standard normal CDF, elementwise, accurate in relative terms deep into both tails."""
    return 0.5 * erfc(-np.asarray(x, dtype=float) / _SQRT2)


def norm_pdf(x):
    """Standard normal density, elementwise."""
    x = np.asarray(x, dtype=float)
    return _INV_SQRT2PI * np.exp(-0.5 * x * x)
