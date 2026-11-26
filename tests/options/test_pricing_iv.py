import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sortefi_ta import options as o  # noqa: E402
from sortefi_ta.options._normal import erfc, erfcx, norm_cdf  # noqa: E402

FIX = Path(__file__).with_name("fixtures")


# -- normal distribution ---------------------------------------------------

def test_erfc_matches_math_erfc():
    x = np.linspace(-6, 27, 20001)
    ref = np.array([math.erfc(v) for v in x])
    m = ref > 0
    assert np.max(np.abs(erfc(x)[m] - ref[m]) / ref[m]) < 2e-15


def test_erfcx_and_cdf_basics():
    assert erfcx(0.0) == pytest.approx(1.0, abs=1e-16)
    assert norm_cdf(0.0) == 0.5
    assert norm_cdf(-10.0) == pytest.approx(7.619853024160527e-24, rel=1e-13)
    assert np.isnan(erfc(np.nan))


# -- textbook prices -------------------------------------------------------

def test_hull_example():
    # Hull, Options Futures and Other Derivatives: S=42, K=40, r=10%, vol=20%, T=0.5
    assert o.bsm_price(42, 40, 0.5, 0.2, 0.10, 0.0, "C") == pytest.approx(4.7594, abs=1e-4)
    assert o.bsm_price(42, 40, 0.5, 0.2, 0.10, 0.0, "P") == pytest.approx(0.8086, abs=1e-4)


def test_black76_hull_futures_example():
    # Hull: futures 20, K 20, r 9%, vol 25%, T 4/12 -> put 1.12
    assert o.black76_price(20, 20, 4 / 12, 0.25, 0.09, "P") == pytest.approx(1.12, abs=5e-3)


@pytest.mark.parametrize("r,q", [(0.0, 0.0), (0.05, 0.0), (0.07, 0.03)])
def test_put_call_parity(r, q):
    k = np.linspace(50, 200, 31)
    for t in (0.01, 0.5, 2.0):
        c = o.bsm_price(100, k, t, 0.3, r, q, "C")
        p = o.bsm_price(100, k, t, 0.3, r, q, "P")
        np.testing.assert_allclose(c - p, 100 * np.exp(-q * t) - k * np.exp(-r * t), atol=1e-11)


def test_reference_bsm_greeks_quantlib():
    ref = json.loads((FIX / "reference_values.json").read_text())
    for c in ref["bsm_greeks"]:
        g = o.bsm_greeks(c["spot"], c["strike"], c["t"], c["vol"], c["r"], c["q"], c["opt_type"])
        for key in ("price", "delta", "gamma", "vega", "theta", "rho"):
            assert g[key] == pytest.approx(c[key], rel=1e-8, abs=1e-10), key


# -- greeks vs finite differences -----------------------------------------

@pytest.mark.parametrize("typ", ["C", "P"])
@pytest.mark.parametrize("k", [70.0, 100.0, 130.0])
def test_black76_greeks_finite_difference(typ, k):
    f, t, v, r = 100.0, 0.4, 0.35, 0.04
    g = o.black76_greeks(f, k, t, v, r, typ)
    p = lambda **kw: o.black76_price(kw.get("f", f), k, kw.get("t", t), kw.get("v", v), kw.get("r", r), typ)
    h = 1e-3
    assert g["delta"] == pytest.approx((p(f=f + h) - p(f=f - h)) / (2 * h), abs=1e-7)
    assert g["gamma"] == pytest.approx((p(f=f + h) - 2 * p() + p(f=f - h)) / h ** 2, abs=1e-5)
    assert g["vega"] == pytest.approx((p(v=v + 1e-5) - p(v=v - 1e-5)) / 2e-5 / 100, abs=1e-8)
    assert g["theta"] == pytest.approx(-(p(t=t + 1e-5) - p(t=t - 1e-5)) / 2e-5 / 365, abs=1e-8)
    assert g["rho"] == pytest.approx((p(r=r + 1e-6) - p(r=r - 1e-6)) / 2e-6 / 100, abs=1e-8)


@pytest.mark.parametrize("typ", ["C", "P"])
def test_bsm_greeks_finite_difference(typ):
    s, k, t, v, r, q = 100.0, 95.0, 0.7, 0.25, 0.06, 0.02
    g = o.bsm_greeks(s, k, t, v, r, q, typ)
    p = lambda **kw: o.bsm_price(kw.get("s", s), k, kw.get("t", t), kw.get("v", v), kw.get("r", r), q, typ)
    h = 1e-3
    assert g["delta"] == pytest.approx((p(s=s + h) - p(s=s - h)) / (2 * h), abs=1e-7)
    assert g["gamma"] == pytest.approx((p(s=s + h) - 2 * p() + p(s=s - h)) / h ** 2, abs=1e-5)
    assert g["vega"] == pytest.approx((p(v=v + 1e-5) - p(v=v - 1e-5)) / 2e-5 / 100, abs=1e-8)
    assert g["theta"] == pytest.approx(-(p(t=t + 1e-5) - p(t=t - 1e-5)) / 2e-5 / 365, abs=1e-8)
    assert g["rho"] == pytest.approx((p(r=r + 1e-6) - p(r=r - 1e-6)) / 2e-6 / 100, abs=1e-8)


def test_expiry_and_zero_vol_give_intrinsic():
    assert o.black76_price(110, 100, 0.0, 0.3, 0.0, "C") == 10.0
    assert o.black76_price(90, 100, 0.0, 0.3, 0.0, "C") == 0.0
    assert o.black76_price(90, 100, 1.0, 0.0, 0.0, "P") == pytest.approx(10.0)


def test_bad_inputs_are_nan_not_errors():
    bad = o.black76_greeks([-1, 100, 100, 100, 100, np.nan], [100, 0, 100, 100, 100, 100],
                           [1, 1, -1, 1, 1, 1], [0.2, 0.2, 0.2, -0.1, 0.2, 0.2], 0.0,
                           ["C", "C", "C", "C", "X", "C"])
    for v in bad.values():
        assert np.isnan(v).all()
    assert np.isnan(o.bsm_price("abc", 100, 1, 0.2))


def test_option_type_aliases():
    a = o.black76_price(100, 90, 1, 0.2, 0, ["C", "CE", "call", True, 1])
    assert np.ptp(a) == 0
    b = o.black76_price(100, 90, 1, 0.2, 0, ["P", "PE", "put", False, -1])
    assert np.ptp(b) == 0


# -- implied volatility ----------------------------------------------------

def _grid():
    m = np.geomspace(0.5, 2.0, 21)
    t = np.array([1, 7, 30, 91, 365, 730]) / 365.0
    v = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0])
    M, T, V, S = np.meshgrid(m, t, v, [1.0, -1.0], indexing="ij")
    return 100.0, 100 * M.ravel(), T.ravel(), V.ravel(), np.where(S.ravel() > 0, "C", "P")


def test_iv_round_trip_grid():
    f, k, t, v, typ = _grid()
    price = o.black76_price(f, k, t, v, 0.0, typ)
    iv = o.implied_vol_black76(price, f, k, t, 0.0, typ)
    tv = price - np.maximum(np.where(typ == "C", 1, -1) * (f - k), 0)
    ident = tv > 1e-12 * f
    assert not np.isnan(iv[ident]).any()
    back = o.black76_price(f, k, t, iv, 0.0, typ)
    assert np.max(np.abs(back - price)[ident]) < 1e-10 * f
    vega = np.asarray(o.black76_greeks(f, k, t, v, 0.0, typ)["vega"]) * 100
    well = ident & (np.finfo(float).eps * price / np.maximum(vega, 1e-300) < 1e-10)
    assert np.max(np.abs(iv - v)[well]) < 1e-8


def test_iv_matches_lets_be_rational_fixture():
    ref = json.loads((FIX / "reference_values.json").read_text())["black76_iv"]
    p = np.array([r["price"] for r in ref])
    iv = o.implied_vol_black76(p, np.array([r["forward"] for r in ref]),
                               np.array([r["strike"] for r in ref]), np.array([r["t"] for r in ref]),
                               0.0, [r["opt_type"] for r in ref])
    np.testing.assert_allclose(iv, [r["iv"] for r in ref], atol=1e-8, rtol=0)


def test_iv_bsm_and_nse():
    p = o.bsm_price(42, 40, 0.5, 0.2, 0.1, 0.0, "C")
    assert o.implied_vol_bsm(p, 42, 40, 0.5, 0.1, 0.0, "C") == pytest.approx(0.2, abs=1e-12)
    p = o.bsm_price(24000, 24200, 10 / 365, 0.14, 0.10, 0.0, "P")
    assert o.implied_vol_nse(p, 24000, 24200, 10 / 365, "PE") == pytest.approx(0.14, abs=1e-12)


def test_iv_invalid_prices_are_nan():
    # below intrinsic, at intrinsic, above the forward, zero/negative price, t<=0, bad type
    iv = o.implied_vol_black76([5, 10, 120, 0, -1, 5, 5], 110, 100, [1, 1, 1, 1, 1, 0, 1], 0.0,
                               ["C", "C", "C", "C", "C", "C", "Z"])
    assert np.isnan(iv).all()
    assert np.isnan(o.implied_vol_black76(np.nan, 100, 100, 1))
    assert np.isnan(o.implied_vol_black76(5, 100, None, 1))


def test_iv_speed_40k():
    rng = np.random.default_rng(0)
    n = 40000
    k = 100 * np.exp(rng.uniform(np.log(0.5), np.log(2), n))
    t = rng.uniform(1 / 365, 2, n)
    v = rng.uniform(0.05, 3.0, n)
    typ = np.where(rng.random(n) < 0.5, "C", "P")
    price = o.black76_price(100.0, k, t, v, 0.0, typ)
    t0 = time.perf_counter()
    o.implied_vol_black76(price, 100.0, k, t, 0.0, typ)
    assert time.perf_counter() - t0 < 3.0   # ~0.1 s measured; generous for slow CI machines
