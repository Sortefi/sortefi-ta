"""Tests for what 2.0.0 adds or changes: TradingView seeding (RSI, Stoch RSI, ADX), VWAP on hlc3,
flat-price zeros, flat open / close patterns, Renko bricks, performance statistics and the two
options additions (gamma exposure, put-call-parity forward)."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sortefi_ta as T                   # noqa: E402
from sortefi_ta import options as opt   # noqa: E402
from sortefi_ta import performance as pf  # noqa: E402
from _data import make_df                # noqa: E402


def _frame(close, high=None, low=None, open_=None, volume=None):
    close = np.asarray(close, dtype=float)
    n = close.size
    return pd.DataFrame({
        "open_time": pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
        "open": close if open_ is None else np.asarray(open_, float),
        "high": close if high is None else np.asarray(high, float),
        "low": close if low is None else np.asarray(low, float),
        "close": close,
        "volume": np.ones(n) if volume is None else np.asarray(volume, float),
        "symbol": "TEST",
    })


# ── RSI / Stoch RSI / ADX seeded like TradingView ───────────────────────────

def test_rsi_first_value_is_simple_average_of_first_changes():
    df = make_df(n=60)
    p = 14
    rsi = T.calc_rsi(df, period=p)["rsi"]
    ch = np.diff(df["close"].to_numpy())[:p]
    up, dn = ch[ch > 0].sum() / p, -ch[ch < 0].sum() / p
    assert np.isnan(rsi[:p]).all()
    assert rsi[p] == pytest.approx(100 * up / (up + dn), rel=1e-12)
    nxt = np.diff(df["close"].to_numpy())[p]
    up2 = up + (max(nxt, 0) - up) / p
    dn2 = dn + (max(-nxt, 0) - dn) / p
    assert rsi[p + 1] == pytest.approx(100 * up2 / (up2 + dn2), rel=1e-12)


def test_stochrsi_period_is_the_stochastic_window():
    df = make_df(n=200)
    a = T.calc_stochrsi(df, period=14, rsi_period=14)["stochrsi_k"]
    b = T.calc_stochrsi(df, period=5, rsi_period=14)["stochrsi_k"]
    ok = np.isfinite(a) & np.isfinite(b)
    assert ok.sum() > 50 and not np.allclose(a[ok], b[ok])


def test_adx_first_values_land_where_tradingview_starts_them():
    df = make_df(n=120)
    p = 14
    out = T.calc_adx(df, period=p)
    assert np.isnan(out["di_plus"][:p]).all() and np.isfinite(out["di_plus"][p])
    assert np.isnan(out["adx"][:2 * p - 1]).all() and np.isfinite(out["adx"][2 * p - 1])


# ── VWAP on hlc3 ────────────────────────────────────────────────────────────

def test_vwap_uses_high_low_close_average():
    df = _frame([10.0, 11.0], high=[12.0, 13.0], low=[9.0, 10.0], volume=[1.0, 3.0])
    v = T.calc_vwap(df)["vwap"]
    hlc3 = (df["high"] + df["low"] + df["close"]).to_numpy() / 3
    assert v[0] == pytest.approx(hlc3[0])
    assert v[1] == pytest.approx((hlc3[0] * 1 + hlc3[1] * 3) / 4)


# ── flat price gives exact zeros ────────────────────────────────────────────

@pytest.mark.parametrize("price", [0.1234, 1.0, 100.0, 98765.4321])
def test_flat_price_oscillators_are_exactly_zero(price):
    df = _frame(np.full(120, price))
    m = T.calc_macd(df)
    for k in ("macd", "signal", "histogram"):
        if k in m:
            v = m[k][np.isfinite(m[k])]
            assert (v == 0.0).all(), k
    ao = T.calc_ao(df)
    v = next(iter(ao.values()))
    assert (v[np.isfinite(v)] == 0.0).all()


def test_zero_snap_leaves_moving_prices_alone():
    from sortefi_ta._ta import _momentum as M
    s = pd.Series(make_df(n=300)["close"].to_numpy())
    snapped = M._snap_zero(s - s.rolling(10).mean(), s)
    raw = (s - s.rolling(10).mean())
    ok = raw.abs() > 1e-9
    assert np.array_equal(snapped[ok].to_numpy(), raw[ok].to_numpy())


# ── flat open / flat close ──────────────────────────────────────────────────

def test_flat_open_and_flat_close():
    #          opened at low (bull)  opened at high (bear)  normal candle  doji
    o = np.array([100.0, 100.0, 100.0, 100.0])
    c = np.array([105.0, 95.0, 102.0, 100.0])
    hi = np.array([106.0, 100.2, 103.0, 101.0])
    lo = np.array([99.9, 94.0, 98.0, 99.0])
    df = pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c})
    got = T.calc_candlestick_patterns(df, ["flat_open", "flat_close"])
    fo = np.asarray(got["flat_open"] if isinstance(got, dict) else got["flat_open"])
    assert fo.tolist() == [1.0, 1.0, 0.0, 0.0]
    #           closed at high (bull)  closed at low (bear)
    df2 = pd.DataFrame({"open": [100.0, 100.0], "high": [105.1, 101.0],
                        "low": [99.0, 95.0], "close": [105.0, 95.1]})
    got2 = T.calc_candlestick_patterns(df2, ["flat_close"])
    assert np.asarray(got2["flat_close"]).tolist() == [1.0, 1.0]


# ── Renko ───────────────────────────────────────────────────────────────────

def test_renko_hand_example():
    df = _frame([100, 101, 102.5, 103, 100.4, 99])
    r = T.calc_renko(df, box=1.0)
    assert r["renko_new"].tolist() == [0, 1, 1, 1, -1, -2]
    assert r["renko_dir"].tolist() == [0, 1, 1, 1, -1, -1]
    assert r["renko_run"].tolist() == [0, 1, 2, 3, 1, 3]
    assert np.isnan(r["renko_level"][0])
    assert r["renko_level"][1:].tolist() == [101, 102, 103, 101, 99]
    bricks = T.renko_bricks(df, box=1.0)
    assert bricks["direction"].tolist() == [1, 1, 1, -1, -1, -1]
    assert bricks["close"].tolist() == [101, 102, 103, 101, 100, 99]


def test_renko_reversal_needs_two_boxes():
    r = T.calc_renko(_frame([100, 102, 101.5, 100.9]), box=1.0)   # top 102; reversal needs <= 100
    assert r["renko_dir"].tolist() == [0, 1, 1, 1]


def test_renko_atr_box_and_no_box():
    df = make_df(n=100)
    r = T.calc_renko(df)
    assert np.isfinite(r["box"]).all() and r["box"][0] > 0
    assert (T.calc_renko(_frame([100, 101]), box=0)["renko_new"] == 0).all()


# ── performance statistics ──────────────────────────────────────────────────

def test_performance_against_hand_formulas():
    r = np.random.default_rng(7).normal(0.0005, 0.01, 500)
    sd = r.std(ddof=1)
    assert pf.sharpe_ratio(r) == pytest.approx(r.mean() / sd * math.sqrt(252))
    assert pf.annualized_volatility(r) == pytest.approx(sd * math.sqrt(252))
    assert pf.value_at_risk(r, 0.05) == pytest.approx(np.percentile(r, 5))
    assert pf.beta(r, r) == pytest.approx(1.0)
    assert pf.correlation(r, -r) == pytest.approx(-1.0)
    assert pf.cumulative_return([0.1, -0.1]) == pytest.approx(1.1 * 0.9 - 1)
    assert pf.total_return([100, 125]) == pytest.approx(0.25)
    assert pf.win_rate([0.01, -0.01, 0.0, 0.02]) == pytest.approx(0.5)


def test_normal_quantile_and_normal_var():
    assert pf._norm_ppf(0.975) == pytest.approx(1.959963984540054, abs=1e-9)
    assert pf._norm_ppf(0.001) == pytest.approx(-3.090232306167813, abs=1e-9)
    r = np.random.default_rng(3).normal(0, 0.02, 2000)
    assert pf.value_at_risk(r, 0.05, "normal") == pytest.approx(r.mean() - 1.6448536269514722 * r.std(ddof=1))
    assert pf.conditional_value_at_risk(r, 0.05, "normal") < pf.value_at_risk(r, 0.05, "normal")


def test_performance_degenerate_inputs_return_none():
    for fn in (pf.sharpe_ratio, pf.sortino_ratio, pf.annualized_volatility, pf.skewness):
        assert fn([]) is None
    assert pf.sharpe_ratio([0.01, 0.01, 0.01]) is None          # zero variance
    with pytest.raises(ValueError):
        pf.value_at_risk([0.1, 0.2], method="nope")


def test_no_drawdown_functions_published():
    for name in ("max_drawdown", "drawdown_series", "calmar_ratio", "ulcer_index"):
        assert not hasattr(pf, name)


# ── options additions ───────────────────────────────────────────────────────

def test_gamma_exposure_matches_black76_gamma():
    f, k, t, v, oi = 100.0, 105.0, 30 / 365, 0.2, 5000.0
    g = opt.black76_greeks(f, k, t, v)["gamma"]
    assert opt.gamma_exposure(f, k, t, v, oi) == pytest.approx(g * oi * f * f * 0.01)
    arr = opt.gamma_exposure(f, np.array([95.0, 100.0, 105.0]), t, v, oi)
    assert arr.shape == (3,) and arr[1] == arr.max()


def test_parity_forward_recovers_forward():
    f, t, r = 101.3, 0.25, 0.05
    k = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
    call = opt.black76_price(f, k, t, 0.25, r, "C")
    put = opt.black76_price(f, k, t, 0.25, r, "P")
    assert opt.parity_forward(k, call, put, t, r) == pytest.approx(f, rel=1e-10)
    assert math.isnan(opt.parity_forward(k, np.full(5, np.nan), put, t, r))
