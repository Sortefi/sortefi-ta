import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sortefi_ta import options as o  # noqa: E402


# -- NSE published IV -----------------------------------------------------

_NSE = Path(__file__).with_name("market_data") / "nse_nifty_sample.json"


@pytest.mark.skipif(not _NSE.exists(), reason="local-only exchange data (not published)")
def test_implied_vol_nse_matches_published_iv():
    fx = json.loads(_NSE.read_text())
    rows = fx["rows"]
    iv = o.implied_vol_nse([r["last_price"] for r in rows], [r["spot"] for r in rows],
                           [r["strike"] for r in rows], [r["days"] / 365 for r in rows],
                           [r["opt_type"] for r in rows])
    err = np.abs(iv * 100 - [r["nse_iv_pct"] for r in rows])
    assert err.max() < 0.01        # NSE rounds IV to 2 decimals


def test_implied_vol_nse_synthetic_round_trip():
    # premiums priced with NSE's convention (BSM, r = 10 %, q = 0) and rounded to the
    # 0.05 price tick; the solver must recover the vol wherever the tick barely moves it
    spot, t = 22600.0, 11 / 365
    k = np.arange(21000, 24001, 100.0)
    rows = [(kk, typ, 0.12 + 0.00004 * abs(kk - spot)) for kk in k for typ in ("CE", "PE")]
    strike = np.array([r[0] for r in rows])
    typ = np.array([r[1] for r in rows])
    vol = np.array([r[2] for r in rows])
    price = np.round(np.asarray(o.bsm_price(spot, strike, t, vol, 0.10, 0.0, typ)) / 0.05) * 0.05
    iv = np.asarray(o.implied_vol_nse(price, spot, strike, t, typ))
    vega = np.asarray(o.bsm_greeks(spot, strike, t, vol, 0.10, 0.0, typ)["vega"])
    sharp = (price > 1) & (0.025 / vega < 0.005)       # half a tick moves the vol < 0.005 pts
    assert sharp.sum() > 20
    assert np.max(np.abs(iv[sharp] - vol[sharp]) * 100) < 0.01


# -- a synthetic chain ------------------------------------------------------

def _chain(fwd=100.0, t=30 / 365, skew=0.002):
    rows = []
    for exp, tt, f in (("2026-11-27", t, fwd), ("2026-12-31", 2 * t, fwd * 1.001)):
        for k in np.arange(80, 121, 5.0):
            vol = 0.2 + skew * (f - k)          # puts (low strikes) richer
            for typ in ("CE", "PE"):
                rows.append(dict(expiry=exp, strike=k, opt_type=typ, t=tt, forward=f,
                                 price=o.black76_price(f, k, tt, vol, 0.0, typ[0]),
                                 oi=1000 + (100 - k) * (10 if typ == "PE" else -10),
                                 volume=50, true_iv=vol))
    return pd.DataFrame(rows)


def test_pcr():
    df = pd.DataFrame({"opt_type": ["C", "P", "CE", "PE"], "oi": [100, 150, 50, 0],
                       "volume": [10, 30, 10, 10]})
    assert o.pcr(df) == pytest.approx(1.0)
    assert o.pcr(df, by="volume") == pytest.approx(2.0)
    assert np.isnan(o.pcr(df.iloc[[1, 3]]))


def test_max_pain_hand_example():
    df = pd.DataFrame({"strike": [90, 100, 110, 90, 100, 110], "opt_type": list("CCCPPP"),
                       "oi": [10, 50, 100, 100, 50, 10]})
    # payouts: at 90 -> puts 50*10+10*20 = 700; at 100 -> calls 100 + puts 100 = 200; at 110 -> 700
    assert o.max_pain(df) == 100.0
    s = o.max_pain(_chain())
    assert isinstance(s, pd.Series) and len(s) == 2


def test_atm_iv_term_structure_and_cm():
    df = _chain()
    atm = o.atm_iv(df.drop(columns=["price"]).rename(columns={"true_iv": "iv"}))
    assert atm.iloc[0] == pytest.approx(0.2, abs=1e-12)
    from_price = o.atm_iv(df)                                  # IV implied from prices
    assert from_price.iloc[0] == pytest.approx(0.2, abs=1e-9)
    ts = o.term_structure(df)
    assert list(ts.columns) == ["expiry", "t", "atm_iv"] and ts["t"].is_monotonic_increasing
    # forward outside listed strikes -> NaN
    assert np.isnan(o.atm_iv(df, forward=150.0).iloc[0])
    # constant maturity: flat vol stays flat; interpolation is in total variance
    assert o.constant_maturity_iv([20 / 365, 40 / 365], [0.2, 0.2], days=30) == pytest.approx(0.2)
    v = o.constant_maturity_iv([20 / 365, 40 / 365], [0.2, 0.3], days=30)
    assert v == pytest.approx(np.sqrt((0.04 * 20 + 0.09 * 40) / 2 / 30), rel=1e-12)
    assert np.isnan(o.constant_maturity_iv([40 / 365, 60 / 365], [0.2, 0.3], days=30))
    assert o.constant_maturity_iv([40 / 365], [0.25], days=30, extrapolate=True) == 0.25


def test_skew_sign():
    s = o.skew_25d(_chain())
    assert (s > 0).all()                       # puts priced at higher vol
    flat = o.skew_25d(_chain(skew=0.0))
    assert np.allclose(flat, 0.0, atol=1e-9)


def test_iv_rank_and_percentile():
    x = np.array([10, 20, 30, 40, 50, 25.0])
    rank = o.iv_rank(x, lookback=5)
    assert np.isnan(rank[3]) and rank[4] == pytest.approx(100.0)
    assert rank[5] == pytest.approx((25 - 20) / (50 - 20) * 100)
    pct = o.iv_percentile(x, lookback=5)
    assert np.isnan(pct[4]) and pct[5] == pytest.approx(40.0)   # 10, 20 below 25
    assert np.isnan(o.iv_rank(np.full(10, 0.2), 5)[-1])          # flat -> NaN


def test_iv_rank_and_percentile_skip_missing_days():
    x = np.array([10, np.nan, 30, 40, 50, 25.0])
    rank = o.iv_rank(x, lookback=5)
    assert rank[4] == pytest.approx(100.0)                       # window 10, -, 30, 40, 50
    assert rank[5] == pytest.approx(0.0)                         # window -, 30, 40, 50, 25: 25 is the low
    pct = o.iv_percentile(x, lookback=5)
    assert pct[5] == pytest.approx(25.0)                         # earlier 10, -, 30, 40, 50: 1 of 4 below 25
    y = x.copy()
    y[-1] = np.nan
    assert np.isnan(o.iv_rank(y, 5)[-1]) and np.isnan(o.iv_percentile(y, 5)[-1])  # no IV today
    assert np.isnan(o.iv_rank(x, lookback=5, min_periods=5)[5])  # caller can demand a full window
    assert np.isnan(o.iv_rank(x[:4], lookback=5)).all()          # window not yet elapsed


def test_moves_and_buildup():
    assert o.expected_move(100, 0.2, 0.25) == pytest.approx(10.0)
    assert np.isnan(o.expected_move(-1, 0.2, 0.25))
    assert o.straddle_move(3.0, 2.5) == 5.5
    b = o.buildup([1, -1, 1, -1, 0, np.nan], [1, 1, -1, -1, 5, 1])
    assert list(b) == ["long_buildup", "short_buildup", "short_covering", "long_unwinding",
                       "neutral", "neutral"]


# -- strategies -----------------------------------------------------------

def test_bull_call_spread():
    legs = [o.Leg("C", 100, 5.0, 1), dict(kind="C", strike=110, premium=2.0, qty=-1)]
    np.testing.assert_allclose(o.payoff_at_expiry(legs, [90, 103, 120]), [-3, 0, 7])
    np.testing.assert_allclose(o.breakevens(legs), [103.0])
    assert o.max_profit(legs) == 7 and o.max_loss(legs) == 3


def test_short_straddle_unbounded_and_lot_size():
    legs = [dict(kind="CE", strike=100, premium=4, qty=-1, multiplier=50),
            dict(kind="PE", strike=100, premium=3, qty=-1, multiplier=50)]
    np.testing.assert_allclose(o.breakevens(legs), [93.0, 107.0])
    assert o.max_profit(legs) == 350 and o.max_loss(legs) == np.inf
    long = [dict(l, qty=1) for l in legs]
    assert o.max_profit(long) == np.inf and o.max_loss(long) == 350


def test_covered_call_and_flat_zero_region():
    legs = [dict(kind="S", strike=100, qty=1), dict(kind="C", strike=110, premium=2, qty=-1)]
    np.testing.assert_allclose(o.breakevens(legs), [98.0])
    assert o.max_profit(legs) == 12 and o.max_loss(legs) == 98
    box = [dict(kind="C", strike=100, premium=0, qty=1), dict(kind="C", strike=110, premium=0, qty=-1)]
    assert o.breakevens(box)[0] == 0 and 100 in o.breakevens(box)


def test_probability_of_profit():
    legs = [dict(kind="C", strike=100, premium=0, qty=1)]
    # P(S_T > 100) with F = 100 = N(-vol sqrt t / 2)
    p = o.probability_of_profit(legs, 100, 0.2, 1.0)
    assert p == pytest.approx(float(o.norm_cdf(-0.1)), rel=1e-12)
    strangle = [dict(kind="C", strike=110, premium=2, qty=-1), dict(kind="P", strike=90, premium=2, qty=-1)]
    p = o.probability_of_profit(strangle, 100, 0.2, 30 / 365)
    lo, hi = o.breakevens(strangle)
    d = lambda x: (np.log(x / 100) + 0.5 * 0.04 * 30 / 365) / (0.2 * np.sqrt(30 / 365))
    assert p == pytest.approx(float(o.norm_cdf(d(hi)) - o.norm_cdf(d(lo))), rel=1e-12)
    assert np.isnan(o.probability_of_profit(strangle, 100, -0.2, 1))


def test_value_now_converges_to_payoff():
    legs = [dict(kind="C", strike=100, premium=5, qty=1), dict(kind="P", strike=95, premium=2, qty=-1)]
    f = np.array([80.0, 95.0, 100.0, 120.0])
    np.testing.assert_allclose(o.value_now(legs, f, 0.3, 0.0), o.payoff_at_expiry(legs, f))
    v = o.value_now(legs, f, [0.3, 0.35], 0.5, 0.05)
    manual = (o.black76_price(f, 100, 0.5, 0.3, 0.05, "C") - 5) - (o.black76_price(f, 95, 0.5, 0.35, 0.05, "P") - 2)
    np.testing.assert_allclose(v, manual)
    assert np.isnan(o.max_profit([dict(kind="X", strike=1)]))
