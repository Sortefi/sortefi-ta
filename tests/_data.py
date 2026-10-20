"""Deterministic OHLCV series used by the tests."""
import numpy as np
import pandas as pd


def make_df(n=400, seed=1, drift=0.0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.015, n)))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    volume = rng.lognormal(12, 0.6, n).round()
    t = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"open_time": t, "open": open_, "high": high, "low": low,
                         "close": close, "volume": volume, "symbol": "TEST"})


EXTRA_ARGS = {
    "calc_anchored_vwap": {"anchor": 10},
    "calc_level_state": {"level": 100.0},
    "calc_range_state": {"range_high": 110.0, "range_low": 95.0},
}
