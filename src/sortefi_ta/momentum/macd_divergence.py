import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array
from sortefi_ta._divergence import _calc_divergence
from sortefi_ta.volatility.atr import calc_atr
from sortefi_ta.momentum.macd import calc_macd


def calc_macd_divergence(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    lookback_left: int = 5,
    lookback_right: int = 5,
    range_lower: int = 5,
    range_upper: int = 60,
    method: str = "tv",
    atr_period: int = 14,
    atr_mult: float = 1.0,
    fractal_period: int = 2,
) -> dict[str, np.ndarray]:
    """MACD Histogram Divergence — regular bullish and bearish.

    Uses MACD histogram (not MACD line) as the indicator series.
    Same pivot logic as TradingView's built-in RSI divergence.

    Returns:
      bull_div: 1.0 at confirmation bar where bullish divergence detected, else 0.0
      bear_div: 1.0 at confirmation bar where bearish divergence detected, else 0.0
    """
    n = len(df)
    empty = {"bull_div": np.zeros(n), "bear_div": np.zeros(n), "bull_strength": np.zeros(n), "bear_strength": np.zeros(n)}
    if n < slow + signal + lookback_left + lookback_right + 1:
        return empty

    macd_data = calc_macd(df, fast=fast, slow=slow, signal=signal)
    histogram = macd_data["histogram"]
    price_low = df["low"].to_numpy(dtype=float)
    price_high = df["high"].to_numpy(dtype=float)

    atr_vals = calc_atr(df, period=atr_period)["atr"] if method == "swing" else None
    bull_div, bear_div, bull_strength, bear_strength = _calc_divergence(
        price_low, price_high, histogram,
        lookback_left, lookback_right, range_lower, range_upper,
        method=method, atr=atr_vals, atr_mult=atr_mult, fractal_period=fractal_period,
    )
    return {"bull_div": bull_div, "bear_div": bear_div, "bull_strength": bull_strength, "bear_strength": bear_strength}
