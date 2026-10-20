import numpy as np
import pandas as pd
from sortefi_ta._divergence import _calc_divergence
from sortefi_ta.volatility.atr import calc_atr
from .rsi import calc_rsi


def calc_rsi_divergence(
    df: pd.DataFrame,
    rsi_period: int = 14,
    lookback_left: int = 5,
    lookback_right: int = 5,
    range_lower: int = 5,
    range_upper: int = 60,
    method: str = "tv",
    atr_period: int = 14,
    atr_mult: float = 1.0,
    fractal_period: int = 2,
) -> dict[str, np.ndarray]:
    """RSI Divergence — regular bullish and bearish.

    method="tv" (default): fixed-window method, the common charting-platform convention —
    pivots on RSI within a fixed bar distance (range_lower=5, range_upper=60).

    method="swing": dynamic ATR-significant price swings with no bar-distance cap
    (for multi-month divergences between significant swing highs/lows). See
    sortefi_ta/_divergence.py for the swing algorithm.

    Regular Bullish: price lower low + RSI higher low  → reversal up signal
    Regular Bearish: price higher high + RSI lower high → reversal down signal

    Returns:
      bull_div: 1.0 at confirmation bar where bullish divergence detected, else 0.0
      bear_div: 1.0 at confirmation bar where bearish divergence detected, else 0.0
    """
    n = len(df)
    empty = {"bull_div": np.zeros(n), "bear_div": np.zeros(n), "bull_strength": np.zeros(n), "bear_strength": np.zeros(n)}

    min_bars = rsi_period + lookback_left + lookback_right + 1
    if n < min_bars:
        return empty

    rsi_vals = calc_rsi(df, period=rsi_period)["rsi"]
    price_low = df["low"].to_numpy(dtype=float)
    price_high = df["high"].to_numpy(dtype=float)

    atr_vals = calc_atr(df, period=atr_period)["atr"] if method == "swing" else None
    bull_div, bear_div, bull_strength, bear_strength = _calc_divergence(
        price_low, price_high, rsi_vals,
        lookback_left, lookback_right, range_lower, range_upper,
        method=method, atr=atr_vals, atr_mult=atr_mult, fractal_period=fractal_period,
    )
    return {"bull_div": bull_div, "bear_div": bear_div, "bull_strength": bull_strength, "bear_strength": bear_strength}
