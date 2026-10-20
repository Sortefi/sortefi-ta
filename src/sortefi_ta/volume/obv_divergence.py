import numpy as np
import pandas as pd
from sortefi_ta._divergence import _calc_divergence
from sortefi_ta.volatility.atr import calc_atr
from sortefi_ta.volume.obv import calc_obv


def calc_obv_divergence(
    df: pd.DataFrame,
    lookback_left: int = 5,
    lookback_right: int = 5,
    range_lower: int = 5,
    range_upper: int = 60,
    method: str = "tv",
    atr_period: int = 14,
    atr_mult: float = 1.0,
    fractal_period: int = 2,
) -> dict[str, np.ndarray]:
    """OBV Divergence — regular bullish and bearish.

    Price higher high + OBV lower high = bearish (selling into rally).
    Price lower low + OBV higher low = bullish (accumulation on dip).
    Classic Wyckoff/volume analysis divergence.

    Returns:
      bull_div: 1.0 at confirmation bar where bullish divergence detected, else 0.0
      bear_div: 1.0 at confirmation bar where bearish divergence detected, else 0.0
    """
    n = len(df)
    empty = {"bull_div": np.zeros(n), "bear_div": np.zeros(n), "bull_strength": np.zeros(n), "bear_strength": np.zeros(n)}
    if n < lookback_left + lookback_right + 1:
        return empty

    obv_vals = calc_obv(df)["obv"]
    price_low = df["low"].to_numpy(dtype=float)
    price_high = df["high"].to_numpy(dtype=float)

    atr_vals = calc_atr(df, period=atr_period)["atr"] if method == "swing" else None
    bull_div, bear_div, bull_strength, bear_strength = _calc_divergence(
        price_low, price_high, obv_vals,
        lookback_left, lookback_right, range_lower, range_upper,
        method=method, atr=atr_vals, atr_mult=atr_mult, fractal_period=fractal_period,
    )
    return {"bull_div": bull_div, "bear_div": bear_div, "bull_strength": bull_strength, "bear_strength": bear_strength}
