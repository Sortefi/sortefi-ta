import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array


def calc_stochastic(
    df: pd.DataFrame, k: int = 14, d: int = 3, smooth_k: int = 1
) -> dict[str, np.ndarray]:
    """Stochastic Oscillator (TradingView definition). Returns {"stoch_k", "stoch_d"}.

    raw %K = 100 × (close − lowest low over k) ÷ (highest high over k − lowest low over k)
    %K     = SMA(raw %K, smooth_k)   (smooth_k = 1 → raw %K, TradingView's default 14/1/3)
    %D     = SMA(%K, d)

    Computed here rather than via pandas-ta: pandas-ta 0.4.x crashes when smooth_k = 1
    (its SMA of length 1 returns an empty series), so it is computed directly here. A flat window (highest = lowest) gives NaN, never a divide-by-zero.
    """
    n = len(df)
    empty = {"stoch_k": _nan_array(n), "stoch_d": _nan_array(n)}
    k, d, smooth_k = int(k), int(d), max(1, int(smooth_k))
    if n < k + smooth_k - 1 or k < 1 or d < 1:
        return empty
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    lowest = low.rolling(k, min_periods=k).min()
    highest = high.rolling(k, min_periods=k).max()
    span = (highest - lowest).replace(0.0, np.nan)
    raw_k = 100.0 * (close - lowest) / span
    stoch_k = raw_k.rolling(smooth_k, min_periods=smooth_k).mean() if smooth_k > 1 else raw_k
    stoch_d = stoch_k.rolling(d, min_periods=d).mean()
    return {
        "stoch_k": stoch_k.to_numpy(dtype=float),
        "stoch_d": stoch_d.to_numpy(dtype=float),
    }
