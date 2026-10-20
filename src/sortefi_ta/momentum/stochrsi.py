import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_stochrsi(
    df: pd.DataFrame, period: int = 14, rsi_period: int = 14, k: int = 3, d: int = 3
) -> dict[str, np.ndarray]:
    """Stochastic RSI. Returns {"stochrsi_k", "stochrsi_d"}."""
    n = len(df)
    empty = {"stochrsi_k": _nan_array(n), "stochrsi_d": _nan_array(n)}
    if ta is None or n < rsi_period + period:
        return empty
    result = ta.stochrsi(df["close"], length=rsi_period, rsi_length=rsi_period, k=k, d=d)
    if result is None or result.empty:
        return empty
    return {
        "stochrsi_k": result.iloc[:, 0].to_numpy(dtype=float),
        "stochrsi_d": result.iloc[:, 1].to_numpy(dtype=float),
    }
