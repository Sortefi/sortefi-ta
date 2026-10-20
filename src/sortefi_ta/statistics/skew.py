import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array


def calc_skew(df: pd.DataFrame, period: int = 30, col: str = "close", returns: bool = True) -> dict[str, np.ndarray]:
    """Rolling log-return skewness; returns=False explicitly measures price levels."""
    n = len(df)
    if n < period + int(returns):
        return {"skew": _nan_array(n)}
    prices = df[col].astype(float)
    values = np.log(prices.where(prices > 0)).diff() if returns else prices
    result = values.rolling(period).skew()
    return {"skew": result.to_numpy(dtype=float)}
