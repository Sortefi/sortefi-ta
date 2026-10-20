import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array


def calc_kurtosis(df: pd.DataFrame, period: int = 30, col: str = "close", returns: bool = True, excess: bool = False) -> dict[str, np.ndarray]:
    """Rolling log-return Pearson kurtosis (normal=3); excess=True uses normal=0."""
    n = len(df)
    if n < period + int(returns):
        return {"kurtosis": _nan_array(n)}
    prices = df[col].astype(float)
    values = np.log(prices.where(prices > 0)).diff() if returns else prices
    result = values.rolling(period).kurt()
    if not excess:
        result = result + 3
    return {"kurtosis": result.to_numpy(dtype=float)}
