import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_dema(df: pd.DataFrame, period: int = 20, col: str = "close") -> dict[str, np.ndarray]:
    """Double Exponential Moving Average. Returns {"dema"}."""
    n = len(df)
    if ta is None or n < period * 2:
        return {"dema": _nan_array(n)}
    result = ta.dema(df[col], length=period)
    if result is None:
        return {"dema": _nan_array(n)}
    return {"dema": result.to_numpy(dtype=float)}
