import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_wma(df: pd.DataFrame, period: int = 20, col: str = "close") -> dict[str, np.ndarray]:
    """Weighted Moving Average. Returns {"wma"}."""
    n = len(df)
    if ta is None or n < period:
        return {"wma": _nan_array(n)}
    result = ta.wma(df[col], length=period)
    if result is None:
        return {"wma": _nan_array(n)}
    return {"wma": result.to_numpy(dtype=float)}
