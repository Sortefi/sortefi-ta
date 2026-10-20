import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_mad(df: pd.DataFrame, period: int = 20, col: str = "close") -> dict[str, np.ndarray]:
    """Mean Absolute Deviation. Returns {"mad"}."""
    n = len(df)
    if ta is None or n < period:
        return {"mad": _nan_array(n)}
    result = ta.mad(df[col], length=period)
    if result is None:
        return {"mad": _nan_array(n)}
    return {"mad": result.to_numpy(dtype=float)}
