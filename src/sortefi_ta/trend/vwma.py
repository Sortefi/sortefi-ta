import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_vwma(df: pd.DataFrame, period: int = 20) -> dict[str, np.ndarray]:
    """Volume Weighted Moving Average. Returns {"vwma"}."""
    n = len(df)
    if ta is None or n < period:
        return {"vwma": _nan_array(n)}
    result = ta.vwma(df["close"], df["volume"], length=period)
    if result is None:
        return {"vwma": _nan_array(n)}
    return {"vwma": result.to_numpy(dtype=float)}
