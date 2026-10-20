import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_zscore(df: pd.DataFrame, period: int = 20, col: str = "close") -> dict[str, np.ndarray]:
    """Z-Score (close - sma) / stdev, using population stdev (ddof=0) to match TV. Returns {"zscore"}."""
    n = len(df)
    if n < period:
        return {"zscore": _nan_array(n)}
    mean = df[col].rolling(period).mean()
    std  = df[col].rolling(period).std(ddof=0)
    result = (df[col] - mean) / std
    return {"zscore": result.to_numpy(dtype=float)}
