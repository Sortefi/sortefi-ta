import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_rsi(df: pd.DataFrame, period: int = 14, col: str = "close") -> dict[str, np.ndarray]:
    """RSI (Wilder's smoothing). Returns {"rsi": ndarray}."""
    n = len(df)
    if ta is None or n < period + 1:
        return {"rsi": _nan_array(n)}
    result = ta.rsi(df[col], length=period)
    if result is None:
        return {"rsi": _nan_array(n)}
    return {"rsi": result.to_numpy(dtype=float)}
