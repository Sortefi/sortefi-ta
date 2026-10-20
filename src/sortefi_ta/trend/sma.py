import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_sma(df: pd.DataFrame, period: int = 20, col: str = "close") -> dict[str, np.ndarray]:
    """Simple Moving Average. Returns {"sma": ndarray}."""
    n = len(df)
    if ta is None or n < period:
        return {"sma": _nan_array(n)}
    result = ta.sma(df[col], length=period)
    if result is None:
        return {"sma": _nan_array(n)}
    return {"sma": result.to_numpy(dtype=float)}
