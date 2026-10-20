import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_ema(df: pd.DataFrame, period: int = 20, col: str = "close") -> dict[str, np.ndarray]:
    """Exponential Moving Average. Returns {"ema": ndarray}."""
    n = len(df)
    if ta is None or n < period:
        return {"ema": _nan_array(n)}
    result = ta.ema(df[col], length=period)
    if result is None:
        return {"ema": _nan_array(n)}
    return {"ema": result.to_numpy(dtype=float)}
