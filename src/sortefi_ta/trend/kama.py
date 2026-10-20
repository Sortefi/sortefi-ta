import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_kama(
    df: pd.DataFrame, period: int = 10, fast: int = 2, slow: int = 30, col: str = "close"
) -> dict[str, np.ndarray]:
    """Kaufman Adaptive Moving Average. Returns {"kama"}."""
    n = len(df)
    if ta is None or n < period + slow:
        return {"kama": _nan_array(n)}
    result = ta.kama(df[col], length=period, fast=fast, slow=slow)
    if result is None:
        return {"kama": _nan_array(n)}
    return {"kama": result.to_numpy(dtype=float)}
