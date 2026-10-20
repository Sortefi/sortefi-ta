import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_tema(df: pd.DataFrame, period: int = 20, col: str = "close") -> dict[str, np.ndarray]:
    """Triple Exponential Moving Average. Returns {"tema"}."""
    n = len(df)
    if ta is None or n < period * 3:
        return {"tema": _nan_array(n)}
    result = ta.tema(df[col], length=period)
    if result is None:
        return {"tema": _nan_array(n)}
    return {"tema": result.to_numpy(dtype=float)}
