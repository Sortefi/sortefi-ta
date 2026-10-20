import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_hma(df: pd.DataFrame, period: int = 20, col: str = "close") -> dict[str, np.ndarray]:
    """Hull Moving Average. Returns {"hma"}."""
    n = len(df)
    if ta is None or n < period:
        return {"hma": _nan_array(n)}
    result = ta.hma(df[col], length=period)
    if result is None:
        return {"hma": _nan_array(n)}
    return {"hma": result.to_numpy(dtype=float)}
