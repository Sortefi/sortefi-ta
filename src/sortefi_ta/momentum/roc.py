import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_roc(df: pd.DataFrame, period: int = 9, col: str = "close") -> dict[str, np.ndarray]:
    """Rate of Change. Returns {"roc"}."""
    n = len(df)
    if ta is None or n < period + 1:
        return {"roc": _nan_array(n)}
    result = ta.roc(df[col], length=period)
    if result is None:
        return {"roc": _nan_array(n)}
    return {"roc": result.to_numpy(dtype=float)}
