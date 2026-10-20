import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_mfi(df: pd.DataFrame, period: int = 14) -> dict[str, np.ndarray]:
    """Money Flow Index. Returns {"mfi"}."""
    n = len(df)
    if ta is None or n < period + 1:
        return {"mfi": _nan_array(n)}
    result = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=period)
    if result is None:
        return {"mfi": _nan_array(n)}
    return {"mfi": result.to_numpy(dtype=float)}
