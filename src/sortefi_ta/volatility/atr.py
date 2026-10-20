import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_atr(df: pd.DataFrame, period: int = 14) -> dict[str, np.ndarray]:
    """ATR (Wilder's smoothing). Returns {"atr": ndarray}."""
    n = len(df)
    if ta is None or n < period + 1:
        return {"atr": _nan_array(n)}
    result = ta.atr(df["high"], df["low"], df["close"], length=period)
    if result is None:
        return {"atr": _nan_array(n)}
    return {"atr": result.to_numpy(dtype=float)}
