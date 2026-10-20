import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_chop(df: pd.DataFrame, period: int = 14) -> dict[str, np.ndarray]:
    """Choppiness Index (0-100, >61.8 = choppy, <38.2 = trending). Returns {"chop"}."""
    n = len(df)
    if ta is None or n < period + 1:
        return {"chop": _nan_array(n)}
    result = ta.chop(df["high"], df["low"], df["close"], length=period)
    if result is None:
        return {"chop": _nan_array(n)}
    return {"chop": result.to_numpy(dtype=float)}
