import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_williams_r(df: pd.DataFrame, period: int = 14) -> dict[str, np.ndarray]:
    """Williams %R. Returns {"willr"}."""
    n = len(df)
    if ta is None or n < period:
        return {"willr": _nan_array(n)}
    result = ta.willr(df["high"], df["low"], df["close"], length=period)
    if result is None:
        return {"willr": _nan_array(n)}
    return {"willr": result.to_numpy(dtype=float)}
