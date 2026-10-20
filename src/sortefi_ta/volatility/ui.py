import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_ui(df: pd.DataFrame, period: int = 14) -> dict[str, np.ndarray]:
    """Ulcer Index (downside volatility). Returns {"ui"}."""
    n = len(df)
    if ta is None or n < period:
        return {"ui": _nan_array(n)}
    result = ta.ui(df["close"], length=period)
    if result is None:
        return {"ui": _nan_array(n)}
    return {"ui": result.to_numpy(dtype=float)}
