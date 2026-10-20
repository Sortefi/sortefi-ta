import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_uo(
    df: pd.DataFrame, fast: int = 7, medium: int = 14, slow: int = 28
) -> dict[str, np.ndarray]:
    """Ultimate Oscillator. Returns {"uo"}."""
    n = len(df)
    if ta is None or n < slow + 1:
        return {"uo": _nan_array(n)}
    result = ta.uo(df["high"], df["low"], df["close"], fast=fast, medium=medium, slow=slow)
    if result is None:
        return {"uo": _nan_array(n)}
    return {"uo": result.to_numpy(dtype=float)}
