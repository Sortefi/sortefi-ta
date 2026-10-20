import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_kdj(
    df: pd.DataFrame, period: int = 9, signal: int = 3
) -> dict[str, np.ndarray]:
    """KDJ Oscillator. Returns {"kdj_k", "kdj_d", "kdj_j"}."""
    n = len(df)
    empty = {"kdj_k": _nan_array(n), "kdj_d": _nan_array(n), "kdj_j": _nan_array(n)}
    if ta is None or n < period + signal:
        return empty
    result = ta.kdj(df["high"], df["low"], df["close"], length=period, signal=signal)
    if result is None or (hasattr(result, "empty") and result.empty):
        return empty
    return {
        "kdj_k": result.iloc[:, 0].to_numpy(dtype=float),
        "kdj_d": result.iloc[:, 1].to_numpy(dtype=float),
        "kdj_j": result.iloc[:, 2].to_numpy(dtype=float) if result.shape[1] > 2 else _nan_array(n),
    }
