import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_tsi(
    df: pd.DataFrame, fast: int = 13, slow: int = 25, signal: int = 13, col: str = "close"
) -> dict[str, np.ndarray]:
    """True Strength Index. Returns {"tsi", "tsi_signal"}."""
    n = len(df)
    empty = {"tsi": _nan_array(n), "tsi_signal": _nan_array(n)}
    if ta is None or n < slow + fast:
        return empty
    result = ta.tsi(df[col], fast=fast, slow=slow, signal=signal)
    if result is None or (hasattr(result, "empty") and result.empty):
        return empty
    if isinstance(result, pd.DataFrame):
        return {
            "tsi": result.iloc[:, 0].to_numpy(dtype=float),
            "tsi_signal": result.iloc[:, 1].to_numpy(dtype=float) if result.shape[1] > 1 else _nan_array(n),
        }
    return {"tsi": result.to_numpy(dtype=float), "tsi_signal": _nan_array(n)}
