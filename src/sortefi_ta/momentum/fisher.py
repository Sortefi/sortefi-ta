import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_fisher(df: pd.DataFrame, period: int = 9) -> dict[str, np.ndarray]:
    """Fisher Transform. Returns {"fisher", "fisher_signal"}."""
    n = len(df)
    empty = {"fisher": _nan_array(n), "fisher_signal": _nan_array(n)}
    if ta is None or n < period:
        return empty
    result = ta.fisher(df["high"], df["low"], length=period)
    if result is None or (hasattr(result, "empty") and result.empty):
        return empty
    if isinstance(result, pd.DataFrame):
        return {
            "fisher": result.iloc[:, 0].to_numpy(dtype=float),
            "fisher_signal": result.iloc[:, 1].to_numpy(dtype=float) if result.shape[1] > 1 else _nan_array(n),
        }
    return {"fisher": result.to_numpy(dtype=float), "fisher_signal": _nan_array(n)}
