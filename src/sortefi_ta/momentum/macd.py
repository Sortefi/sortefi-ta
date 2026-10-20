import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_macd(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> dict[str, np.ndarray]:
    """MACD. Returns {"macd", "signal", "histogram"}."""
    n = len(df)
    empty = {"macd": _nan_array(n), "signal": _nan_array(n), "histogram": _nan_array(n)}
    if ta is None or n < slow + signal:
        return empty
    result = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)
    if result is None or result.empty:
        return empty
    # columns returned: MACD_f_s_sig, MACDh_f_s_sig, MACDs_f_s_sig
    # order is MACD, Histogram, Signal — use names to avoid positional swap
    cols = result.columns.tolist()
    return {
        "macd":      result[cols[0]].to_numpy(dtype=float),
        "signal":    result[cols[2]].to_numpy(dtype=float),
        "histogram": result[cols[1]].to_numpy(dtype=float),
    }
