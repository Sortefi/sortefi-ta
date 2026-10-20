import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_adx(df: pd.DataFrame, period: int = 14) -> dict[str, np.ndarray]:
    """ADX with +DI/-DI (TradingView mode). Returns {"adx", "di_plus", "di_minus"}."""
    n = len(df)
    empty = {"adx": _nan_array(n), "di_plus": _nan_array(n), "di_minus": _nan_array(n)}
    if ta is None or n < period * 2:
        return empty
    result = ta.adx(df["high"], df["low"], df["close"], length=period, tvmode=True)
    if result is None or result.empty:
        return empty
    adx_col  = f"ADX_{period}"
    dmp_col  = f"DMP_{period}"
    dmn_col  = f"DMN_{period}"
    if adx_col not in result.columns:
        return empty
    return {
        "adx":      result[adx_col].to_numpy(dtype=float),
        "di_plus":  result[dmp_col].to_numpy(dtype=float),
        "di_minus": result[dmn_col].to_numpy(dtype=float),
    }
