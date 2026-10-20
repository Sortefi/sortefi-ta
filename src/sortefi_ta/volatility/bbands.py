import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_bbands(
    df: pd.DataFrame, period: int = 20, std: float = 2.0, col: str = "close",
    squeeze_length: int = 125,
) -> dict[str, np.ndarray]:
    """Bollinger Bands (population stdev, ddof=0). Matches TV BB formula.
    Returns {"bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_pct_b",
             "bb_width_high", "bb_width_low"}.
    bb_width_high/low = rolling squeeze_length max/min of bb_width (TV BBW expansion/contraction lines).
    """
    n = len(df)
    empty = {
        "bb_upper": _nan_array(n), "bb_middle": _nan_array(n), "bb_lower": _nan_array(n),
        "bb_width": _nan_array(n), "bb_pct_b": _nan_array(n),
        "bb_width_high": _nan_array(n), "bb_width_low": _nan_array(n),
    }
    if n < period:
        return empty
    mid   = df[col].rolling(period).mean()
    sigma = df[col].rolling(period).std(ddof=0)
    upper = mid + std * sigma
    lower = mid - std * sigma
    width = (upper - lower) / mid * 100
    pct_b = (df[col] - lower) / (upper - lower)
    return {
        "bb_upper":      upper.to_numpy(dtype=float),
        "bb_middle":     mid.to_numpy(dtype=float),
        "bb_lower":      lower.to_numpy(dtype=float),
        "bb_width":      width.to_numpy(dtype=float),
        "bb_pct_b":      pct_b.to_numpy(dtype=float),
        "bb_width_high": width.rolling(squeeze_length).max().to_numpy(dtype=float),
        "bb_width_low":  width.rolling(squeeze_length).min().to_numpy(dtype=float),
    }
