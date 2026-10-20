import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_donchian(df: pd.DataFrame, period: int = 20) -> dict[str, np.ndarray]:
    """Donchian Channels. Returns {"dc_upper", "dc_middle", "dc_lower"}."""
    n = len(df)
    empty = {"dc_upper": _nan_array(n), "dc_middle": _nan_array(n), "dc_lower": _nan_array(n)}
    if ta is None or n < period:
        return empty
    result = ta.donchian(df["high"], df["low"], lower_length=period, upper_length=period)
    if result is None or result.empty:
        return empty
    return {
        "dc_lower": result.iloc[:, 0].to_numpy(dtype=float),
        "dc_middle": result.iloc[:, 1].to_numpy(dtype=float),
        "dc_upper": result.iloc[:, 2].to_numpy(dtype=float),
    }
