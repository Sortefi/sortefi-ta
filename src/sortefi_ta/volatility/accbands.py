import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_accbands(df: pd.DataFrame, period: int = 20) -> dict[str, np.ndarray]:
    """Acceleration Bands. Returns {"accband_upper", "accband_middle", "accband_lower"}."""
    n = len(df)
    empty = {"accband_upper": _nan_array(n), "accband_middle": _nan_array(n), "accband_lower": _nan_array(n)}
    if ta is None or n < period:
        return empty
    result = ta.accbands(df["high"], df["low"], df["close"], length=period)
    if result is None or (hasattr(result, "empty") and result.empty):
        return empty
    return {
        "accband_lower": result.iloc[:, 0].to_numpy(dtype=float),
        "accband_middle": result.iloc[:, 1].to_numpy(dtype=float),
        "accband_upper": result.iloc[:, 2].to_numpy(dtype=float),
    }
