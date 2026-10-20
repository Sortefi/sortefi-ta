import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array


def calc_range(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """high - low per candle. Returns {"range": ndarray}."""
    n = len(df)
    if n < 1:
        return {"range": _nan_array(n)}
    return {"range": df["high"].to_numpy(dtype=float) - df["low"].to_numpy(dtype=float)}


def calc_range_pct(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """(high - low) / low * 100. Returns {"range_pct": ndarray}."""
    n = len(df)
    if n < 1:
        return {"range_pct": _nan_array(n)}
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    return {"range_pct": np.where(l != 0, (h - l) / l * 100.0, np.nan)}
