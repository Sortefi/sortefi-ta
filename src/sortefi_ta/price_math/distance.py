import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array


def calc_distance_from_high(
    df: pd.DataFrame, period: int = 20
) -> dict[str, np.ndarray]:
    """(close - highest_high) / highest_high * 100 over rolling period.
    Returns {"dist_high": ndarray}."""
    n = len(df)
    if n < period:
        return {"dist_high": _nan_array(n)}
    c = df["close"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    hh = pd.Series(h).rolling(period).max().to_numpy()
    dist = np.where(hh != 0, (c - hh) / hh * 100.0, np.nan)
    return {"dist_high": dist}


def calc_distance_from_low(
    df: pd.DataFrame, period: int = 20
) -> dict[str, np.ndarray]:
    """(close - lowest_low) / lowest_low * 100 over rolling period.
    Returns {"dist_low": ndarray}."""
    n = len(df)
    if n < period:
        return {"dist_low": _nan_array(n)}
    c = df["close"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    ll = pd.Series(l).rolling(period).min().to_numpy()
    dist = np.where(ll != 0, (c - ll) / ll * 100.0, np.nan)
    return {"dist_low": dist}
