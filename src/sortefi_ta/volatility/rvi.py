import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_rvi(df: pd.DataFrame, stdev_length: int = 10, ema_length: int = 14) -> dict[str, np.ndarray]:
    """Relative Volatility Index. Matches TV formula: stdev(close,10,ddof=0) split by direction, EMA(14).
    Returns {"rvi"}."""
    n = len(df)
    if n < max(stdev_length, ema_length) + 1:
        return {"rvi": _nan_array(n)}
    std = df["close"].rolling(stdev_length).std(ddof=0)
    change = df["close"].diff()
    upper = std.where(change > 0, 0.0).ewm(span=ema_length, adjust=False).mean()
    lower = std.where(change <= 0, 0.0).ewm(span=ema_length, adjust=False).mean()
    rvi = upper / (upper + lower) * 100
    return {"rvi": rvi.to_numpy(dtype=float)}
