import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array


def calc_relative_position(
    df: pd.DataFrame, period: int = 20
) -> dict[str, np.ndarray]:
    """Where close sits in H/L range: (close - rolling_low) / (rolling_high - rolling_low).
    Returns {"rel_pos": ndarray} (0=at low, 1=at high)."""
    n = len(df)
    if n < period:
        return {"rel_pos": _nan_array(n)}
    c = df["close"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    hh = pd.Series(h).rolling(period).max().to_numpy()
    ll = pd.Series(l).rolling(period).min().to_numpy()
    rng = hh - ll
    pos = np.where(rng != 0, (c - ll) / rng, np.nan)
    return {"rel_pos": pos}


def calc_cumulative_return(
    df: pd.DataFrame, period: int | None = None
) -> dict[str, np.ndarray]:
    """Cumulative % return over N candles (or full window).
    Returns {"cum_return": ndarray}."""
    n = len(df)
    if n < 2:
        return {"cum_return": _nan_array(n)}
    c = df["close"].to_numpy(dtype=float)
    if period is None:
        base = c[0]
        cum_ret = np.where(base != 0, (c - base) / base * 100.0, np.nan)
    else:
        cum_ret = _nan_array(n)
        for i in range(period, n):
            base = c[i - period]
            if base != 0:
                cum_ret[i] = (c[i] - base) / base * 100.0
    return {"cum_return": cum_ret}


def calc_avg_daily_range(
    df: pd.DataFrame, period: int = 20
) -> dict[str, np.ndarray]:
    """Rolling average of (high - low). Returns {"avg_range": ndarray}."""
    n = len(df)
    if n < period:
        return {"avg_range": _nan_array(n)}
    rng = df["high"].to_numpy(dtype=float) - df["low"].to_numpy(dtype=float)
    avg_range = pd.Series(rng).rolling(period).mean().to_numpy()
    return {"avg_range": avg_range}
