import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array


def calc_ssl_channel(df: pd.DataFrame, period: int = 10) -> dict[str, np.ndarray]:
    """
    SSL Channel — trend indicator based on SMA of high/low crossover.

    When close > SMA(high), trend is up. When close < SMA(low), trend is down.
    SSL Up (green) and SSL Down (red) swap positions based on trend direction.

    Args:
        df:     DataFrame with 'high', 'low', 'close' columns.
        period: SMA period (default 10).

    Returns:
        ssl_up   — Upper SSL line (ndarray)
        ssl_down — Lower SSL line (ndarray)
        direction — 1.0 = bullish, -1.0 = bearish (ndarray)
    """
    n = len(df)
    if n < period:
        return {
            "ssl_up": _nan_array(n),
            "ssl_down": _nan_array(n),
            "direction": _nan_array(n),
        }

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    # SMA of high and low
    sma_high = _sma(high, period)
    sma_low = _sma(low, period)

    # Hlv state: 1 if close > sma_high, -1 if close < sma_low, else previous
    hlv = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isfinite(sma_high[i]) or not np.isfinite(close[i]):
            continue
        if close[i] > sma_high[i]:
            hlv[i] = 1.0
        elif close[i] < sma_low[i]:
            hlv[i] = -1.0
        elif i > 0 and np.isfinite(hlv[i - 1]):
            hlv[i] = hlv[i - 1]

    ssl_down = np.full(n, np.nan)
    ssl_up = np.full(n, np.nan)
    direction = np.full(n, np.nan)

    for i in range(n):
        if not np.isfinite(hlv[i]):
            continue
        if hlv[i] < 0:
            ssl_down[i] = sma_high[i]
            ssl_up[i] = sma_low[i]
        else:
            ssl_down[i] = sma_low[i]
            ssl_up[i] = sma_high[i]
        direction[i] = hlv[i]

    return {"ssl_up": ssl_up, "ssl_down": ssl_down, "direction": direction}


def _sma(arr: np.ndarray, period: int) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan)
    cumsum = np.nancumsum(arr)
    out[period - 1] = cumsum[period - 1] / period
    for i in range(period, n):
        out[i] = (cumsum[i] - cumsum[i - period]) / period
    return out
