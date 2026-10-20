import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array


def calc_coppock(
    df: pd.DataFrame,
    wma_length: int = 10,
    long_roc: int = 14,
    short_roc: int = 11,
) -> dict[str, np.ndarray]:
    """
    Coppock Curve — momentum oscillator, WMA of sum of two ROC periods.

    Originally designed to identify long-term buying opportunities.
    Bullish when curve crosses above zero from below.

    Args:
        df:         DataFrame with 'close' column.
        wma_length: WMA smoothing period (default 10).
        long_roc:   Long Rate of Change period (default 14).
        short_roc:  Short Rate of Change period (default 11).

    Returns:
        coppock — Coppock Curve values (ndarray).
    """
    n = len(df)
    if n < 1:
        return {"coppock": _nan_array(n)}

    close = df["close"].to_numpy(dtype=float)

    # ROC = (close - close[n_ago]) / close[n_ago] * 100
    roc_long = _roc(close, long_roc)
    roc_short = _roc(close, short_roc)

    roc_sum = roc_long + roc_short

    # WMA of the sum
    coppock = _wma(roc_sum, wma_length)

    return {"coppock": coppock}


def _roc(arr: np.ndarray, period: int) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(period, n):
        prev = arr[i - period]
        if prev != 0 and np.isfinite(prev) and np.isfinite(arr[i]):
            out[i] = (arr[i] - prev) / prev * 100.0
    return out


def _wma(arr: np.ndarray, period: int) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan)
    weights = np.arange(1, period + 1, dtype=float)
    w_sum = weights.sum()
    for i in range(period - 1, n):
        window = arr[i - period + 1 : i + 1]
        if np.all(np.isfinite(window)):
            out[i] = np.dot(window, weights) / w_sum
    return out
