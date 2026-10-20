import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average with alpha = 2/(period+1)."""
    alpha = 2.0 / (period + 1)
    out = np.full(len(arr), np.nan)
    # seed: first non-nan value
    start = 0
    while start < len(arr) and np.isnan(arr[start]):
        start += 1
    if start >= len(arr):
        return out
    out[start] = arr[start]
    for i in range(start + 1, len(arr)):
        if np.isnan(arr[i]):
            out[i] = np.nan
        else:
            out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def calc_trix(
    df: pd.DataFrame, period: int = 18, signal: int = 9, col: str = "close"
) -> dict[str, np.ndarray]:
    """TRIX matching TradingView's formula exactly.

    TV formula:
        out = 10000 * ta.change(ta.ema(ta.ema(ta.ema(math.log(close), length), length), length))
        signal = ta.sma(out, signal_length)   [TV default uses SMA for signal]

    Uses natural log of close before triple EMA, scalar 10000, signal = SMA of trix.
    """
    n = len(df)
    empty = {"trix": _nan_array(n), "trix_signal": _nan_array(n)}
    if n < period * 3 + signal:
        return empty

    close = df[col].to_numpy(dtype=float)
    log_close = np.log(close)

    ema1 = _ema(log_close, period)
    ema2 = _ema(ema1, period)
    ema3 = _ema(ema2, period)

    trix = np.full(n, np.nan)
    trix[1:] = 10000.0 * np.diff(ema3)  # change = ema3[i] - ema3[i-1]

    # signal = SMA of trix (TV default)
    trix_signal = np.full(n, np.nan)
    for i in range(n):
        window_start = i - signal + 1
        if window_start >= 0:
            window = trix[window_start: i + 1]
            if not np.any(np.isnan(window)):
                trix_signal[i] = window.mean()

    return {"trix": trix, "trix_signal": trix_signal}
