import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array


def calc_smma(df: pd.DataFrame, period: int = 7, col: str = "close") -> dict[str, np.ndarray]:
    """Smoothed Moving Average (SMMA / RMA / Wilder's Smoothing).

    Ports TradingView's Pine Script exactly:
      smma[0] = sma(src, len)          # seed with SMA on first valid bar
      smma[i] = (smma[i-1] * (len-1) + src[i]) / len

    Same smoothing used internally by RSI and ATR. Returns {"smma": ndarray}.
    """
    n = len(df)
    if n < period:
        return {"smma": _nan_array(n)}

    src = df[col].to_numpy(dtype=float)
    out = np.full(n, np.nan)

    # Seed: first valid value is SMA of first `period` bars
    seed_idx = period - 1
    seed = np.nanmean(src[:period])
    if np.isnan(seed):
        return {"smma": out}

    out[seed_idx] = seed
    for i in range(seed_idx + 1, n):
        if np.isnan(src[i]):
            out[i] = out[i - 1]
        else:
            out[i] = (out[i - 1] * (period - 1) + src[i]) / period

    return {"smma": out}
