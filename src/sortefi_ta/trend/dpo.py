import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array


def calc_dpo(
    df: pd.DataFrame, period: int = 21, centered: bool = False, col: str = "close"
) -> dict[str, np.ndarray]:
    """Detrended Price Oscillator. Matches TradingView.
    Non-centered (default): dpo = close - SMA(close, period)[barsback bars ago]
    Centered:               dpo = SMA(close, period) - close[barsback bars ago]
    barsback = period // 2 + 1.
    Returns {"dpo"}."""
    n = len(df)
    barsback = period // 2 + 1
    if n < period + barsback:
        return {"dpo": _nan_array(n)}
    ma = df[col].rolling(period).mean()
    if centered:
        dpo = ma - df[col].shift(barsback)
    else:
        dpo = df[col] - ma.shift(barsback)
    return {"dpo": dpo.to_numpy(dtype=float)}
