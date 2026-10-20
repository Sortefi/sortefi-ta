import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array


def calc_gk_volatility(df: pd.DataFrame, period: int = 30, annual: int = 365) -> dict[str, np.ndarray]:
    """Garman-Klass Volatility — uses OHLC data for optimal estimation.

    Best volatility estimator from candlestick data. More efficient than
    close-to-close (HV) and high-low (Parkinson) because it uses all four
    price points: open, high, low, close.

    GK = 0.5 * ln(H/L)^2 - (2*ln(2) - 1) * ln(C/O)^2

    Returns {"gk_vol": ndarray} — annualized, in percentage.
    """
    n = len(df)
    if n < period:
        return {"gk_vol": _nan_array(n)}
    hl = np.log(df["high"] / df["low"]) ** 2
    co = np.log(df["close"] / df["open"]) ** 2
    gk = 0.5 * hl - (2.0 * np.log(2.0) - 1.0) * co
    rolling_mean = gk.rolling(period).mean()
    gk_vol = np.sqrt(rolling_mean) * np.sqrt(annual) * 100
    return {"gk_vol": gk_vol.to_numpy(dtype=float)}
