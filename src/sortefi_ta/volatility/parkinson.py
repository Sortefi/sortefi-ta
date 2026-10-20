import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array


def calc_parkinson_vol(df: pd.DataFrame, period: int = 30, annual: int = 365) -> dict[str, np.ndarray]:
    """Parkinson Volatility — uses high-low range instead of close-to-close.

    More efficient estimator than close-to-close HV because it captures
    intraday price movement. Factor: 1/(4*ln(2)).

    Returns {"parkinson_vol": ndarray} — annualized, in percentage.
    """
    n = len(df)
    if n < period:
        return {"parkinson_vol": _nan_array(n)}
    hl_log = np.log(df["high"] / df["low"])
    hl_sq = hl_log ** 2
    factor = 1.0 / (4.0 * np.log(2.0))
    rolling_mean = hl_sq.rolling(period).mean()
    parkinson = np.sqrt(factor * rolling_mean) * np.sqrt(annual) * 100
    return {"parkinson_vol": parkinson.to_numpy(dtype=float)}
