import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array


def calc_realized_vol(df: pd.DataFrame, period: int = 30, annual: int = 365) -> dict[str, np.ndarray]:
    """Realized Volatility — sum of squared log returns over a rolling window.

    Unlike HV (which uses std dev), realized vol directly sums squared returns.
    More responsive to recent price moves. Common in academic finance and
    risk management.

    RV = sqrt(sum(r_i^2) * annual/period) * 100

    Returns {"realized_vol": ndarray} — annualized, in percentage.
    """
    n = len(df)
    if n < period + 1:
        return {"realized_vol": _nan_array(n)}
    log_ret = np.log(df["close"] / df["close"].shift(1))
    sq_ret = log_ret ** 2
    rolling_sum = sq_ret.rolling(period).sum()
    rv = np.sqrt(rolling_sum * annual / period) * 100
    return {"realized_vol": rv.to_numpy(dtype=float)}
