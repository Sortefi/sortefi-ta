import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array


def calc_hv(df: pd.DataFrame, period: int = 30, annual: int = 365) -> dict[str, np.ndarray]:
    """Historical Volatility — annualized std dev of log returns (close-to-close).

    Matches TradingView's HV indicator: annual=365 for all markets.

    Returns {"hv": ndarray} — values in percentage (e.g. 45.0 = 45%).
    """
    n = len(df)
    if n < period + 1:
        return {"hv": _nan_array(n)}
    log_ret = np.log(df["close"] / df["close"].shift(1))
    rolling_std = log_ret.rolling(period).std(ddof=1)
    hv = rolling_std * np.sqrt(annual) * 100
    return {"hv": hv.to_numpy(dtype=float)}
