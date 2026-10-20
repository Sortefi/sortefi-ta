import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array


def calc_vol_ratio(
    df: pd.DataFrame, short_period: int = 10, long_period: int = 60
) -> dict[str, np.ndarray]:
    """Volatility Ratio — short-term HV / long-term HV.

    Measures whether volatility is expanding or contracting:
    - > 1.0 = vol expanding (short-term vol higher than long-term)
    - < 1.0 = vol contracting (compression, potential breakout setup)
    - ~ 1.0 = stable vol regime

    Both HV values are close-to-close annualized standard deviation.

    Returns {"vol_ratio": ndarray}.
    """
    n = len(df)
    if n < long_period + 1:
        return {"vol_ratio": _nan_array(n)}

    log_ret = np.log(df["close"] / df["close"].shift(1))
    hv_short = log_ret.rolling(short_period).std(ddof=1)
    hv_long = log_ret.rolling(long_period).std(ddof=1)

    # Avoid division by zero
    ratio = np.where(hv_long > 0, hv_short / hv_long, np.nan)
    out = np.full(n, np.nan)
    out[: len(ratio)] = ratio
    return {"vol_ratio": out.astype(float)}
