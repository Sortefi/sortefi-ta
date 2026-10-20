import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array


def calc_yz_volatility(df: pd.DataFrame, period: int = 30, annual: int = 365) -> dict[str, np.ndarray]:
    """Yang-Zhang Volatility — combines overnight and intraday components.

    Most accurate volatility estimator for equity data with overnight gaps.
    Three components:
      1. Overnight volatility (close-to-open)
      2. Open-to-close volatility
      3. Rogers-Satchell (uses OHLC, drift-independent)

    YZ = sqrt(overnight_var + k * open_close_var + (1-k) * rogers_satchell_var)
    where k = 0.34 / (1.34 + (n+1)/(n-1))

    Returns {"yz_vol": ndarray} — annualized, in percentage.
    """
    n = len(df)
    if n < period + 1:
        return {"yz_vol": _nan_array(n)}

    # Log returns
    log_oc = np.log(df["close"] / df["open"])
    log_co = np.log(df["open"] / df["close"].shift(1))  # overnight
    log_hc = np.log(df["high"] / df["close"])
    log_ho = np.log(df["high"] / df["open"])
    log_lc = np.log(df["low"] / df["close"])
    log_lo = np.log(df["low"] / df["open"])

    # Rogers-Satchell variance (drift-independent)
    rs = log_ho * log_hc + log_lo * log_lc

    # Rolling variances
    k = 0.34 / (1.34 + (period + 1) / (period - 1))
    overnight_var = log_co.rolling(period).var(ddof=1)
    oc_var = log_oc.rolling(period).var(ddof=1)
    rs_var = rs.rolling(period).mean()

    yz_var = overnight_var + k * oc_var + (1 - k) * rs_var
    # Clamp to zero before sqrt (numerical noise can make it slightly negative)
    yz_var = yz_var.clip(lower=0)
    yz_vol = np.sqrt(yz_var) * np.sqrt(annual) * 100
    return {"yz_vol": yz_vol.to_numpy(dtype=float)}
