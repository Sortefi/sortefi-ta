import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array


def calc_vol_percentile(
    df: pd.DataFrame, hv_period: int = 30, rank_period: int = 252, annual: int = 365
) -> dict[str, np.ndarray]:
    """Volatility Percentile — where current HV sits relative to its own history.

    Computes rolling HV (close-to-close, annualized), then ranks the current
    value against the last `rank_period` HV values as a percentile (0-100).

    - 90+ = unusually high vol for this symbol (fear, event)
    - 10-  = unusually low vol (complacency, potential squeeze)

    Returns {"vol_percentile": ndarray} — 0 to 100.
    """
    n = len(df)
    if n < hv_period + rank_period + 1:
        return {"vol_percentile": _nan_array(n)}

    log_ret = np.log(df["close"] / df["close"].shift(1))
    hv = log_ret.rolling(hv_period).std(ddof=1) * np.sqrt(annual)

    # Minimum rank minus one counts strictly smaller values (ties counted as
    # not smaller). The current value is not in the
    # denominator: compare it with the preceding rank_period observations.
    vol_pct = (hv.rolling(rank_period + 1).rank(method="min") - 1) / rank_period * 100
    return {"vol_percentile": vol_pct.to_numpy(dtype=float)}
