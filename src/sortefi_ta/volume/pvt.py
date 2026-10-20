import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array


def calc_pvt(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Price Volume Trend. TV formula: cum(change(close)/close[1] * volume).
    pandas_ta multiplies pct change by 100 — replaced with direct formula.
    Returns {"pvt": ndarray}.
    """
    n = len(df)
    if n < 2:
        return {"pvt": _nan_array(n)}

    close = df["close"].to_numpy(dtype=float)
    vol = df["volume"].to_numpy(dtype=float)

    pct_change = np.empty(n)
    pct_change[0] = 0.0
    pct_change[1:] = (close[1:] - close[:-1]) / close[:-1]

    pvt = np.nancumsum(pct_change * vol)
    return {"pvt": pvt}
